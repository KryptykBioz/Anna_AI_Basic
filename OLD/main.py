import os
import re
import pickle
import threading
import queue
import base64
import json
import time
from io import BytesIO
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Optional

import pyautogui
import requests
import sounddevice as sd
from vosk import KaldiRecognizer
from colorama import Fore
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from text_to_voice import speak_through_vbcable
from voice_to_text import load_vosk_model
from search_agent import maybe_fetch_articles

# --- Configuration ---
PROMPT_TIMEOUT = 90  # seconds before auto-prompting
VISION_KEYWORDS = ['screen', 'image', 'see', 'look', 'game']
MAX_VECTORS = 1000

@dataclass
class Config:
    llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:3b-instruct-q4_K_M")
    vision_endpoint: str = os.getenv("VISION_MODEL_ENDPOINT", "http://localhost:11434/api/generate")
    vision_model: str = os.getenv("VISION_MODEL_NAME", "llava:7b-v1.5-q4_K_M")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./long_term_store")
    past_convo_file: str = os.getenv("PAST_CONVO_FILE", "pastconvo.txt")
    summary_trigger: int = int(os.getenv("SUMMARY_TRIGGER", "20"))
    backup_file: str = os.getenv("BACKUP_FILE", "chat_backup.pkl")
    samplerate: int = 16000
    system_prompt: str = (
        "You are Anna: a helpful assistant. "
        "You have access to internet search, voice, an avatar, and a vision model. "
        "Keep replies 3-4 short sentences unless more detail requested. "
        "Always respond starting with 'I'. No fabrications; speak in first person. "
        "If provided an image, use it as a reference without describing it."
    )

config = Config()

class AssistantAI:
    def __init__(self):
        self._init_audio()
        self._init_models()
        self.msg_buffer: List[str] = []
        self.history: List[dict] = []
        self.speaking_thread: Optional[threading.Thread] = None
        self.processing = False
        self.last_interaction = time.time()

    def _init_audio(self):
        sd.default.device = (1, None)
        self.raw_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.vosk_model = load_vosk_model()
        self._start_vosk()

    def _start_vosk(self):
        def worker():
            rec = KaldiRecognizer(self.vosk_model, config.samplerate)
            while True:
                data = self.raw_queue.get()
                if data == b"__EXIT__":
                    break
                if rec.AcceptWaveform(data):
                    text = json.loads(rec.Result()).get("text", "").strip()
                    if len(text) >= 5 and "Anna" not in text.lower():
                        self.text_queue.put(text)
        threading.Thread(target=worker, daemon=True).start()

        def callback(indata, frames, time_info, status):
            if status:
                print(Fore.RED + f"[Audio status]: {status}" + Fore.RESET)
            self.raw_queue.put(bytes(indata))

        self.stream = sd.RawInputStream(
            samplerate=config.samplerate,
            blocksize=4096,
            dtype='int16',
            channels=1,
            callback=callback
        )
        self.stream.start()
        print(Fore.CYAN + "[Listener] Audio stream started." + Fore.RESET)

    def _init_models(self):
        self.llm = OllamaLLM(model=config.llm_model, temperature=0.7, max_new_tokens=250)
        # Updated PromptTemplate includes search_results and vision_context
        template = (
            "{system_prompt}\n\n"
            "[Chat History]\n{memory}\n\n"
            "{search_results_section}"
            "{vision_section}"
            "[User Input]\n{user_input}\n"
            "Anna:"
        )
        self.chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=[
                    "system_prompt",
                    "memory",
                    "search_results_section",
                    "vision_section",
                    "user_input"
                ],
                template=template
            )
        )
        self.embeddings = OllamaEmbeddings(model=config.embed_model)
        self.vector_store = self._load_or_create_vector_store()

    def _load_or_create_vector_store(self) -> FAISS:
        path = config.vector_store_path
        flag_file = os.path.join(path, ".initialized")

        # Ensure the vector‐store directory exists so our sentinel check will work
        os.makedirs(path, exist_ok=True)

        # If we've already initialized once, just reload from disk
        if os.path.exists(flag_file):
            try:
                # pass embeddings as positional arg, and enable pickle deserialization
                return FAISS.load_local(
                    path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(Fore.YELLOW + f"[Warning] failed loading FAISS store ({e}), rebuilding…" + Fore.RESET)

        # First‐time initialization (or load truly failed): read past conversations
        docs: List[Document] = []
        if os.path.exists(config.past_convo_file):
            with open(config.past_convo_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    m = re.search(r"\((.*?)\)$", line)
                    timestamp = m.group(1) if m else datetime.now().isoformat()
                    content = re.sub(r"\s*\(.*?\)$", "", line)
                    docs.append(Document(page_content=content, metadata={"timestamp": timestamp}))

        # Build the FAISS index and save it locally
        store = FAISS.from_documents(docs, self.embeddings)
        store.save_local(path)

        # Write our sentinel so next start just reloads
        with open(flag_file, "w", encoding="utf-8") as f:
            f.write(f"initialized: {datetime.now().isoformat()}")

        return store

    def _retrieve_memory(self, query: str, k: int = 3) -> str:
        hits = self.vector_store.similarity_search(query, k=k) if self.vector_store else []
        return "\n\n".join(f"- {d.page_content}" for d in hits)

    def _summarize_and_store(self):
        summarizer = OllamaLLM(model="mistral:7b-instruct-q4_K_M", temperature=0.7, max_new_tokens=250)
        prompt = ("You are Anna in the following conversation. Summarize the conversation from the perspective of Anna. "
                  "Create a one paragraph summary and include all memorable details from the conversation.\n" + "\n".join(self.msg_buffer))
        summary = summarizer.invoke(prompt)
        print("Summary: " + summary)
        doc = Document(page_content=summary, metadata={"timestamp": datetime.now().isoformat()})
        self.vector_store.add_documents([doc])
        self._prune_faiss_index()
        self.vector_store.save_local(config.vector_store_path)
        self.msg_buffer.clear()

    def _prune_faiss_index(self):
        if not hasattr(self.vector_store, 'docstore'):
            return
        items = list(self.vector_store.docstore._dict.items())
        if len(items) <= MAX_VECTORS:
            return
        items.sort(key=lambda x: datetime.fromisoformat(x[1].metadata.get("timestamp", "")))
        keep = items[-MAX_VECTORS:]
        docs = [doc for _, doc in keep]
        self.vector_store = FAISS.from_documents(docs, self.vector_store.embedding_function)

    def _capture_screenshot(self) -> str:
        img = pyautogui.screenshot()
        buf = BytesIO(); img.save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _vision_query(self, prompt: str, img_b64: str) -> str:
        resp = requests.post(
            config.vision_endpoint,
            json={"model": config.vision_model, "prompt": prompt, "images": [img_b64], "stream": False}
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _backup(self, role: str, content: str):
        entry = {"role": role, "content": content, "timestamp": datetime.now(timezone.utc).isoformat()}
        logs = pickle.load(open(config.backup_file, 'rb')) if os.path.exists(config.backup_file) else []
        logs.append(entry)
        pickle.dump(logs, open(config.backup_file, 'wb'))

    def _prepare_prompt(self,
                        user_input: str,
                        search_ctx: Optional[str] = None,
                        vision_ctx: Optional[str] = None
                        ) -> dict:
        memory = self._retrieve_memory(user_input)
        # Build search results section, always include header but empty if None
        if search_ctx:
            search_results_section = f"[Search Results]\n{search_ctx}\n\n"
        else:
            search_results_section = ""
        # Build vision section for game mode
        if vision_ctx:
            vision_section = f"[Visual Context]\n{vision_ctx}\n\n"
        else:
            vision_section = ""
        return {
            "system_prompt": config.system_prompt,
            "memory": memory,
            "search_results_section": search_results_section,
            "vision_section": vision_section,
            "user_input": user_input
        }

    def _handle_input(self, user_input: str, mode: str) -> str:
        vision_trigger = any(kw in user_input.lower() for kw in VISION_KEYWORDS)
        self.last_interaction = time.time()
        self.processing = True

        try:
            # --- 1. Fetch search results (for any mode) ---
            search_ctx = None
            articles = maybe_fetch_articles(user_input, max_results=3)
            if articles:
                snippets = []
                for i, art in enumerate(articles, 1):
                    snippet = art['text'][:1000] + ('…' if len(art['text']) > 1000 else '')
                    snippets.append(f"{i}. {art['title']} ({art['url']})\n{snippet}")
                search_ctx = "\n\n".join(snippets)
                print(Fore.LIGHTYELLOW_EX + "SEARCH RESULTS:\n" + search_ctx + Fore.RESET)

            # --- 2. Capture vision context if in game mode ---
            vision_ctx = None
            if mode == 'game' and vision_trigger:
                print(Fore.CYAN + "[Vision] Capturing screenshot..." + Fore.RESET)
                img_b64 = self._capture_screenshot()
                # Here we're passing raw base64; you may choose to decode/describe it instead
                vision_ctx = img_b64
                print(Fore.CYAN + "[Vision] Image captured for context." + Fore.RESET)

            # --- 3. Build prompt vars with all sections ---
            prompt_vars = self._prepare_prompt(
                user_input=user_input,
                search_ctx=search_ctx,
                vision_ctx=vision_ctx
            )

            # --- 4. Invoke the LLM chain ---
            reply = self.chain.run(**prompt_vars)

            # --- 5. Persist conversation turn to vector store ---
            timestamp = datetime.now().isoformat()
            new_docs = [
                Document(page_content=f"User: {user_input}", metadata={"timestamp": timestamp}),
                Document(page_content=f"Anna: {reply}",   metadata={"timestamp": timestamp})
            ]
            self.vector_store.add_documents(new_docs)
            self.vector_store.save_local(config.vector_store_path)

            return reply

        finally:
            self.processing = False


    def _drain_text(self) -> List[str]:
        texts = []
        while not self.text_queue.empty():
            texts.append(self.text_queue.get())
        return texts

    def _voice_loop(self, mode: str):
        while True:
            if (self.speaking_thread and self.speaking_thread.is_alive()) or self.processing:
                time.sleep(0.1)
                continue
            elapsed = time.time() - self.last_interaction
            queued = self._drain_text()
            user_text = " ".join(queued)
            if not user_text and elapsed < PROMPT_TIMEOUT:
                time.sleep(0.1)
                continue
            if not user_text:
                user_text = "Ask me a question or comment."
                self.last_interaction = time.time()
            print(Fore.GREEN + f"User: {user_text}" + Fore.RESET)
            if user_text.lower() == 'exit': break
            reply = self._handle_input(user_text, mode)
            print(Fore.MAGENTA + f"Anna: {reply}" + Fore.RESET)
            print(Fore.LIGHTYELLOW_EX + "Listening..." + Fore.RESET)
            t = threading.Thread(target=speak_through_vbcable, args=(reply,), daemon=True)
            t.start(); self.speaking_thread = t
            self.history.extend([{'role':'user','content':user_text},{'role':'assistant','content':reply}])
            self._backup('user', user_text); self._backup('assistant', reply)
            self.msg_buffer.append(f"User: {user_text}\nAnna: {reply}")
            if len(self.msg_buffer) >= config.summary_trigger:
                # self._summarize_and_store()
                print(Fore.YELLOW+"SHOULD SUMMARIZE"+Fore.RESET)

    def _text_loop(self):
        while True:
            user_in = input("User: ").strip()
            if user_in.lower() == 'exit': break
            reply = self._handle_input(user_in, mode='text')
            print(Fore.MAGENTA + f"Anna: {reply}" + Fore.RESET)

    def run(self):
        print("[INFO] Starting Bot. Type 'exit' to quit.")
        mode = input(Fore.GREEN + "Mode (talk/text/game): " + Fore.RESET).strip().lower()
        while mode not in ('talk','text','game'):
            mode = input("Invalid. Choose talk/text/game: ").strip().lower()
        try:
            if mode == 'text': self._text_loop()
            else: self._voice_loop(mode)
        except (KeyboardInterrupt, SystemExit): print(Fore.LIGHTYELLOW_EX + "\nGoodbye!" + Fore.RESET)
        finally:
            self.raw_queue.put(b"__EXIT__"); self.stream.stop()

if __name__ == '__main__':
    AssistantAI().run()
