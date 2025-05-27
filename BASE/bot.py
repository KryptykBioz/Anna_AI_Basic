import os
import sys
import threading
import queue
import base64
import json
import time
import asyncio
import requests
from io import BytesIO
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict, Any
import pyautogui
import sounddevice as sd
from vosk import KaldiRecognizer
from colorama import Fore
import socketio
from pathlib import Path
import numpy as np

project_root = Path(__file__).parent.parent  # Go up to Toma_AI directory
sys.path.insert(0, str(project_root))

from BASE.tools.text_to_voice import speak_through_vbcable
from BASE.tools.voice_to_text import load_vosk_model, init_audio, start_vosk_stream, audio_callback, listen_and_transcribe
from BASE.tools.query import web_search_summary
from BASE.memory_methods.summarizer import summarize_memory
from BASE.mindcraft_integration.mc_methods import init_mindserver, on_mind_msg

from personality.SYS_MSG import system_prompt
from personality.bot_info import botname, username, textmodel, visionmodel, embedmodel, botColor, userColor, systemColor, \
    botTColor, userTColor, systemTColor, toolTColor, errorTColor, resetTColor

# Resolve project root: two levels up from this file (BASE → Toma_AI)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Configuration ---
PROMPT_TIMEOUT = 120
VISION_KEYWORDS = ["screen", "image", "see", "look", "monitor"]
SEARCH_KEYWORDS = ["search", "find", "look up", "web", "internet"]
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", "500"))
MIND_SERVER_HOST = os.getenv("MINDSERVER_HOST", "localhost")
MIND_SERVER_PORT = os.getenv("MINDSERVER_PORT", "8080")
AGENT_NAME = os.getenv("MINDCRAFT_AGENT_NAME", f"{botname}")

# Memory file paths
MEMORY_FILE = PROJECT_ROOT / "personality" / "memory" / "memory.json"
EMBEDDINGS_FILE = PROJECT_ROOT / "personality" / "memory" / "embeddings.json"

# Colorize Terminal Outputs
redColor = "\033[91m"
greenColor = "\033[92m"
resetColor = "\033[0m"
yellowColor = "\033[93m"
magentaColor = "\033[95m"
cyanColor = "\033[96m"
blueColor = "\033[94m"


@dataclass
class Config:
    text_llm_model: str = os.getenv("TEXT_LLM_MODEL", textmodel)
    vision_llm_model: str = os.getenv("VISION_LLM_MODEL", visionmodel)
    system_prompt: str = system_prompt
    embed_model: str = os.getenv("EMBED_MODEL", embedmodel)
    botname: str = botname
    ollama_endpoint: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    vision_endpoint: str = os.getenv("VISION_MODEL_ENDPOINT", "http://localhost:11434/api/generate")

    max_context_entries: int = 50
    recent_convo_limit: int = 100

config = Config()

# Create a dedicated asyncio loop in a separate thread
async_loop = asyncio.new_event_loop()
threading.Thread(target=async_loop.run_forever, daemon=True).start()

def _missing(field, idx, item):
    """Log a warning when a field is missing, and return a placeholder."""
    print(f"Search item #{idx} missing '{field}'. Keys: {list(item.keys())}")
    return f"<no {field}>"

class VTuberAI:
    def __init__(self):
        self.sio = None
        self.mindserver_connected = False

        # Queues and audio
        self.raw_queue = queue.Queue()
        self.text_queue = queue.Queue()
        
        # Memory system
        self.memory = []  # List of conversation entries
        self.embeddings_data = []  # List of embedded summaries
        
        # Init
        self._init_audio()
        self._init_ollama()
        self._init_embeddings()
        self._load_memory()
        
        self.msg_buffer: List[str] = []
        self.history: List[dict] = []
        self.speaking_thread: Optional[threading.Thread] = None
        self.processing = False
        self.last_interaction = time.time()
        self.botname = config.botname
        self.username = username
        self.training = False

    def _init_ollama(self):
        """Initialize Ollama client"""
        self.ollama_endpoint = config.ollama_endpoint
        
    def _init_embeddings(self):
        """Test Ollama embedding model availability"""
        try:
            # Test if the embedding model is available
            test_response = self._get_ollama_embedding("test")
            if test_response is not None:
                print(userTColor + f"[Embeddings] Loaded {config.embed_model} model." + resetTColor)
            else:
                print(errorTColor + f"[Error] Failed embed model init: {config.embed_model}" + resetTColor)
        except Exception as e:
            print(errorTColor + f"[Error] Failed embed test: {e}" + resetTColor)

    def _get_ollama_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from Ollama API"""
        try:
            url = f"{self.ollama_endpoint}/api/embeddings"
            payload = {
                "model": config.embed_model,
                "prompt": text
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding")
            
            if embedding is None:
                print(errorTColor + f"[Error] No embedding return: {text[:50]}..." + resetTColor)
                return None
                
            return embedding
            
        except requests.exceptions.RequestException as e:
            print(errorTColor + f"[Error] Ollama embed API req fail: {e}" + resetTColor)
            return None
        except Exception as e:
            print(errorTColor + f"[Error] Failed to get embed: {e}" + resetTColor)
            return None

    def _load_memory(self):
        """Load memory from JSON file"""
        try:
            if MEMORY_FILE.exists():
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                print(userTColor + f"[Memory] Loaded {len(self.memory)} entries." + resetTColor)
            else:
                self.memory = []
                self._save_memory()
        except Exception as e:
            print(errorTColor + f"[Error] Failed to load mem: {e}" + resetTColor)
            self.memory = []
            
        # Load embeddings
        try:
            if EMBEDDINGS_FILE.exists():
                with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
                    self.embeddings_data = json.load(f)
                print(userTColor + f"[Embeddings] Loaded {len(self.embeddings_data)} embed." + resetTColor)
            else:
                self.embeddings_data = []
                self._save_embeddings()
        except Exception as e:
            print(errorTColor + f"[Error] Failed embed load: {e}" + resetTColor)
            self.embeddings_data = []

    def _save_memory(self):
        """Save memory to JSON file"""
        try:
            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
            print(toolTColor + f"[Memory] Saved {len(self.memory)} entries to file." + resetTColor)
        except Exception as e:
            print(errorTColor + f"[Error] Failed mem save: {e}" + resetTColor)

    def summarize_memory(self):
        return summarize_memory(self)

    def _save_embeddings(self):
        """Save embeddings to JSON file"""
        try:
            EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.embeddings_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(errorTColor + f"[Error] Failed embed save: {e}" + resetTColor)

    def _parse_human_datetime(self, timestamp_str: str) -> datetime:
        """Parse human-readable timestamp to datetime object"""
        try:
            # Try multiple formats
            formats = [
                "%A, %B %d, %Y at %I:%M %p UTC",
                "%A, %B %d, %Y at %H:%M UTC",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(timestamp_str, fmt).replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
            
            # If no format works, try parsing ISO format
            return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except Exception:
            # Fallback to current time
            return datetime.now(timezone.utc)

    def _format_timestamp(self, dt: Optional[datetime] = None) -> str:
        """Format datetime to human-readable string"""
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%A, %B %d, %Y at %I:%M %p UTC")

    def _add_embedding(self, text: str, metadata: Dict[str, Any] = {}):
        """Add text to embeddings with metadata using Ollama"""
        if not text.strip():
            return
            
        try:
            embedding = self._get_ollama_embedding(text)
            if embedding is None:
                print(errorTColor + f"[Error] Failed to get embedding for text: {text[:50]}..." + resetTColor)
                return
                
            self.embeddings_data.append({
                'text': text,
                'embedding': embedding,
                'metadata': metadata or {},
                'timestamp': self._format_timestamp()
            })
            self._save_embeddings()
            print(toolTColor + f"[Embeddings] Added embedding for: {text[:50]}..." + resetTColor)
        except Exception as e:
            print(errorTColor + f"[Error] Failed to create embedding: {e}" + resetTColor)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            # Convert to numpy arrays for easier calculation
            a = np.array(vec1)
            b = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception as e:
            print(errorTColor + f"[Error] Failed to calculate similarity: {e}" + resetTColor)
            return 0.0

    def _search_embeddings(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search embeddings for similar content using Ollama"""
        if not self.embeddings_data:
            return []
            
        try:
            query_embedding = self._get_ollama_embedding(query)
            if query_embedding is None:
                print(errorTColor + "[Error] Failed to get query embedding" + resetTColor)
                return []
            
            results = []
            for item in self.embeddings_data:
                embedding = item.get('embedding')
                if not embedding:
                    continue
                    
                similarity = self._cosine_similarity(query_embedding, embedding)
                results.append({
                    'text': item['text'],
                    'similarity': float(similarity),
                    'metadata': item.get('metadata', {}),
                    'timestamp': item.get('timestamp', '')
                })
            
            # Sort by similarity and return top k
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:k]
        except Exception as e:
            print(errorTColor + f"[Error] Failed to search embeddings: {e}" + resetTColor)
            return []

 

    def _call_ollama(self, prompt: str, model: str, system_prompt: Optional[str] = None, image_data: str = "") -> str:
        """Call Ollama API with proper vision support"""
        try:
            if image_data:
                # Use chat API for vision models with images
                url = f"{self.ollama_endpoint}/api/chat"
                
                messages = []
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                
                # Add user message with image
                user_message = {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data]  # Base64 encoded image
                }
                messages.append(user_message)
                
                payload = {
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "temperature": 0.7,
                }
            else:
                # Use generate API for text-only requests
                url = f"{self.ollama_endpoint}/api/generate"
                
                full_prompt = ""
                if system_prompt:
                    full_prompt += f"{system_prompt}\n\n"
                full_prompt += prompt
                
                payload = {
                    "model": model,
                    "prompt": full_prompt,
                    "stream": False,
                    "temperature": 0.7,
                }
            
            print(toolTColor + f"[Ollama] Calling {model} with {'vision' if image_data else 'text'} input" + resetTColor)
            
            response = requests.post(url, json=payload, timeout=120)  # Increased timeout for vision
            response.raise_for_status()
            result = response.json()
            
            print(systemTColor + "[DEBUG] Ollama Raw Response:" + resetTColor)
            print(json.dumps(result, indent=2))

            # Log performance metrics
            if "created_at" in result:
                print(systemTColor + f"Created at: {result['created_at']}" + resetTColor)
            if "model" in result:
                print(systemTColor + f"Model: {result['model']}" + resetTColor)
            if "done" in result:
                print(systemTColor + f"Done: {result['done']}" + resetTColor)
            if "total_duration" in result:
                ms = result["total_duration"] / 1e6
                print(systemTColor + f"Total duration: {ms:.2f} ms" + resetTColor)

            # Extract content from response
            content = ""
            if "response" in result:
                content = result["response"]
            elif "message" in result and "content" in result["message"]:
                content = result["message"]["content"]
            elif "choices" in result and result["choices"]:
                content = result["choices"][0].get("message", {}).get("content", "")

            # Parse thinking tags and actual response
            raw_content = content.strip()
            thinking_content = ""
            actual_response = raw_content
            
            if "<think>" in raw_content and "</think>" in raw_content:
                import re
                think_match = re.search(r'<think>(.*?)</think>', raw_content, re.DOTALL)
                if think_match:
                    thinking_content = think_match.group(1).strip()
                    actual_response = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
            
            if thinking_content:
                print(toolTColor + "[Model Thinking Process]:" + resetTColor)
                print(toolTColor + thinking_content + resetTColor)
                print()
            
            # Clean up response
            cleaned_response = "".join(c for c in actual_response if c not in "#*`>")
            
            print(botTColor + "[Ollama Final Response Text]:" + resetTColor, cleaned_response)
            return cleaned_response
            
        except Exception as e:
            print(errorTColor + f"[Error] Ollama API call failed: {e}" + resetTColor)
            return ""

    def _get_recent_context(self, limit: int = 0) -> List[Dict[str, Any]]:
        """Get recent conversation entries for context"""
        if limit == 0:
            limit = config.max_context_entries
        
        # Get recent raw entries
        recent_entries = self.memory[-limit:] if len(self.memory) > limit else self.memory
        return recent_entries

    def _get_memory_context(self, query: str) -> str:
        """Get relevant memory context from embeddings and recent conversations"""
        context_parts = []
        
        # Get embedded summaries
        similar_memories = self._search_embeddings(query, k=3)
        if similar_memories:
            context_parts.append("=== RELEVANT MEMORIES ===")
            for memory in similar_memories:
                context_parts.append(f"- {memory['text']} (similarity: {memory['similarity']:.2f})")
            context_parts.append("")
        
        # Get recent conversations
        recent = self._get_recent_context()
        if recent:
            context_parts.append("=== RECENT CONVERSATIONS ===")
            for entry in recent:
                role = entry.get('role', '')
                content = entry.get('content', '')
                timestamp = entry.get('timestamp', '')
                
                if role == 'user':
                    context_parts.append(f"[{timestamp}] {self.username}: {content}")
                elif role == 'assistant':
                    context_parts.append(f"[{timestamp}] {self.botname}: {content}")
            context_parts.append("")
        
        return "\n".join(context_parts)

    def _save_interaction(self, user_input: str, bot_response: str):
        """Save user input and bot response to memory"""
        timestamp = self._format_timestamp()
        
        # Add user message
        self.memory.append({
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        })
        
        # Add assistant response
        self.memory.append({
            "role": "assistant", 
            "content": bot_response,
            "timestamp": timestamp
        })
        
        self._save_memory()
        print(userTColor + f"[Memory] Saved interaction to memory.json" + resetTColor)

    def print_long_term_memory(self):
        """Print memory statistics and recent entries"""
        print(f"[Memory] Total entries: {len(self.memory)}")
        print(f"[Memory] Total embeddings: {len(self.embeddings_data)}")
        
        if self.memory:
            print("\n=== RECENT MEMORY ===")
            recent = self.memory[-10:] if len(self.memory) > 10 else self.memory
            for entry in recent:
                role = entry.get('role', '')
                content = entry.get('content', '')[:100] + "..." if len(entry.get('content', '')) > 100 else entry.get('content', '')
                timestamp = entry.get('timestamp', '')
                print(f"[{timestamp}] {role}: {content}")
        
        if self.embeddings_data:
            print("\n=== EMBEDDED SUMMARIES ===")
            for emb in self.embeddings_data[-5:]:  # Show last 5 summaries
                text = emb.get('text', '')[:100] + "..." if len(emb.get('text', '')) > 100 else emb.get('text', '')
                timestamp = emb.get('timestamp', '')
                print(f"[{timestamp}] {text}")

    def _init_audio(self):
        return init_audio(self)
    
    def _start_vosk_stream(self):
        start_vosk_stream(self)

    def _audio_callback(self, indata, frames, time_info, status):
        return audio_callback(self, indata, frames, time_info, status)
    
    def stop_stream(self):
        return sd.stop()

    def _capture_screenshot(self) -> str:
        print(systemTColor + "Taking screenshot" + resetTColor)
        buf = BytesIO()
        pyautogui.screenshot().save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    async def _generate_response_async(self, text: str, mode: str) -> str:
        """Generate response with proper vision handling"""
        if not text.strip():
            return ""
        
        # Check if this is a vision request
        needs_vision = any(keyword in text.lower() for keyword in VISION_KEYWORDS)
        
        # Get screenshot and vision summary if needed
        vision_summary = ""
        screenshot_data = ""
        
        if needs_vision or mode in ("vision", "game"):
            print(systemTColor + "[Vision] Taking screenshot for analysis..." + resetTColor)
            screenshot_data = self._capture_screenshot()
            
            # Get vision model summary first
            vision_prompt = f"""Describe what you see in this screenshot in detail. Focus on:
                - UI elements and text visible
                - Any applications or windows open  
                - Current state of the screen
                - Any relevant visual information

                User query: {text}"""
                
            print(toolTColor + "[Vision] Analyzing screenshot..." + resetTColor)
            vision_summary = self._call_ollama(
                vision_prompt, 
                config.vision_llm_model, 
                "You are a helpful vision assistant that describes screenshots accurately and concisely.",
                screenshot_data
            )
            
            if vision_summary:
                print(userTColor + f"[Vision] Screenshot analyzed: {vision_summary[:100]}..." + resetTColor)
            else:
                print(errorTColor + "[Vision] Failed to analyze screenshot" + resetTColor)
        
        # Get context
        memory_context = self._get_memory_context(text)
        
        # Get search results (disabled for now)
        needs_search = any(keyword in text.lower() for keyword in SEARCH_KEYWORDS)
        # if needs_search:
        #     print(systemTColor + "[Search] Performing web search for query..." + resetTColor)
        search_results = web_search_summary(text) if needs_search else "[]"
        # search_results = []

        # Build comprehensive prompt
        full_prompt = f"""
        [[SYSTEM]]
        {config.system_prompt}
        
        [[MEMORY_CONTEXT]]
        {memory_context}
        
        [[SEARCH_RESULTS]]
        {search_results}
        """
        
        # Add vision context if available
        if vision_summary:
            full_prompt += f"""
        [[VISION_ANALYSIS]]
        Current screenshot shows: {vision_summary}
        """
        
        full_prompt += f"""
        [[USER]]
        {text}
        
        Response:
        """

        print(toolTColor + f"[Prompting] Using {config.text_llm_model} for response generation" + resetTColor)
        
        # Generate final response using text model (with vision context if available)
        reply = self._call_ollama(full_prompt, config.text_llm_model)
        
        return reply.strip() if reply else ""

    async def _process_prompt_async(self, text: str, mode: str) -> str:
        """Full processing including storage - used for normal mode"""
        reply = await self._generate_response_async(text, mode)
        
        # Save to storage
        self._save_interaction(text, reply)
        return reply

    # MindServer methods remain the same
    def _init_mindserver(self):
       return init_mindserver(self)

    def _on_mind_msg(self, agent, payload):
        return on_mind_msg(self, agent, payload)

    def _interaction_loop(self, get_input: Callable[[], Optional[str]], mode: str):
        print(systemTColor + f"[{mode.upper()} MODE] Listening..." + resetTColor)
        if self.training:
            print(systemTColor + "[TRAINING MODE] Responses will not be saved unless approved." + resetTColor)
        
        while True:
            if (self.speaking_thread and self.speaking_thread.is_alive()) or self.processing:
                time.sleep(0.1)
                continue
                
            texts = []
            if mode == "text":
                inp = get_input()
                if inp is None:
                    break
                texts = [inp]
            else:
                # Collect all available text from the queue
                while not self.text_queue.empty():
                    text = self.text_queue.get()
                    if text.strip():  # Only add non-empty text
                        texts.append(text)
                        
            # Join texts and check if we have actual content
            user_text = " ".join(texts).strip()
            
            # Skip processing if no text or only whitespace
            if not user_text:
                # Only check timeout if we haven't had any interaction recently
                if time.time() - self.last_interaction > PROMPT_TIMEOUT:
                    time.sleep(0.1)
                continue
                
            # Update interaction time only when we have actual text
            self.last_interaction = time.time()
            
            if user_text.lower() == "exit":
                break
            if user_text.lower() == "/memory":
                self.print_long_term_memory()
                continue
            if user_text.lower() == "/summarize":
                self.summarize_memory()
                continue
                
            print(userTColor + f"{self.username}: {user_text}" + resetTColor)
            
            # Set processing flag to prevent multiple simultaneous requests
            self.processing = True
            
            try:
                # Generate response based on mode
                if self.training:
                    # In training mode, generate response without saving
                    future = asyncio.run_coroutine_threadsafe(
                        self._generate_response_async(user_text, mode), async_loop
                    )
                    reply = future.result()
                    print(botTColor + f"{self.botname}: {reply}" + resetTColor)
                    
                    # Ask for approval before saving
                    determination = input(systemTColor + "Is this a good response? (y/n): " + resetTColor)
                    if determination.lower() == "y":
                        self._save_interaction(user_text, reply)
                        print(userTColor + "[TRAINING] Response saved to memory." + resetTColor)
                    else:
                        print(errorTColor + "[TRAINING] Response discarded." + resetTColor)
                        continue
                else:
                    # In normal mode, generate and save automatically
                    future = asyncio.run_coroutine_threadsafe(
                        self._process_prompt_async(user_text, mode), async_loop
                    )
                    reply = future.result()
                    
                    # Only proceed if we got a valid reply
                    if reply and reply.strip():
                        print(botTColor + f"{self.botname}: {reply}" + resetTColor)
                        
                        # Voice synthesis for non-text modes
                        if mode != "text" and len(reply) < 600:
                            t = threading.Thread(
                                target=speak_through_vbcable, args=(reply,), daemon=True
                            )
                            t.start()
                            self.speaking_thread = t
                            
                        # Update history
                        self.history.extend([
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": reply},
                        ])
                        self.msg_buffer.append(f"{self.username}: {user_text}\n{self.botname}: {reply}")
                    else:
                        print(errorTColor + "[ERROR] Received empty response from model" + resetTColor)
            
            except Exception as e:
                print(errorTColor + f"[ERROR] Failed to process response: {e}" + resetTColor)
            finally:
                # Always clear the processing flag
                self.processing = False

    def run(self):
        print("[INFO] Starting VTuber AI. Type 'exit' to quit.")
        print("[INFO] Available commands: /memory, /summarize")
        mode = (
            input(
                userTColor
                + "Mode (text/talk/vision/game/minecraft/train): "
                + resetTColor
            )
            .strip()
            .lower()
        )
        valid = ("text", "talk", "vision", "game", "minecraft", "train")
        while mode not in valid:
            mode = (
                input("Invalid. Choose from text/talk/vision/game/minecraft/train: ")
                .strip()
                .lower()
            )
        if mode == "train":
            self.training = True
            mode = "text"
        input_getter = input if mode == "text" else lambda: None
        try:
            self._interaction_loop(input_getter, mode)
        finally:
            self.raw_queue.put(b"__EXIT__")
            self.stop_stream()
            if self.sio:
                self.sio.disconnect()


if __name__ == "__main__":
    VTuberAI().run()