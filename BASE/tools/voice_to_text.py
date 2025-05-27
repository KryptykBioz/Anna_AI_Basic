import os
import pyaudio
from vosk import Model, KaldiRecognizer
import json
import threading
from colorama import Fore, init
import sounddevice as sd

from personality.bot_info import botname, toolTColor, resetTColor, errorTColor
# Optional: ANSI color for console output
YELLOW = "\033[93m"

samplerate: int = 16000

def load_vosk_model(model_dir="models/vosk-model-small-en-us-0.15"):
    """
    Loads the Vosk speech recognition model from the specified directory.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError("Vosk model not found! Download and place it in the 'models' directory.")
    return Model(model_dir)

# def get_device_index_by_name(target_name):
#     p = pyaudio.PyAudio()
#     for i in range(p.get_device_count()):
#         info = p.get_device_info_by_index(i)
#         if target_name.lower() in info['name'].lower():
#             return i
#     raise ValueError(f"Device with name '{target_name}' not found.")

def listen_and_transcribe(min_length=5, max_attempts=50):
    """
    Listens through the microphone and returns transcribed text using the provided Vosk model.
    """
    model = load_vosk_model()
    recognizer = KaldiRecognizer(model, 16000)
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        input_device_index=10,  # Replace with actual headset mic index
        frames_per_buffer=2000
    )

    print(YELLOW + "Listening...")
    attempts = 0

    try:
        while True:
            data = stream.read(2000, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = recognizer.Result()
                text = result[14:-3].strip()  # Extract actual text from JSON-like string

                if len(text) > min_length:
                    break
                else:
                    attempts += 1
                    if attempts >= max_attempts:
                        text = ""
                        break
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    return text

# Audio methods for VTuberAI class
def init_audio(self):
    """Initialize audio system for the VTuberAI instance"""
    self.vosk_model = load_vosk_model()
    self._start_vosk_stream()

def start_vosk_stream(self):
    """Start the Vosk recognition worker thread and audio stream"""
    def recognition_worker():
        rec = KaldiRecognizer(self.vosk_model, samplerate)
        while True:
            data = self.raw_queue.get()
            if data == b"__EXIT__":
                break
            if rec.AcceptWaveform(data):
                text = json.loads(rec.Result()).get("text", "").strip()
                if len(text) >= 5 and f"{botname}" not in text.lower():
                    self.text_queue.put(text)
                    print(toolTColor + f"[Speech] Recognized: {text}" + resetTColor)

    # Start the recognition worker thread
    threading.Thread(target=recognition_worker, daemon=True).start()
    
    # Start the audio input stream
    self.stream = sd.RawInputStream(
        samplerate=samplerate,
        blocksize=4096,
        dtype="int16",
        channels=1,
        callback=self._audio_callback,
    )
    self.stream.start()
    print(toolTColor + "[Listener] Audio stream started." + resetTColor)

def audio_callback(self, indata, frames, time_info, status):
    """Audio callback function to queue raw audio data"""
    if status:
        print(toolTColor + f"[Audio status]: {status}" + resetTColor)
    self.raw_queue.put(bytes(indata))