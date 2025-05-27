import os
import pyaudio
from vosk import Model, KaldiRecognizer

# Optional: ANSI color for console output
YELLOW = "\033[93m"

def load_vosk_model(model_dir="models/vosk-model-small-en-us-0.15"):
    """
    Loads the Vosk speech recognition model from the specified directory.
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError("Vosk model not found! Download and place it in the 'models' directory.")
    return Model(model_dir)

# USE THIS IF YOU NEED TO FIND YOUR AUDIO DEVICE FOR INPUT
# def get_device_index_by_name(target_name):
#     p = pyaudio.PyAudio()
#     for i in range(p.get_device_count()):
#         info = p.get_device_info_by_index(i)
#         if target_name.lower() in info['name'].lower():
#             return i
#     raise ValueError(f"Device with name '{target_name}' not found.")

def listen_and_transcribe(model, min_length=5, max_attempts=50):
    """
    Listens through the microphone and returns transcribed text using the provided Vosk model.
    """
    recognizer = KaldiRecognizer(model, 16000)
    audio = pyaudio.PyAudio()

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        # input_device_index=10,  # Replace with actual headset mic index to hardcode
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
                        # Default prompt sent to Bot to elicit response when the user hasn't spoken
                        text = "Make a comment about this conversation"
                        break
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

    return text

