# tts_vbcable.py

import os
import tempfile
import pyttsx3
import sounddevice as sd
import soundfile as sf

def speak_through_vbcable(text):
    # Set up pyttsx3 with desired voice settings
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    # You may need to adjust the index depending on your system's installed voices
    engine.setProperty('voice', voices[1].id)  # Example: Microsoft David or Zira
    engine.setProperty('rate', 250)
    engine.setProperty('volume', 1.0)

    # Generate speech and save to a temporary WAV file
    temp_wav = os.path.join(tempfile.gettempdir(), "tts_output.wav")
    engine.save_to_file(text, temp_wav)
    engine.runAndWait()

    # Locate VB-Cable playback device
    devices = sd.query_devices()
    cable_index = None
    for i, device in enumerate(devices):
        if "CABLE Input" in device['name']:  # Match VB-Cable playback device
            cable_index = i
            break

    if cable_index is None:
        raise RuntimeError("VB-Cable device not found!")

    # Load and play the audio through VB-Cable
    data, samplerate = sf.read(temp_wav, dtype='float32')
    sd.play(data, samplerate, device=cable_index)
    sd.wait()

    # Clean up
    os.remove(temp_wav)

    return "TTS executed successfully through VB-Cable."


# Example usage (only if run directly, not when imported)
if __name__ == "__main__":
    test_text = "This is a test of the VB-Cable text to speech system."
    print(speak_through_vbcable(test_text))
