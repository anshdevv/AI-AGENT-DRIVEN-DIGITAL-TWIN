# --- 1. Import the libraries we need ---

import google.generativeai as genai  # The official Google Gemini library
import sounddevice as sd              # A library to record audio from your mic
from scipy.io.wavfile import write    # A library to save your recording as a .wav file
import os                             # Helps clean up the file afterward

# --- 2. Setup ---

# Ask for your secret password
API_KEY = input("Paste your Google AI Studio API key here: ")
print(API_KEY)
genai.configure(api_key=API_KEY)

# Settings for your recording
SAMPLE_RATE = 16000  # 16,000 "snapshots" of audio per second. This is standard.
DURATION = 5         # We will record for 5 seconds.
FILENAME = "temp_audio.wav" # The name of the file we'll create.

print("\n--- Your First Transcription ---")
input("Press ENTER to start recording for 5 seconds...")

# --- 3. Record the Audio ---

print("Recording... Speak now!")

# This command tells 'sounddevice' (sd) to record.
my_recording = sd.rec(int(DURATION * SAMPLE_RATE), 
                      samplerate=SAMPLE_RATE, 
                      channels=1,
                      dtype='int16')

# This command tells the script to PAUSE and wait for the recording to finish.
sd.wait() 

print("Recording finished!")

# --- 4. Save the Audio to a File ---

# This command takes your recording (my_recording) and saves it
# as a .wav file with the name we chose (FILENAME).
write(FILENAME, SAMPLE_RATE, my_recording)
print(f"Audio saved as {FILENAME}")

# --- 5. Send the File to Gemini ---

print("Uploading file to Gemini...")
# This uploads your .wav file and prepares it for the AI
audio_file = genai.upload_file(path=FILENAME, mime_type="audio/wav")

# --- 6. Transcribe with Gemini ---

print("File uploaded. Asking Gemini to transcribe...")
# Choose the AI model. 1.5-flash is fast and perfect for this.
model = genai.GenerativeModel(model_name="models/gemini-2.5-flash")

# Give Gemini its instructions:
# 1. The prompt: "Please transcribe this."
# 2. The file: audio_file
response = model.generate_content([
    "Please transcribe this audio.In english or roman english and give some responds and also translate it ", 
    audio_file
])

# --- 7. Show the Result! ---

print("\n--- TRANSCRIPTION ---")
print(response.text)
print("---------------------\n")

# --- 8. Cleanup ---
# Delete the temporary file from your computer
os.remove(FILENAME)
print(f"Cleaned up {FILENAME}.")
print("Done!")  