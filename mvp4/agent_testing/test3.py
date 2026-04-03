# --- 1. Import the libraries we need ---
import google.generativeai as genai
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
import queue
import threading
import uuid # To create unique filenames
import time
from playsound3 import playsound
import requests
from elevenlabsTest import tts_sentence_audio
# --- 2. Setup ---

# --- API KEY ---
# IMPORTANT: For security, it's better to set this as an environment variable
# or load from a secure file, but we'll use input() for this example.
print("--- Real-Time Gemini Transcription ---")
print("Note: This script will continuously record and transcribe until you stop it.")
try:
    API_KEY = os.environ['GOOGLE_AI_API_KEY']
    print("Found API key in environment variables.")
except KeyError:
    API_KEY = input("Paste your Google AI Studio API key here: ")

genai.configure(api_key=API_KEY)

# --- Audio Settings ---
SAMPLE_RATE = 16000  # 16kHz
CHANNELS = 1
DTYPE = 'int16'      # 16-bit audio
FILENAME_PREFIX = "temp_audio_chunk_"

# --- VAD (Voice Activity Detection) Settings ---
# This is our "silence detection"
# We'll check the audio volume (RMS) in small windows.
VAD_WINDOW_MS = 30           # Size of audio window to check for silence (in ms)
VAD_WINDOW_SAMPLES = int(SAMPLE_RATE * (VAD_WINDOW_MS / 1000.0))
VAD_THRESHOLD = 0.01         # Audio level (RMS) to consider "not silent". Tweak this!
SILENCE_DURATION_MS = 1500   # How long to wait in silence before sending (in ms)
MAX_RECORDING_SEC = 15       # Max length of a single chunk (prevents infinite recording)

# --- Global Threading Objects ---
partial_sentence = ""     # To hold any partial sentence between chunks
final_history = []          # To hold the full conversation history
audio_queue = queue.Queue() # A queue to pass audio from the recording thread to the main thread
is_recording = True         # A flag to stop the recording thread
model = genai.GenerativeModel(model_name="models/gemini-2.5-flash") # Load model once

# --- 3. The "Processing" Function (Runs in a separate thread) ---
def is_sentence_complete(text):
    """Simple heuristic to check if a sentence is complete."""
    if not text:
        return False
    return text.strip()[-1] in {'.', '!', '?'}
def process_audio_chunk(audio_data, file_path):
    """
    Saves, uploads, transcribes, and cleans up a single audio chunk.
    This function is designed to be run in its own thread to avoid
    blocking the main recording loop.
    """
    try:
        # 1. Save the audio data to a .wav file
        #    We use np.int16(audio_data * 32767) to convert our
        #    floating-point data back to the 16-bit integer format .wav expects.
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        write(file_path, SAMPLE_RATE, audio_data_int16)

        # 2. Upload the file to Gemini
        # print(f"Uploading {file_path}...")
        audio_file = genai.upload_file(path=file_path, mime_type="audio/wav")
        
        # 3. Transcribe with Gemini
        # print("Transcribing...")
        response = model.generate_content([
            # Added a more specific prompt for better results
            "Please transcribe the following audio. The user is speaking for a real-time transcription. If there is no speech, just return an empty string.In roman urdu or english.  Add proper punctuation; return the sentence as it should appear.",
            audio_file
        ])

        # 4. Print the result
        if response.text:
            # We use end=" " to make the output flow like a conversation
            print(f"Transcript: {response.text}")
            if is_sentence_complete(response.text):
                print("sentence completed")
                full_sentence=partial_sentence+" "+response.text
                print(f"Final Sentence: {partial_sentence.strip()}")
                try:
                    resp2 = model.generate_content([
            # Added a more specific prompt for better results
         "Rewrite this sentence in clean Roman Urdu. Fix grammar and punctuation.Return ONLY the rewritten sentence. Nothing else. No explanations.",

            full_sentence])
                    Final_sentence = getattr(resp2, "text", "").strip()
                    if not Final_sentence:
                        Final_sentence = full_sentence

                    print(f"Cleaned & Translated: {Final_sentence}\n")
                except:
                    Final_sentence = full_sentence
                print(f"Cleaned & Translated: {Final_sentence}\n")
                final_history.append(Final_sentence)

                url = "http://127.0.0.1:8000/chat"
                payload = {"user_input": Final_sentence}
                                
                headers = {
                    "Content-Type": "application/json"
                }
                response = requests.post(url, json=payload, headers=headers)

                print(response.status_code)
                import ast 
                import re
                res=response.text
                cleaned=ast.literal_eval(res)
                reply=cleaned["reply"].strip()
                clean_reply = re.split(r"\*\*intent:\*\*", reply)[0].strip()

                print(clean_reply)
                
                print(response.json())
                tts_thread = threading.Thread(target=tts_sentence_audio, args=(clean_reply,))
                tts_thread.daemon = True # <--- THIS IS CRITICAL
                tts_thread.start()
                # threading.Thread(target=tts_sentence_audio, args=(clean_reply,)).start()


                

    except Exception as e:
        print(f"Error processing chunk: {e}")
        # Handle API errors, e.g., rate limits
        if "rate limit" in str(e).lower():
            print("Rate limit hit. Pausing for 5 seconds...")
            time.sleep(5)

    finally:
        # 5. Cleanup
        # Delete the file from Google's servers
        try:
            genai.delete_file(audio_file.name)
        except Exception as e:
            # print(f"Warning: Could not delete uploaded file {audio_file.name}. {e}")
            pass # Continue anyway
        
        # Delete the local file
        if os.path.exists(file_path):
            os.remove(file_path)
            # print(f"Cleaned up {file_path}.")


# --- 4. The "Recording" Function (Runs in the audio thread) ---

def audio_callback(indata, frames, time, status):
    """
    This function is called by sounddevice for each new audio chunk.
    It runs in a separate, high-priority thread.
    We just put the data into our queue and do no real processing here.
    """
    if status:
        print(f"Audio callback status: {status}", flush=True)
    
    # Add the new audio data (which is a numpy array) to the queue
    audio_queue.put(indata.copy())

# --- 5. The Main Loop (VAD and Threading) ---

def main():
    global is_recording
    
    # This list will hold all audio data for the current "utterance"
    current_utterance_buffer = []
    
    # Store silent chunks to check for the end of speech
    silent_chunks = 0
    silence_threshold = int((SILENCE_DURATION_MS / 1000.0) * (SAMPLE_RATE / VAD_WINDOW_SAMPLES))
    max_chunks = int((MAX_RECORDING_SEC / 1.0) * (SAMPLE_RATE / VAD_WINDOW_SAMPLES))

    print("\nStarting audio stream... (Speak now!)")
    print(f"Waiting for speech (VAD Threshold: {VAD_THRESHOLD})...")
    print("Press ENTER at any time to stop.")

    try:
        # Start the continuous audio recording stream
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32', # Use float32 for easier RMS calculation
            blocksize=VAD_WINDOW_SAMPLES,
            callback=audio_callback
        ):
            is_speaking = False

            while is_recording:
                try:
                    # Get a chunk of audio from the queue
                    # This will wait until a new chunk is available
                    audio_chunk = audio_queue.get(timeout=1) # Wait max 1 sec
                    
                    # Calculate volume (RMS)
                    rms = np.sqrt(np.mean(audio_chunk**2))

                    if is_speaking:
                        # --- We are currently recording an utterance ---
                        current_utterance_buffer.append(audio_chunk)

                        if rms < VAD_THRESHOLD:
                            # This chunk is silent, increment silence counter
                            silent_chunks += 1
                        else:
                            # This chunk has speech, reset silence counter
                            silent_chunks = 0

                        # Check if silence duration is met OR max length is hit
                        if silent_chunks > silence_threshold or len(current_utterance_buffer) > max_chunks:
                            is_speaking = False
                            
                            # Combine all audio chunks for this utterance
                            full_audio_data = np.concatenate(current_utterance_buffer)
                            
                            # Create a unique filename
                            temp_filename = FILENAME_PREFIX + str(uuid.uuid4()) + ".wav"
                            
                            print(f"\n[Speech detected. Sending {len(full_audio_data)/SAMPLE_RATE:.2f}s chunk...] ")
                            
                            # Start a NEW THREAD to process this audio chunk
                            # This lets us go back to listening immediately
                            threading.Thread(
                                target=process_audio_chunk,
                                args=(full_audio_data, temp_filename)
                            ).start()
                            
                            # Clear the buffer for the next utterance
                            current_utterance_buffer = []
                            silent_chunks = 0

                    elif rms > VAD_THRESHOLD:
                        # --- Speech has just started ---
                        print("[Speech started...]", end="", flush=True)
                        is_speaking = True
                        silent_chunks = 0
                        current_utterance_buffer = [audio_chunk] # Start a new buffer

                except queue.Empty:
                    # This happens if no audio is received for 1 sec
                    # We just continue waiting
                    if is_speaking:
                        # If we were speaking and the queue is empty (e.g., mic unplugged)
                        # just process what we have.
                        print("[Audio stream timeout. Processing last chunk.]")
                        is_speaking = False
                        full_audio_data = np.concatenate(current_utterance_buffer)
                        temp_filename = FILENAME_PREFIX + str(uuid.uuid4()) + ".wav"
                        threading.Thread(
                            target=process_audio_chunk,
                            args=(full_audio_data, temp_filename)
                        ).start()
                        current_utterance_buffer = []

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        print("Stopping... (Waiting for last chunks to finish)")
        is_recording = False
        # The 'with' block will automatically close the stream

# --- 6. Graceful Stop ---

def listen_for_stop():
    """Runs in a separate thread to listen for the ENTER key."""
    global is_recording
    input() # Wait for user to press ENTER
    is_recording = False

# --- 7. Run the application ---
if __name__ == "__main__":
    # Start the thread that listens for the ENTER key
    playsound("intro.mp3")
    stop_thread = threading.Thread(target=listen_for_stop)
    stop_thread.daemon = True # Allows program to exit even if this thread is running
    stop_thread.start()
    
    main()
    
    print("Program finished.")