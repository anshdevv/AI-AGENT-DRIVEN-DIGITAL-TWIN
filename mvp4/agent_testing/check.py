import simpleaudio as sa
import threading
import queue
import time
from elevenlabs import ElevenLabs

# --- Configuration ---
SAMPLE_RATE = 16000
NUM_CHANNELS = 1
BYTES_PER_SAMPLE = 2
# Require a minimum of 0.1 seconds of audio (16000 * 2 * 0.1 = 3200 bytes)
MIN_BUFFER_SIZE = int(SAMPLE_RATE * BYTES_PER_SAMPLE * 0.1) 
# ---------------------

# Initialize ElevenLabs Client (Use your actual key)
client = ElevenLabs(api_key="a3a8de04b8cedd965d80f90f1f7b6058888943e27f7a1930fa4ec163abe78779")

# Create a queue for thread-safe audio chunk passing
audio_queue = queue.Queue()
# A sentinel value to signal the end of the stream
STREAM_END_SENTINEL = object()
# Synchronization Event: Signals when playback has officially started
playback_started_event = threading.Event() 

def audio_producer(text):
    """Downloads audio chunks from ElevenLabs and puts them into the queue."""
    print("Producer: Starting download from ElevenLabs...")
    try:
        # Stream bytes from ElevenLabs
        stream = client.text_to_speech.convert(
            voice_id="oH8YmZXJYEZq5ScgoGn9",
            model_id="eleven_multilingual_v2",
            text=text,
            output_format="pcm_16000"
        )
        
        # Read streamed chunks and put them into the queue
        for chunk in stream:
            if chunk:
                audio_queue.put(chunk)
                # Optional: slight delay can help system stability, but usually not needed
                # time.sleep(0.0001) 
        
        print("Producer: Finished downloading all chunks.")

    except Exception as e:
        print(f"Producer Error: {e}")
    finally:
        # Signal that the stream has ended
        audio_queue.put(STREAM_END_SENTINEL)

def audio_consumer_and_playback():
    """Pulls audio chunks from the queue and streams them to simpleaudio."""
    print("Consumer: Starting playback thread...")
    
    wave_params = (NUM_CHANNELS, BYTES_PER_SAMPLE, SAMPLE_RATE)
    total_audio_data = b''
    play_obj = None

    while True:
        # Wait a short time before getting the chunk to allow buffer to build slightly
        # We use a short timeout instead of blocking forever if we are waiting for start
        try:
            # We use get_nowait() or a timeout and then sleep/continue if the queue is empty 
            # to be more responsive in a real-time scenario, but here, we stick to blocking 
            # `get()` to ensure we process data as soon as it's available.
            chunk = audio_queue.get() 
        except queue.Empty:
            time.sleep(0.01)
            continue
        
        # Check for the end sentinel
        if chunk is STREAM_END_SENTINEL:
            print("Consumer: Stream end received. Waiting for final buffer to finish...")
            
            # Play any remaining data
            if total_audio_data:
                play_obj = sa.play_buffer(total_audio_data, *wave_params)
                print(f"Consumer: Playing final segment ({len(total_audio_data) / (SAMPLE_RATE*BYTES_PER_SAMPLE)}s)")
            
            # CRITICAL: Wait for the VERY LAST playback object to finish
            if play_obj is not None:
                play_obj.wait_done()
            
            break # Exit the loop
        
        # Append the new chunk to our current buffer
        total_audio_data += chunk
        
        # Playback Logic: Start playback only when buffer size is adequate
        # OR if playback has already started (meaning we are playing sequential segments)
        if play_obj is None or not play_obj.is_playing():
            # Check if we have enough data to ensure audible start
            if len(total_audio_data) >= MIN_BUFFER_SIZE:
                # Play the current collected buffer
                play_obj = sa.play_buffer(total_audio_data, *wave_params)
                print("Consumer: **Playback STARTED** (First chunk arrived)")
                playback_started_event.set() # Set the event immediately upon start!
                # Clear the buffer, as this portion is now playing
                total_audio_data = b''
        
        # For subsequent chunks, if the previous buffer finished, play the new accumulated data
        elif not play_obj.is_playing():
            if total_audio_data:
                play_obj = sa.play_buffer(total_audio_data, *wave_params)
                print(f"Consumer: Playing next buffer segment ({len(total_audio_data) / (SAMPLE_RATE*BYTES_PER_SAMPLE)}s)")
                total_audio_data = b''
        
        # Signal to the queue that the item has been processed
        audio_queue.task_done()
        
    print("Consumer: Playback finished and thread exiting.")


def tts_sentence_stream(text):
    """Main function to start producer and consumer threads."""
    
    # 1. Start the Producer thread
    producer_thread = threading.Thread(target=audio_producer, args=(text,))
    producer_thread.start()
    
    # 2. Start the Consumer/Playback thread
    consumer_thread = threading.Thread(target=audio_consumer_and_playback)
    consumer_thread.start()
    
    # 3. CRITICAL WAIT: Wait for playback to actually start (optional, but helps synchronization)
    print("Main: Waiting for audio playback to initialize...")
    # Wait up to 5 seconds for the Consumer to start playback.
    if not playback_started_event.wait(timeout=5):
        print("Main WARNING: Playback did not start within 5 seconds.")
        
    # 4. Wait for BOTH threads to complete their work
    # The consumer_thread.join() now guarantees the program waits for the final audio to play.
    producer_thread.join()
    consumer_thread.join()
    
    print("--- TTS Streaming Complete ---")

# -----------------------------
# 👉 Usage:
tts_sentence_stream("Hello! This sentence will start playing immediately after the first network chunk arrives, long before the entire message has finished downloading. This version is highly synchronized to prevent early program exit.")