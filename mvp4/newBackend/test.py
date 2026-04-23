import requests
import uuid
import time

API_URL = "http://localhost:8000/chat"
SESSION_ID = f"test-session-{uuid.uuid4().hex[:6]}"
OUTPUT_FILE = "conversation_log.txt"

# A simulated patient conversation script
PATIENT_SCRIPT = [
    "I have a very bad stomach ache and I feel nauseous.",
    "It started last night after dinner.",
    "No, I haven't taken any medicine yet.",
    "Yes, I have some mild fever too.",
    "Okay, my phone number is 03001234567.",
    "Dr. Ali sounds good.",
    "Tomorrow at 10:30 AM please.",
    "Yes, please confirm the booking.",
    "Nothing else, thank you. Goodbye!"
]

def log_to_file(text):
    print(text)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")

def main():
    log_to_file(f"=== STARTING TEST SESSION: {SESSION_ID} ===\n")
    
    for user_msg in PATIENT_SCRIPT:
        log_to_file(f"👤 Patient: {user_msg}")
        
        payload = {
            "session_id": SESSION_ID,
            "user_input": user_msg,
            "channel": "chat"
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            bot_reply = data.get("reply", "[No reply found]")
            log_to_file(f"🤖 Bot: {bot_reply}\n")
            
        except requests.exceptions.RequestException as e:
            log_to_file(f"❌ ERROR: Failed to connect to backend: {e}\n")
            break
            
        # Small delay to mimic human interaction and allow backend to breathe
        time.sleep(2)

    log_to_file("=== TEST COMPLETED ===")

if __name__ == "__main__":
    # Clear the file before starting
    open(OUTPUT_FILE, "w").close() 
    main()