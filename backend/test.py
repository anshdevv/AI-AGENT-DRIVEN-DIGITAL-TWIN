import os
from elevenlabs.client import ElevenLabs
from elevenlabs.core.api_error import ApiError


api_key = os.getenv("ELEVENLABS_API_KEY")
print("KEY FOUND:", bool(api_key))

client = ElevenLabs(api_key="dc6560a23f59a7d054fc22cbc3130cc4afc042b2df1b34589cc6ccc231798cd2")
VOICE_ID = "AMYtAh0F0P3iH1uImfZh"   # e.g. your voice id
MODEL_ID = "eleven_v3"           # or "eleven_multilingual_v2"
TEXT = "Hello, this is a test."

def main():
    client = ElevenLabs(api_key="dc6560a23f59a7d054fc22cbc3130cc4afc042b2df1b34589cc6ccc231798cd2")

    try:
        audio = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            model_id=MODEL_ID,
            text=TEXT,
            output_format="mp3_44100_128",
        )

        out_file = "voice_test.mp3"
        with open(out_file, "wb") as f:
            for chunk in audio:
                if chunk:
                    f.write(chunk)

        print(f"SUCCESS: voice works. Audio saved to {out_file}")

    except ApiError as e:
        print(f"FAILED: status_code={e.status_code}")
        print(f"Headers: {e.headers}")
        print(f"Body: {e.body}")

if __name__ == "__main__":
    main()