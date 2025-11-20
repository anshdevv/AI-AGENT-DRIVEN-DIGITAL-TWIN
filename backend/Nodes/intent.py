from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import GOOGLE_API_KEY
from .general import GeneralQuery
from .rec_doc import RecommendDoctor
from .bk_apt import BookAppointment

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)

class IntentClassifier:
    def __call__(self, state):
        user_input = state.get("user_input", "")
        context = state.get("context", [])

        # Ask Gemini explicitly for the intent
        prompt = f"""
You are an intent classifier for a hospital chatbot.

Your job:

Classify the user's intent into one of these categories:

"book_appointment"

"recommend_doctor"

"general_query"

If the user is describing symptoms (e.g., pain, fever, rash, etc.),
set intent = "general_query" and include a probable doctor specialization
(e.g., "dentist", "cardiologist", "dermatologist", "ENT", "general physician").

If the user wants to book an appointment, extract:

The doctor's name (if mentioned)

The preferred date and/or time (if mentioned)

If the user asks for doctors regarding a particular specialization or mentions the name of a person with the title "doctor",
set intent = "recommend_doctor".

If the user asks about a doctor’s availability or timing (e.g., “what time is he available?”, “is Dr. Omer free tomorrow?”),
keep the intent as "recommend_doctor".
Only switch to "book_appointment" if the user explicitly requests or confirms booking
(e.g., “yes book it”, “schedule it”, “confirm appointment”).

If in the context it is mentioned that the user previously wanted an appointment for some day or time,
use that information from the context and include it in the JSON fields for "date" and "time".

Date and Time Formatting:

"date": Use "YYYY/MM/DD" if explicit, else natural phrases like "tomorrow", "today", or "next Monday".

"time": Use 24-hour format (morning=09:00, afternoon=14:00, evening=18:00).

Return only valid JSON in this exact format:


  "intent": "",
  "specialization": "",
  "doctor_name": "",
  "date": "",
  "time": ""



Use empty strings "" for any missing fields.
Do NOT include any explanations or text outside the JSON.

User: "{user_input}"
past context: "{context}"
        """
        response = llm.invoke(prompt)

        response = response.content.strip().lower()
        import json
        import re

        # Remove code block markers and leading/trailing whitespace
        clean_response = re.search(r"\{.*\}", response, re.DOTALL)
        if clean_response:
            clean_response = clean_response.group(0)
        else:
            print("failed cleaning json")

        # print(clean_response)
        result = json.loads(clean_response)
        intent = result["intent"]
        specialization = result.get("specialization")
        date= result.get("date")
        time = result.get("time")
        doctor_name = result.get("doctor_name")

        # doctor_name=result."doctor_name")


        state["intent"] = intent
        state["specialization"] = specialization
        if date!="":
            state["date"] = date
        if time!="":
            state["time"] = time
        state["doctor_name"]=doctor_name
        
        return state
