from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import GOOGLE_API_KEY
from .general import GeneralQuery
from .rec_doc import RecommendDoctor
from .bk_apt import BookAppointment
import os
import json
import re

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
def load_symptom_map():
    path = "backend/rag/faq/symptoms_to_specialization.md"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

class IntentClassifier:
    def __call__(self, state):
        if state.get("triage_active"):
            state["intent"] = "medical_triage"
            return state

        # 2. If we are in the middle of Booking (and not done), stay there.
        booking_step = state.get("booking_step")
        if booking_step and booking_step != "done":
            state["intent"] = "book_appointment"
            return state
        user_input = state.get("user_input", "")
        context = state.get("context", [])
        
        # 1. Load the symptom map content
        symptom_content = load_symptom_map()

        prompt = f"""
You are an intent classifier for a hospital chatbot.

Your job is to classify the user's intent into one of these categories:
- "book_appointment" (Only if user explicitly says "book", "schedule", or confirms a slot)
- "recommend_doctor" (If user asks for a doctor by name, requests a specific specialization, or mentions a symptom)
- "general_query" (timings, insurance, location, etc.)

### EXTRACTION RULES:
1. **Symptoms:**
   Use this mapping:
   \"\"\"{symptom_content}\"\"\"
   If the user mentions a symptom from this list (e.g. fever), set intent to "recommend_doctor" and set "specialization" to the corresponding doctor type.
    1. If user mentions a symptom from the list (e.g. "dizziness"), map it to the "specialization" (e.g. "Neurologist").
    2. ALSO extract the exact phrase into "symptom" (e.g. "dizziness").
2. **Direct Requests:**
   If the user explicitly asks for a specialist (e.g. "I need a Cardiologist", "skin doctor"), set intent to "recommend_doctor" and set "specialization" to that field.

3. **Doctor Name:**
   If a doctor's name is mentioned, extract it into "doctor_name".

4. **Date/Time:**
   Extract any date or time mentioned.

Return only valid JSON:
{{
  "intent": "",
  "specialization": "",
  "doctor_name": "",
  "symptom": "",         <-- The specific complaint
  "date": "",
  "time": ""
}}

User: "{user_input}"
past context: "{context}"
"""
        response = llm.invoke(prompt)

        content = response.content.strip()
        match = re.search(r"\{.*\}", content, re.DOTALL)
        clean_json = match.group(0) if match else "{}"
        try:
            result = json.loads(clean_json)
        except:
            result = {"intent": "general_query"}
        intent =result.get("intent", "general_query")
        specialization = result.get("specialization", "")
        date= result.get("date")
        time = result.get("time")
        doctor_name = result.get("doctor_name","")

        # doctor_name=result."doctor_name")


        state["intent"] = intent
        state["specialization"] = specialization
        state["patient_complaint"] = result.get("symptom", "")
        if date!="":
            state["date"] = date
        if time!="":
            state["time"] = time
        state["doctor_name"]=doctor_name
        
        return state
