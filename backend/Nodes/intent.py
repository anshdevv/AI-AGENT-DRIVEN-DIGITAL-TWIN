# from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import OPENROUTER_API_KEY
from .general import GeneralQuery
from .rec_doc import RecommendDoctor
from .bk_apt import BookAppointment
import os
import json
import re
from langchain_openai import ChatOpenAI




# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
llm =ChatOpenAI(
    model_name="tngtech/deepseek-r1t2-chimera:free",  # pick any free model
    temperature=0,  # deterministic for classification
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1"
)
def load_symptom_map():
    path = "backend/rag/mapping/symptoms_to_specialization.md"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

class IntentClassifier:
    def __call__(self, state):
        print("========== Reached Intent Classifier function ======")
        print("State received from chat:", state)
        
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

Your task is to analyze the user's message and return a JSON containing the user's intent and any relevant details.

### INTENT CATEGORIES:
- "book_appointment": User explicitly wants to book or schedule an appointment.
- "recommend_doctor": User mentions a symptom, asks for a doctor by name, or requests a specialization.
- "general_query": General questions like hospital timings, insurance, location, etc.
- If none of the above applies, set intent to "general_query".
- All fields must always be filled; if unknown, use empty strings "" except for intent.


### EXTRACTION RULES:

1. **Symptoms & Specializations:**
   Use this mapping:
   \"\"\"{symptom_content}\"\"\"
   - If the user mentions a symptom from this list, set intent to "recommend_doctor".
   - Set "specialization" to the corresponding doctor type.
   - Extract the exact phrase mentioned by the user into "symptom".

2. **Direct Requests:**
   - If the user explicitly asks for a specialist (e.g., "I need a Cardiologist", "skin doctor"), set intent to "recommend_doctor" and set "specialization" accordingly.

3. **Doctor Name:**
   - If a doctor's name is mentioned, extract it into "doctor_name".

4. **Date/Time:**
   - Extract any mentioned date or time in the formats YYYY-MM-DD and HH:MM.

### OUTPUT FORMAT:
Return **only valid JSON** with the following fields:
{{
  "intent": "",
  "specialization": "",
  "doctor_name": "",
  "symptom": "",
  "date": "",
  "time": ""
}}

### USER INPUT:
"{user_input}"

### PAST CONTEXT:
"{context}"
 """

        
        # prompt="return the response of what the user is saying"
        
        response = llm.invoke(prompt)
        content = response.content.strip()
        print(response)
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
        print(state)
        #print("Current Classified intent",state["intent"])

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
