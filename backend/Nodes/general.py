# from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import OPENROUTER_API_KEY
import os
from langchain_openai import ChatOpenAI


def load_rag(category, filename):
    """Load markdown file from backend/rag/<category>/<filename>"""
    base_path = "backend/rag"
    full_path = os.path.join(base_path, category, filename)

    if os.path.exists(full_path):
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

class GeneralQuery:
    def __call__(self, state):
        print("from general")


        llm =ChatOpenAI(
            model_name="tngtech/deepseek-r1t2-chimera:free",  # pick any free model
            temperature=0,  # deterministic for classification
            openai_api_key=OPENROUTER_API_KEY,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        user_input=state.get("user_input","")
        context=state.get("context",[])
        user_lower = user_input.lower()
        rag_source = ""
        rag_text = ""


        if "opd" in user_lower or "timing" in user_lower or "clinic" in user_lower:
            rag_source = "opd_timings"
            rag_text = load_rag("faq", "opd_timings.md")

        elif "insurance" in user_lower:
            rag_source = "insurance"
            rag_text = load_rag("faq", "insurance.md")

        elif any(w in user_lower for w in ["lab", "test", "xray", "ultrasound", "ecg"]):
            rag_source = "lab_services"
            rag_text = load_rag("faq", "lab_services.md")

        else:
            rag_source = "general_faq"
            rag_text = load_rag("faq", "hospital_faq.md")


        prompt=f""" 
you are a helpful customer support agent working at abc hospital.
you job is to politely converse with the patient and answer all
his questions that are sensible and within the hospitals domain.
if from the input you feel that the patient wants to get a doctor
recommended to him then change the intent to 'recommend_doctor' OR
if you feel the intent as booking an appointment then change the 
intent to 'book_appointment'. if you decide to recommend doctor then 
tell the docotor which doctor it needs to see. and ask if the patient 
wants any further information about the doctors of that field.
only give one output.Always give Roman Urdu or english alphabet output.
Use ONLY the information given in RAG_CONTENT to answer general queries 
(OPD timings, insurance, lab tests, facilities, services, etc.)
RAG_CONTENT:
\"\"\"
{rag_text}
\"\"\"
Give the best possible reply based ONLY on RAG_CONTENT and the intent rules above.

user:"{user_input}"
past context:"{context}"
"""
        # -----------------------------
        # LLM Invocation
        # -----------------------------
        response = llm.invoke(prompt)

        # Record conversation history
        context.append({"bot": response.content})
        state["context"] = context
        state["response"] = response.content

        return state

