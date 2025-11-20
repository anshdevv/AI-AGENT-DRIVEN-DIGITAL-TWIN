from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import GOOGLE_API_KEY

class GeneralQuery:
    def __call__(self, state):
        print("from general")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
        user_input=state.get("user_input","")
        context=state.get("context",[])
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
only give one output.
user:"{user_input}"
past context:"{context}"
"""
        response=llm.invoke(prompt)
        context.append({"bot":response.content})
        state['context']=context
        
        state["response"] = response.content
        return state


