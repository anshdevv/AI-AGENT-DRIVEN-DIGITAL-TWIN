from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from .graph import create_graph

app = FastAPI()

# Allow frontend (React) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # or specify your React URL e.g. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create the LangGraph workflow
graph = create_graph()
compiled_graph = graph.compile()   # compile once at startup

context=[]
@app.post("/chat")
def chat(user_input: str = Body(..., embed=True)):
    # Initialize the LangGraph state
    # state = {"user_input": user_input}
    context.append({"user":user_input})
    state={"user_input":user_input,"context":context}
    # Run the graph
    result = compiled_graph.invoke(state)
    context.append({"bot":result.get("response", "No response generated.")})
    # print(context

    # Send back the response to frontend
    return {"reply": result.get("response", "No response generated.")}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
