import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize App & API Clients
app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# 3. Mount Static Files & Templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# 4. Data Models
class ChatRequest(BaseModel):
    message: str
    mode: str = "general"  # explain, debug, summarize, general

# 5. Routes
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/projects", response_class=HTMLResponse)
async def projects(request: Request):
    return templates.TemplateResponse("projects.html", {"request": request})

@app.get("/architecture", response_class=HTMLResponse)
async def architecture(request: Request):
    return templates.TemplateResponse("architecture.html", {"request": request})

@app.get("/playground", response_class=HTMLResponse)
async def playground(request: Request):
    return templates.TemplateResponse("playground.html", {"request": request})

@app.post("/chat")
async def chat_endpoint(chat_request: ChatRequest):
    try:
        # System Prompts based on Mode
        system_prompts = {
            "general": "You are Varsh.AI, an advanced portfolio assistant. You are professional, concise, and technical. You answer questions about Varsh's skills (Python, AI, FastAPI) and experience.",
            "explain": "You are a Tutor. Explain complex AI concepts simply (ELI5) using analogies. Keep it brief.",
            "debug": "You are a Senior Debugger. Analyze code snippets, find errors, and suggest fixes concisely.",
            "summarize": "You are a Summarizer. Compress the following text into 3 key bullet points."
        }
        
        selected_prompt = system_prompts.get(chat_request.mode, system_prompts["general"])

        # CALL GROQ API (Fixed Model Version)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # <--- UPDATED TO NEW MODEL
            messages=[
                {"role": "system", "content": selected_prompt},
                {"role": "user", "content": chat_request.message}
            ],
            temperature=0.7,
            max_tokens=300,
            top_p=1,
            stream=False,
            stop=None,
        )

        ai_response = completion.choices[0].message.content
        return {"response": ai_response}

    except Exception as e:
        # Error Handling
        print(f"Error: {e}")
        return JSONResponse(
            status_code=500, 
            content={"response": f"Error: System Malfunction. {str(e)}"}
        )

# Run logic is handled by Uvicorn (Procfile)