from fastapi import FastAPI, HTTPException, UploadFile, File, Query
import io
from fastapi.middleware.cors import CORSMiddleware

# Import your Curo AI pipeline logic
from curo_logic import process_curo_query

app = FastAPI(title="CURO AI Backend", description="Clinical RAG API", version="1.0.0")

# Add CORS middleware if you intend to connect a frontend (React, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from pydantic import BaseModel
from typing import List, Optional

class DemographicData(BaseModel):
    age: Optional[str] = None
    sex: Optional[str] = None
    heartRate: Optional[str] = None
    bloodPressure: Optional[str] = None
    spo2: Optional[str] = None
    temp: Optional[str] = None
    respRate: Optional[str] = None

class SymptomRequest(BaseModel):
    query: str
    demography: Optional[DemographicData] = None
    user_id: Optional[str] = None  # Firebase UID for health records retrieval

class ChatRequest(BaseModel):
    message: str
    history: List[dict]
    context: str # The original clinical analysis response for grounding
    user_id: Optional[str] = None  # Firebase UID for health records retrieval

class CuroResponse(BaseModel):
    response: str
    extracted_ddx: List[str]
    winning_diagnosis: str
    abstracts: List[dict]
    graph_data: dict

class TriageChatRequest(BaseModel):
    history: List[dict] # {role: 'user'|'assistant', content: '...'}
    question_count: int

class TriageChatResponse(BaseModel):
    question: str
    options: List[str]
    finished: bool
    summary: Optional[str] = None

class RecordsQueryRequest(BaseModel):
    query: str
    user_id: str

class RecordsDeleteRequest(BaseModel):
    user_id: str
    filename: Optional[str] = None

class GenericRequest(BaseModel):
    query: str
    user_id: Optional[str] = None

# --- API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "CURO AI Engine is running online."}

@app.post("/api/analyze", response_model=CuroResponse)
def analyze_symptoms(request: SymptomRequest):
    """
    Accepts patient symptoms, runs the Clinical RAG pipeline, 
    and returns an empathetic, grounded response along with the DDx logic.
    Automatically retrieves relevant health records if user_id is provided.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
        
    try:
        # Run the LangChain logic
        demography_dict = request.demography.model_dump(exclude_none=True) if request.demography else None
        result = process_curo_query(request.query, demography_dict, request.user_id)
        return result
    except Exception as e:
        # Catch any unexpected errors (API timeouts, etc.)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generic-analyze")
def generic_analyze(request: GenericRequest):
    """
    Handles simple/common medical queries using a web search ReAct agent.
    Returns a formatted prescription without knowledge graphs or DDx.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    try:
        from curo_logic import process_generic_query
        result = process_generic_query(request.query, request.user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_followup(request: ChatRequest):
    """Handles follow-up questions from the user based on the initial analysis."""
    try:
        from curo_logic import process_chat_followup
        response = process_chat_followup(request.message, request.history, request.context, request.user_id)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/triage/chat")
def triage_chat(request: TriageChatRequest):
    """Agent-driven clinical triage step generation."""
    try:
        from curo_logic import process_triage_chat
        return process_triage_chat(request.history, request.question_count)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/asr")
async def transcribe_audio(file: UploadFile = File(...)):
    """Transcribes uploaded clinical audio blobs using server-side Whisper."""
    try:
        content = await file.read()
        from curo_logic import process_asr
        transcript = process_asr(content)
        return {"transcript": transcript}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ASR Error: {str(e)}")

# --- Health Records Vault Endpoints ---

@app.post("/api/records/upload")
async def upload_health_records(
    files: List[UploadFile] = File(...),
    user_id: str = Query(..., description="Firebase User ID")
):
    """
    Upload multiple PDF health records. Extracts text (with OCR for scanned PDFs),
    chunks, embeds, and stores in a persistent per-user ChromaDB collection.
    """
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    
    file_data = []
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"Only PDF files are supported. Got: {file.filename}")
        
        content = await file.read()
        file_data.append({
            "filename": file.filename,
            "content": content
        })
    
    if not file_data:
        raise HTTPException(status_code=400, detail="No files provided.")
    
    try:
        from curo_logic import process_health_records
        result = process_health_records(file_data, user_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/records/query")
def query_records(request: RecordsQueryRequest):
    """Query your stored health records using natural language."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    
    try:
        from curo_logic import query_health_records
        return query_health_records(request.query, request.user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/records/list")
def list_records(user_id: str = Query(..., description="Firebase User ID")):
    """List all stored health record documents for a user."""
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    
    try:
        from curo_logic import list_health_records
        return list_health_records(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/records/delete")
def delete_records(request: RecordsDeleteRequest):
    """Delete health records. If filename is provided, deletes only that file. Otherwise clears all."""
    if not request.user_id.strip():
        raise HTTPException(status_code=400, detail="user_id is required.")
    
    try:
        from curo_logic import delete_health_records
        return delete_health_records(request.user_id, request.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Keep the legacy single-file upload endpoint for backward compatibility
@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Extracts text from an uploaded clinical PDF (legacy endpoint)."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        content = await file.read()
        from curo_logic import extract_text_from_pdf
        text = extract_text_from_pdf(content, file.filename)
        return {"text": text, "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Runs the app on port 8000
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)