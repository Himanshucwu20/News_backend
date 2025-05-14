# app/routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.session import create_session, get_session_history, clear_session
from app.services.embeddings import get_response_from_query, get_response_with_gemini

chatbot_router = APIRouter(prefix="/api", tags=["chatbot"])

class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None

@chatbot_router.post("/session")
async def create_new_session():
    session_id=create_session()
    return {"session_id": session_id}

@chatbot_router.post("/chat")
async def chat(request: QueryRequest):
    session_id = request.session_id or create_session()

    # Step 1: Get relevant context chunks using FAISS
    context = get_response_from_query(request.query)

    # Step 2: Build a prompt that instructs Gemini to refer to context but generate a full answer
    full_prompt = f"""
You are a smart AI assistant. Below is some related information that may help answer the user's question.

Reference Context (use this only if relevant, or ignore if unhelpful):
\"\"\"
{context}
\"\"\"

User's Question:
{request.query}

Generate a complete, clear, and accurate response based on your own knowledge, but use the context above if it's relevant, if there is no context then give response according to your understanding (Do not mention about the context given to you, give response as you are answering the query).
"""

    # Step 3: Get Gemini's response
    response = get_response_with_gemini(full_prompt)

    return {"session_id": session_id, "response": response}

@chatbot_router.get("/history/{session_id}")
async def history(session_id: str):
    history = get_session_history(session_id)
    if not history:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": history}

@chatbot_router.post("/clear_session/{session_id}")
async def clear(session_id: str):
    result = clear_session(session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session cleared successfully"}
