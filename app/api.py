
from fastapi import FastAPI ,HTTPException
from pydantic import BaseModel, Field
import logging

from app.pipeline import run_question

app = FastAPI(
    title="Research & Decision Intelligence Agent",
    version="0.1.0",
)
logger = logging.getLogger(__name__)

class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class AskResponse(BaseModel):
    answer: str
    answer_status: str
    decision: str


@app.get("/")
def home():
    return {"message": "Research Agent API is running"}



@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    try:
        result = run_question(request.question)

        return AskResponse(
            answer=result["answer"],
            answer_status=result["answer_status"],
            decision=result["decision"],
        )

    except Exception:
        logger.exception("Unexpected error while processing /ask")

        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your question.",
        )
