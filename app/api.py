
from fastapi import FastAPI ,HTTPException
from pydantic import BaseModel, Field

from fastapi import Request
from fastapi.responses import JSONResponse
import logging
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)
from app.pipeline import run_question

app = FastAPI(
    title="Research & Decision Intelligence Agent",
    version="0.1.0",
)
logger = logging.getLogger(__name__)



@app.exception_handler(OpenAIError)
async def handle_openai_error(
    request: Request,
    exc: OpenAIError,
):
    logger.exception(
        "Upstream OpenAI error on %s",
        request.url.path,
        exc_info=exc,
    )

    if isinstance(exc, APITimeoutError):
        status_code = 504
        message = "The AI service timed out. Please try again."

    elif isinstance(exc, RateLimitError):
        status_code = 503
        message = "The AI service is temporarily busy. Please try again."

    elif isinstance(exc, APIConnectionError):
        status_code = 503
        message = "The AI service is temporarily unavailable."

    else:
        status_code = 502
        message = "The AI service could not complete the request."

    return JSONResponse(
        status_code=status_code,
        content={"detail": message},
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(
    request: Request,
    exc: Exception,
):
    logger.error(
        "Unexpected error on %s",
        request.url.path,
        exc_info=(type(exc), exc, exc.__traceback__),
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected error occurred while processing your request."
        },
    )

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
    result = run_question(request.question)

    return AskResponse(
        answer=result["answer"],
        answer_status=result["answer_status"],
        decision=result["decision"],
    )
