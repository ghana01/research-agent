
from fastapi import FastAPI ,HTTPException
from pydantic import BaseModel, Field
from langsmith import tracing_context
from fastapi import Request
from typing import Literal
from fastapi.responses import JSONResponse
import logging
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAIError,
    RateLimitError,
)
from uuid import uuid4
from starlette.middleware.base import BaseHTTPMiddleware
from app.pipeline import run_question



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s %(levelname)s %(name)s "
        "request_id=%(request_id)s %(message)s"
    ),
)

logging.getLogger().addFilter(RequestIdFilter())

app = FastAPI(
    title="Research & Decision Intelligence Agent",
    version="0.1.0",
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id

    logger.info(
        "Request started",
        extra={"request_id": request_id, "method": request.method,
               "path": request.url.path},
    )

    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        logger.info(
            "Request completed",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
            },
        )
        return response

    except Exception:
        logger.exception(
            "Request failed",
            extra={"request_id": request_id},
        )
        raise 




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
    answer_status: Literal["ANSWERED", "ABSTAINED"]
    decision: Literal["ACCEPT", "REJECT"]


@app.get("/")
def home():
    return {"message": "Research Agent API is running"}





@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest, http_request: Request):
    request_id = http_request.state.request_id

    with tracing_context(
        metadata={"request_id": request_id}
    ):
        result = run_question(request.question)

    return AskResponse(
        answer=result["answer"],
        answer_status=result["answer_status"],
        decision=result["decision"],
    )
