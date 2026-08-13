from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Import the core agent functions from the backend module
from backend import run_travel_agent, resume_travel_agent

# Apply nest_asyncio to allow running async code in environments that already have an event loop
import nest_asyncio

nest_asyncio.apply()

# Base directory of the application
BASE_DIR = Path(__file__).resolve().parent

# Initialize the FastAPI application with metadata
app = FastAPI(
    title="TripWeave AI",
    description=(
        "LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, "
        "Human-in-the-Loop, and FastAPI Frontend"
    ),
    version="2.0.0",
)

# Mount static files (CSS, JS, etc.) from the /static directory
app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

# Set up Jinja2 templates from the /templates directory
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Pydantic models for request validation


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


# Serve the main HTML page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


# Endpoint to initiate or continue a travel planning conversation
@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        # Reject empty messages
        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        # Call the backend agent with the user's input and optional thread_id
        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        # Log the error and return a 500 response
        print("ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


# Endpoint for the user to approve or reject a draft travel plan
@app.post("/api/travel/approve")
async def approve_travel_plan(request_data: ApprovalRequest):
    try:
        # If rejecting, feedback is mandatory
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        # Resume the agent with the user's decision
        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print("APPROVAL ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


# Simple health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TripWeave AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
        ],
    }


# Dummy favicon endpoint to avoid 404 errors
@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


# Run the application with uvicorn when executed directly
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )