import os
import shutil
import sys
from pathlib import Path
from typing import Any

import certifi
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient


# =========================================================
# Environment setup
# =========================================================


# Determine the base directory of the project and load environment variables from .env
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


# Set SSL certificate paths for secure HTTP requests
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# Retrieve API keys from environment variables
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATION_STACK_API_KEY =  os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"
UVX_COMMAND = shutil.which("uvx") or "uvx"


def _require_env(name: str, value: str | None) -> str:
    """
    Return an environment value or raise a clear error if missing.
    This is used to ensure required API keys are present.
    """
    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )
    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """
    Build a subprocess environment by copying the current environment
    and adding (or updating) the given key-value pairs.
    This is used to pass API keys to MCP server subprocesses.
    """
    env = os.environ.copy()
    for key, value in updates.items():
        if value:
            env[key] = value
    return env


# =========================================================
# LLM
# =========================================================

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=_require_env("GROQ_API_KEY", GROQ_API_KEY),
)


# =========================================================
# MCP client
# =========================================================

# Create a multi‑server MCP client that connects to three different MCP servers:
# - tavily: for web search (streamable HTTP)
# - aviationstack: for flight/aviation data (stdio, using a Python package)
# - weather: for current weather and forecast (stdio, using our custom script)

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={TAVILY_API_KEY or ''}"
            ),
        },
        "aviationstack": {
            "transport": "stdio",
            "command": "python",
            "args": [
                "aviationstack-mcp",
            ],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },
        "weather": {
            "transport": "stdio",
            "command": sys.executable,
            "args": [
                str(WEATHER_SERVER_PATH),
            ],
            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    """
    Load one specific tool from one MCP server.
    This is designed to prevent a failing server (e.g., weather or aviation)
    from crashing an unrelated request to Tavily.
    It also performs prerequisite checks (API keys, file existence, uvx presence).
    """
    
    # Validate required API keys and other prerequisites per server
    if server_name == "tavily":
        _require_env(
            "TAVILY_API_KEY",
            TAVILY_API_KEY,
        )
    elif server_name == "aviationstack":
        _require_env(
            "AVIATION_STACK_API_KEY",
            AVIATION_STACK_API_KEY,
        )
        # Ensure uvx is available (used to run aviationstack-mcp)
        if shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv, reopen the terminal, "
                "activate the travel environment, and run "
                "`uvx --version`."
            )
    elif server_name == "weather":
        _require_env(
            "OPENWEATHER_API_KEY",
            OPENWEATHER_API_KEY,
        )
        # Ensure the weather server script exists
        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(
                f"Weather MCP server not found: "
                f"{WEATHER_SERVER_PATH}"
            )

    # Load only the requested server's tools
    tools = await client.get_tools(
        server_name=server_name,
    )

    # Find the exact tool by name
    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(
                sorted(item.name for item in tools)
            )
            or "none"
        )
        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool


# =========================================================
# MCP connection test
# =========================================================

async def get_all_tools() -> None:
    """
    Test every MCP server independently.
    A failure in one server will not stop the testing of the others.
    This is useful for debugging connectivity and tool availability.
    """
    for server_name in (
        "tavily",
        "aviationstack",
        "weather",
    ):
        try:
            tools = await client.get_tools(
                server_name=server_name,
            )
            tool_names = (
                ", ".join(
                    tool.name
                    for tool in tools
                )
                or "no tools"
            )
            print(
                f"{server_name}: OK -> {tool_names}"
            )
        except Exception as exc:
            print(
                f"{server_name}: FAILED -> "
                f"{type(exc).__name__}: {exc}"
            )


# =========================================================
# Tavily MCP
# =========================================================

async def tavily_mcp_search(query: str):
    """Perform a web search using the Tavily MCP tool."""
    search_tool = await _get_server_tool(
        "tavily",
        "tavily_search",
    )
    return await search_tool.ainvoke(
        {
            "query": query,
        }
    )


# =========================================================
# AviationStack MCP
# =========================================================

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    """
    Call a specific AviationStack MCP tool (e.g., flight search).
    The tool name and arguments are provided dynamically.
    """
    aviation_tool = await _get_server_tool(
        "aviationstack",
        tool_name,
    )
    return await aviation_tool.ainvoke(
        tool_args or {}
    )


# =========================================================
# Weather MCP
# =========================================================

async def weather_mcp_search(city: str):
    """Get the current weather for a given city using the weather MCP tool."""
    weather_tool = await _get_server_tool(
        "weather",
        "get_current_weather",
    )
    return await weather_tool.ainvoke(
        {
            "city": city,
        }
    )


async def forecast_mcp_search(city: str):
    """Get a 5‑day forecast for a given city using the weather MCP tool."""
    forecast_tool = await _get_server_tool(
        "weather",
        "get_forecast",
    )
    return await forecast_tool.ainvoke(
        {
            "city": city,
        }
    )


# =========================================================
# Destination extractor
# =========================================================

def extract_destination(query: str) -> str:
    """
    Use the LLM to extract the destination city or country from a travel request.
    The LLM is prompted to return only the destination name without extra text.
    """
    prompt = f"""
    Extract only the destination city or country from the travel request.

    Travel request:
    {query}

    Return only the destination name.
    Do not add any explanation.
    """

    response = llm.invoke(prompt)

    destination = str(
        response.content
    ).strip()

    if not destination:
        raise ValueError(
            "The destination could not be extracted."
        )

    return destination