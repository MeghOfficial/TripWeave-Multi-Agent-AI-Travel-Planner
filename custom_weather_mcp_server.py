import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Base directory of the project and load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# Initialize the MCP server
mcp = FastMCP("Weather MCP Server")

# Retrieve API key from environment
OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)

# Timeout for HTTP requests to OpenWeather
REQUEST_TIMEOUT_SECONDS = 20


def _get_api_key() -> str:
    """Retrieve the OpenWeather API key, raising an error if missing."""
    
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is missing "
            "from the project .env file."
        )
    return OPENWEATHER_API_KEY


def _request_json(
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """
    Make a GET request to the given URL with parameters,
    parse the JSON response, and handle errors.
    """
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        details = ""
        failed_response = getattr(
            exc,
            "response",
            None,
        )
        if failed_response is not None:
            details = (
                f" Response: "
                f"{failed_response.text[:500]}"
            )
        raise RuntimeError(
            f"OpenWeather request failed: "
            f"{exc}.{details}"
        ) from exc


@mcp.tool()
def get_current_weather(
    city: str,
) -> dict[str, Any]:
    """
    MCP tool: Return the current weather for a given city.
    """
    city = city.strip()
    if not city:
        raise ValueError(
            "city cannot be empty"
        )

    # Fetch current weather data from OpenWeather
    data = _request_json(
        "https://api.openweathermap.org/data/2.5/weather",
        {
            "q": city,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    # Extract and return relevant fields
    return {
        "city": data["name"],
        "temperature_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity": data["main"]["humidity"],
        "condition": data["weather"][0]["description"],
        "wind_speed": data["wind"]["speed"],
    }


@mcp.tool()
def get_forecast(
    city: str,
) -> dict[str, Any]:
    """
    MCP tool: Return the first five three-hour forecast entries for a city.
    """
    city = city.strip()
    if not city:
        raise ValueError(
            "city cannot be empty"
        )

    # Fetch 5‑day forecast data (3‑hour steps)
    data = _request_json(
        "https://api.openweathermap.org/data/2.5/forecast",
        {
            "q": city,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    # Take the first 5 forecast items and format them
    forecast = [
        {
            "datetime": item["dt_txt"],
            "temperature_c": item["main"]["temp"],
            "condition": item["weather"][0]["description"],
        }
        for item in data.get("list", [])[:5]
    ]

    return {
        "city": data.get(
            "city",
            {},
        ).get(
            "name",
            city,
        ),
        "forecast": forecast,
    }


if __name__ == "__main__":
    # Run the MCP server over stdio (used by mcp_client.py)
    mcp.run(
        transport="stdio",
    )