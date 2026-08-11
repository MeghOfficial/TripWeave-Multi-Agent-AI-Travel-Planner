import os
import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


# Load environment variables from .env
load_dotenv()


# Create the MCP server
mcp = FastMCP("Weather MCP Server")


# Get OpenWeather API key
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not OPENWEATHER_API_KEY:
    raise ValueError(
        "OPENWEATHER_API_KEY is missing from the .env file."
    )


# =========================
# Current Weather Tool
# =========================

@mcp.tool()
def get_current_weather(city: str):
    """Get the current weather for a city."""

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=10
        )

        data = response.json()

        if response.status_code != 200:
            return {
                "success": False,
                "error": data.get("message", "Unable to get weather data.")
            }

        return {
            "success": True,
            "city": data["name"],
            "temperature_c": data["main"]["temp"],
            "feels_like_c": data["main"]["feels_like"],
            "humidity": data["main"]["humidity"],
            "condition": data["weather"][0]["description"],
            "wind_speed": data["wind"]["speed"]
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Weather API request failed: {str(e)}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


# =========================
# Weather Forecast Tool
# =========================

@mcp.tool()
def get_forecast(city: str):
    """Get the upcoming weather forecast for a city."""

    try:
        response = requests.get(
            "https://api.openweathermap.org/data/2.5/forecast",
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric"
            },
            timeout=10
        )

        data = response.json()

        if response.status_code != 200:
            return {
                "success": False,
                "error": data.get("message", "Unable to get forecast data.")
            }

        forecast = []

        # Keep only the first 5 forecast entries
        for item in data.get("list", [])[:5]:
            forecast.append({
                "datetime": item["dt_txt"],
                "temperature_c": item["main"]["temp"],
                "weather": item["weather"][0]["description"]
            })

        return {
            "success": True,
            "city": data.get("city", {}).get("name", city),
            "forecast": forecast
        }

    except requests.RequestException as e:
        return {
            "success": False,
            "error": f"Weather API request failed: {str(e)}"
        }

    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


# =========================
# Start MCP Server
# =========================

if __name__ == "__main__":
    mcp.run()