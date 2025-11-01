"""Weather tool using Open-Meteo API (free, no API key required)."""
import httpx
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import structlog

from app.tools.registry import Tool, register_tool

logger = structlog.get_logger()

# Simple in-memory cache with TTL
_cache = {}
_cache_ttl = 300  # 5 minutes


class WeatherInput(BaseModel):
    """Input for weather tool."""
    location: str = Field(..., description="City name or 'lat,lon' coordinates")


class WeatherOutput(BaseModel):
    """Output from weather tool."""
    location: str
    temperature: float
    condition: str
    wind_speed: float
    humidity: Optional[int] = None
    forecast_summary: str


async def get_coordinates(location: str) -> tuple[float, float]:
    """Get coordinates from location name using Open-Meteo geocoding API.

    Args:
        location: City name or "lat,lon" string

    Returns:
        Tuple of (latitude, longitude)
    """
    # Check if already in lat,lon format
    if "," in location:
        try:
            lat, lon = map(float, location.split(","))
            return lat, lon
        except ValueError:
            pass

    # Use geocoding API
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

        if not data.get("results"):
            raise ValueError(f"Location '{location}' not found")

        result = data["results"][0]
        return result["latitude"], result["longitude"]


async def weather_handler(input_data: WeatherInput) -> WeatherOutput:
    """Fetch weather data from Open-Meteo API.

    Args:
        input_data: WeatherInput with location

    Returns:
        WeatherOutput with current weather and forecast
    """
    location = input_data.location

    # Check cache
    cache_key = location.lower()
    now = datetime.now().timestamp()

    if cache_key in _cache:
        cached_data, cached_time = _cache[cache_key]
        if now - cached_time < _cache_ttl:
            logger.info("weather_cache_hit", location=location)
            return cached_data

    # Get coordinates
    lat, lon = await get_coordinates(location)

    # Fetch weather data
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                "hourly": "temperature_2m,weather_code",
                "forecast_days": 1,
                "timezone": "auto",
            },
            timeout=10.0,
        )
        response.raise_for_status()
        data = response.json()

    # Parse current weather
    current = data["current"]
    temperature = current["temperature_2m"]
    humidity = current.get("relative_humidity_2m")
    wind_speed = current["wind_speed_10m"]
    weather_code = current["weather_code"]

    # Map WMO weather codes to conditions
    condition = _get_condition_from_code(weather_code)

    # Generate forecast summary for next 12 hours
    hourly = data["hourly"]
    temps_12h = hourly["temperature_2m"][:12]
    avg_temp = sum(temps_12h) / len(temps_12h)
    max_temp = max(temps_12h)
    min_temp = min(temps_12h)

    forecast_summary = (
        f"Next 12h: {min_temp:.1f}°C to {max_temp:.1f}°C (avg {avg_temp:.1f}°C)"
    )

    result = WeatherOutput(
        location=location,
        temperature=temperature,
        condition=condition,
        wind_speed=wind_speed,
        humidity=humidity,
        forecast_summary=forecast_summary,
    )

    # Cache result
    _cache[cache_key] = (result, now)

    logger.info(
        "weather_fetched",
        location=location,
        temperature=temperature,
        condition=condition,
    )

    return result


def _get_condition_from_code(code: int) -> str:
    """Convert WMO weather code to human-readable condition.

    WMO codes: https://open-meteo.com/en/docs
    """
    conditions = {
        0: "Clear",
        1: "Mainly clear",
        2: "Partly cloudy",
        3: "Overcast",
        45: "Foggy",
        48: "Foggy",
        51: "Light drizzle",
        53: "Moderate drizzle",
        55: "Dense drizzle",
        61: "Slight rain",
        63: "Moderate rain",
        65: "Heavy rain",
        71: "Slight snow",
        73: "Moderate snow",
        75: "Heavy snow",
        77: "Snow grains",
        80: "Slight rain showers",
        81: "Moderate rain showers",
        82: "Violent rain showers",
        85: "Slight snow showers",
        86: "Heavy snow showers",
        95: "Thunderstorm",
        96: "Thunderstorm with slight hail",
        99: "Thunderstorm with heavy hail",
    }
    return conditions.get(code, f"Unknown (code {code})")


# Create and register the weather tool
weather_tool = Tool(
    name="weather",
    description="Get current weather and forecast for a location. Provide city name or lat,lon coordinates.",
    input_model=WeatherInput,
    output_model=WeatherOutput,
    handler=weather_handler,
)

register_tool(weather_tool)
