"""
Real Address to Coordinates Geocoding Module for NYC Taxi Fare Intelligence Studio.
Provides structured geocoding using legitimate external geospatial APIs (OpenStreetMap Nominatim / API Key providers)
with NYC bounding box validation, rate limiting protection, and error handling.
"""

import os
import requests

# NYC Metropolitan Service Area Bounding Box (Manhattan, Brooklyn, Queens, Bronx, Staten Island + Hubs)
NYC_LAT_MIN = 40.45
NYC_LAT_MAX = 41.15
NYC_LON_MIN = -74.35
NYC_LON_MAX = -73.65

def is_in_nyc_bbox(lat: float, lon: float) -> bool:
    """Checks whether the given coordinates fall within the supported NYC metropolitan service area."""
    if lat is None or lon is None:
        return False
    return (NYC_LAT_MIN <= lat <= NYC_LAT_MAX) and (NYC_LON_MIN <= lon <= NYC_LON_MAX)

def geocode_address(address: str, api_key: str = None, timeout: float = 5.0) -> dict:
    """
    Geocodes a human-readable NYC street address or landmark to geographic coordinates.
    
    Parameters:
        address (str): The address or landmark query.
        api_key (str, optional): Optional API key if using a commercial geocoding provider.
        timeout (float): Request timeout in seconds.
        
    Returns:
        dict: Structured geocoding response containing:
            - status (str): 'success', 'empty', 'not_found', 'outside_nyc', 'rate_limited', 'network_error', 'error'
            - message (str): Human-readable status label with badge
            - formatted_address (str or None): Official resolved address string
            - latitude (float or None): Resolved latitude
            - longitude (float or None): Resolved longitude
            - confidence (float): Relevance / confidence score (0.0 to 1.0)
            - provider (str): Name of geocoding provider service
    """
    # 1. Input Validation
    if address is None or not str(address).strip():
        return {
            "status": "empty",
            "message": "❌ Address input is empty.",
            "formatted_address": None,
            "latitude": None,
            "longitude": None,
            "confidence": 0.0,
            "provider": "None"
        }
        
    query_str = str(address).strip()
    
    # 2. Key Check (from argument, environment, or Streamlit secrets if available)
    if not api_key:
        api_key = os.environ.get("GEOCODING_API_KEY", None)
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GEOCODING_API_KEY" in st.secrets:
                    api_key = st.secrets["GEOCODING_API_KEY"]
            except Exception:
                pass

    # 3. Query NYC Regional Normalization
    query_lower = query_str.lower()
    has_nyc_context = any(term in query_lower for term in ["new york", "nyc", "manhattan", "brooklyn", "queens", "bronx", "staten island", "jfk", "laguardia", "ewr", "newark"])
    normalized_query = query_str if has_nyc_context else f"{query_str}, New York, NY"

    headers = {
        "User-Agent": "NYCTaxiFareIntelligenceStudio/2.0 (Academic Research Project; contact: student@soa.ac.in)"
    }
    
    # 4. Geocoding Provider Request
    try:
        # Default Primary Service: OpenStreetMap Nominatim with NYC Viewbox bias
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": normalized_query,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
            "viewbox": f"{NYC_LON_MIN},{NYC_LAT_MAX},{NYC_LON_MAX},{NYC_LAT_MIN}",
            "bounded": 0
        }
        if api_key:
            # If a commercial key is supplied, append it (compatible with LocationIQ / Geoapify)
            params["key"] = api_key
            
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        
        if response.status_code == 429:
            return {
                "status": "rate_limited",
                "message": "⚠️ API rate limit reached. Please try again shortly.",
                "formatted_address": None,
                "latitude": None,
                "longitude": None,
                "confidence": 0.0,
                "provider": "OpenStreetMap Nominatim"
            }
            
        if response.status_code != 200:
            return {
                "status": "error",
                "message": f"❌ Geocoding request failed (HTTP {response.status_code}).",
                "formatted_address": None,
                "latitude": None,
                "longitude": None,
                "confidence": 0.0,
                "provider": "OpenStreetMap Nominatim"
            }
            
        data = response.json()
        
        if not data or len(data) == 0:
            # Fallback attempt with original query without appended terms
            if normalized_query != query_str:
                params["q"] = query_str
                fallback_res = requests.get(url, params=params, headers=headers, timeout=timeout)
                if fallback_res.status_code == 200:
                    data = fallback_res.json()
                    
            if not data or len(data) == 0:
                return {
                    "status": "not_found",
                    "message": "❌ Address not found.",
                    "formatted_address": None,
                    "latitude": None,
                    "longitude": None,
                    "confidence": 0.0,
                    "provider": "OpenStreetMap Nominatim"
                }

        top_hit = data[0]
        lat = float(top_hit["lat"])
        lon = float(top_hit["lon"])
        formatted = top_hit.get("display_name", query_str)
        confidence = float(top_hit.get("importance", 0.75))
        
        # 5. NYC Bounding Box Validation
        if not is_in_nyc_bbox(lat, lon):
            return {
                "status": "outside_nyc",
                "message": "⚠️ Location appears outside the supported NYC service area.",
                "formatted_address": formatted,
                "latitude": lat,
                "longitude": lon,
                "confidence": confidence,
                "provider": "OpenStreetMap Nominatim"
            }
            
        return {
            "status": "success",
            "message": "✓ Address found",
            "formatted_address": formatted,
            "latitude": lat,
            "longitude": lon,
            "confidence": confidence,
            "provider": "OpenStreetMap Nominatim"
        }
        
    except requests.exceptions.Timeout:
        return {
            "status": "network_error",
            "message": "⚠️ Geocoding request timed out.",
            "formatted_address": None,
            "latitude": None,
            "longitude": None,
            "confidence": 0.0,
            "provider": "OpenStreetMap Nominatim"
        }
    except requests.exceptions.RequestException:
        return {
            "status": "network_error",
            "message": "⚠️ Geocoding service temporarily unavailable.",
            "formatted_address": None,
            "latitude": None,
            "longitude": None,
            "confidence": 0.0,
            "provider": "OpenStreetMap Nominatim"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"❌ Unexpected geocoding error: {str(e)[:50]}",
            "formatted_address": None,
            "latitude": None,
            "longitude": None,
            "confidence": 0.0,
            "provider": "OpenStreetMap Nominatim"
        }
