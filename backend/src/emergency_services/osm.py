import requests
from math import radians, sin, cos, sqrt, atan2


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS coordinates.

    Returns:
        Distance in kilometers.
    """

    R = 6371.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


# ============================================================
# FIND NEAREST EMERGENCY SERVICES
# ============================================================

def alert_nearest_services(location):
    """
    Find the nearest police station and hospital
    using OpenStreetMap Overpass API.

    Args:
        location:
            Tuple containing:
            (latitude, longitude)

    Returns:
        nearest_police_station,
        nearest_hospital
    """

    lat, lon = location

    # --------------------------------------------------------
    # OVERPASS API
    # --------------------------------------------------------

    overpass_url = (
        "https://overpass-api.de/api/interpreter"
    )

    overpass_query = f"""
    [out:json][timeout:25];
    (
        node["amenity"="police"]
        (around:5000,{lat},{lon});

        node["amenity"="hospital"]
        (around:5000,{lat},{lon});
    );
    out body;
    """

    headers = {
        "User-Agent":
            "AI-Urban-Intelligence/1.0"
    }

    # --------------------------------------------------------
    # API REQUEST
    # --------------------------------------------------------

    try:

        response = requests.post(
            overpass_url,
            data={
                "data": overpass_query
            },
            headers=headers,
            timeout=30
        )

        print(
            "Overpass status:",
            response.status_code
        )

        response.raise_for_status()

        data = response.json()

    except requests.exceptions.RequestException as e:

        print(
            "OpenStreetMap request failed:",
            e
        )

        return None, None

    except ValueError:

        print(
            "OpenStreetMap returned invalid JSON."
        )

        return None, None

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    police_stations = []
    hospitals = []

    for element in data.get("elements", []):

        if "tags" not in element:
            continue

        name = element["tags"].get(
            "name",
            "Unnamed"
        )

        lat_osm = element.get("lat")
        lon_osm = element.get("lon")

        if lat_osm is None or lon_osm is None:
            continue

        amenity = element["tags"].get(
            "amenity"
        )

        # ----------------------------------------------------
        # POLICE
        # ----------------------------------------------------

        if amenity == "police":

            distance = haversine(
                lat,
                lon,
                lat_osm,
                lon_osm
            )

            police_stations.append({

                "name": name,

                "latitude": lat_osm,

                "longitude": lon_osm,

                "distance_km":
                    round(distance, 2)
            })

        # ----------------------------------------------------
        # HOSPITAL
        # ----------------------------------------------------

        elif amenity == "hospital":

            distance = haversine(
                lat,
                lon,
                lat_osm,
                lon_osm
            )

            hospitals.append({

                "name": name,

                "latitude": lat_osm,

                "longitude": lon_osm,

                "distance_km":
                    round(distance, 2)
            })

    # --------------------------------------------------------
    # FIND NEAREST
    # --------------------------------------------------------

    nearest_police_station = min(
        police_stations,
        key=lambda p: p["distance_km"],
        default=None
    )

    nearest_hospital = min(
        hospitals,
        key=lambda h: h["distance_km"],
        default=None
    )

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    if nearest_police_station:

        print(
            f"Nearest police station: "
            f"{nearest_police_station['name']} "
            f"("
            f"{nearest_police_station['distance_km']}"
            f" km)"
        )

    else:

        print(
            "No police station found nearby."
        )

    if nearest_hospital:

        print(
            f"Nearest hospital: "
            f"{nearest_hospital['name']} "
            f"("
            f"{nearest_hospital['distance_km']}"
            f" km)"
        )

    else:

        print(
            "No hospital found nearby."
        )

    return (
        nearest_police_station,
        nearest_hospital
    )