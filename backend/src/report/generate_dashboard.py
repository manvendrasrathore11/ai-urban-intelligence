"""
Generates a standalone interactive HTML map + a summary chart PNG from
the three pipeline outputs (road damage, accident, ANPR). No frontend,
no server — just open outputs/report/map.html in a browser.

Usage:
    python -m report.generate_dashboard
"""

import json
import shutil
from collections import Counter
from pathlib import Path

import folium
from folium.plugins import MarkerCluster

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# PATHS (same pattern as pipeline/main.py)
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVENT_DIR = PROJECT_ROOT / "outputs" / "events"

ROAD_DAMAGE_FILE = EVENT_DIR / "road_damage_events.json"
ACCIDENT_FILE = EVENT_DIR / "accident_events.json"
ANPR_FILE = EVENT_DIR / "anpr_events.json"

REPORT_DIR = PROJECT_ROOT / "outputs" / "report"
ASSETS_DIR = REPORT_DIR / "assets"

# fallback map center if no GPS data at all is found (Jaipur)
FALLBACK_CENTER = (26.9124, 75.7873)

DAMAGE_COLORS = {
    "D00": "orange",     # longitudinal crack
    "D10": "beige",      # transverse crack
    "D20": "purple",     # alligator crack
    "D40": "red",        # pothole
}


# ============================================================
# LOADING
# ============================================================

def load_json(path):
    if not path.exists():
        print(f"[REPORT] {path.name} not found — skipping that layer.")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def normalize_to_list(data):
    """
    Handles both possible shapes for the road-damage/accident JSON:
      - a bare list of records: [ {...}, {...} ]
      - a dict keyed by id:      { "ROAD_DAMAGE_001": {...}, ... }
    Always returns a plain list of record dicts.
    """
    if data is None:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [item for item in data.values() if isinstance(item, dict)]
    return []


def copy_evidence(image_path_str, subfolder):
    """Copies an evidence image next to the report so the HTML can
    reference it with a relative path (works even if you zip/move the
    whole outputs/report folder)."""

    if not image_path_str:
        return None

    src = Path(image_path_str)
    if not src.exists():
        return None

    dest_dir = ASSETS_DIR / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    try:
        shutil.copy(src, dest)
    except Exception as e:
        print(f"[REPORT] Could not copy evidence image {src}: {e}")
        return None

    return f"assets/{subfolder}/{src.name}"


# ============================================================
# MAP
# ============================================================

def build_map(road_damage, accidents, anpr_plates):

    all_coords = []

    for d in road_damage:
        loc = d.get("location", {})
        if loc.get("latitude") is not None and loc.get("longitude") is not None:
            all_coords.append((loc["latitude"], loc["longitude"]))

    for a in accidents:
        gps = a.get("gps", {})
        if gps.get("latitude") is not None and gps.get("longitude") is not None:
            all_coords.append((gps["latitude"], gps["longitude"]))

    for p in anpr_plates:
        gps = p.get("gps", {})
        if gps.get("latitude") is not None and gps.get("longitude") is not None:
            all_coords.append((gps["latitude"], gps["longitude"]))

    if all_coords:
        center_lat = sum(c[0] for c in all_coords) / len(all_coords)
        center_lon = sum(c[1] for c in all_coords) / len(all_coords)
    else:
        center_lat, center_lon = FALLBACK_CENTER

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="cartodbpositron")

    # ------------------------------------------------------
    # ROAD DAMAGE layer
    # ------------------------------------------------------
    damage_layer = folium.FeatureGroup(name=f"Road Damage ({len(road_damage)})", show=True)

    for d in road_damage:
        loc = d.get("location", {})
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if lat is None or lon is None:
            continue

        color = DAMAGE_COLORS.get(d.get("damage_type"), "gray")
        confirmations = d.get("total_confirmations", 1)

        popup_html = f"""
        <b>{d.get('damage_name', d.get('damage_type'))}</b><br>
        ID: {d.get('damage_id')}<br>
        Confirmations: {confirmations}<br>
        First seen: {d.get('first_detected_date', '-')}<br>
        Last seen: {d.get('last_detected_date', '-')}<br>
        Bus: {', '.join(d.get('bus_ids', []))}<br>
        Status: {d.get('status', 'MONITORING')}
        """

        radius = 5 + min(confirmations, 150) / 12

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1.5,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{d.get('damage_name', d.get('damage_type'))} ({confirmations}x)",
        ).add_to(damage_layer)

    damage_layer.add_to(m)

    # ------------------------------------------------------
    # ACCIDENT layer (clustered — repeated detections stack up)
    # ------------------------------------------------------
    accident_layer = MarkerCluster(name=f"Accidents ({len(accidents)})")

    for a in accidents:
        gps = a.get("gps", {})
        lat, lon = gps.get("latitude"), gps.get("longitude")
        if lat is None or lon is None:
            continue

        img_rel = copy_evidence(a.get("evidence", {}).get("image"), "accidents")
        img_html = f'<br><img src="{img_rel}" width="220">' if img_rel else ""

        services = a.get("nearest_services") or {}
        services_html = ""
        if services.get("police"):
            p = services["police"]
            services_html += f"<br>Nearest police: {p['name']} ({p['distance_km']} km)"
        if services.get("hospital"):
            h = services["hospital"]
            services_html += f"<br>Nearest hospital: {h['name']} ({h['distance_km']} km)"

        alert_sent = (a.get("email_alert") or {}).get("sent")

        popup_html = f"""
        <b>ACCIDENT \u2014 {a.get('severity', '')}</b><br>
        Event: {a.get('event_id')}<br>
        Bus: {a.get('bus_id')}<br>
        Time: {a.get('timestamp')}<br>
        Vehicles involved: {a.get('vehicle_count')}<br>
        Collision score: {a.get('collision_count')}<br>
        Alert sent: {'Yes' if alert_sent else 'No'}{services_html}{img_html}
        """

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"Accident \u2014 {a.get('severity')}",
        ).add_to(accident_layer)

    accident_layer.add_to(m)

    # ------------------------------------------------------
    # ANPR layer
    # ------------------------------------------------------
    anpr_layer = folium.FeatureGroup(name=f"Number Plates ({len(anpr_plates)})", show=True)

    for p in anpr_plates:
        gps = p.get("gps", {})
        lat, lon = gps.get("latitude"), gps.get("longitude")
        if lat is None or lon is None:
            continue

        img_rel = copy_evidence(p.get("evidence_image"), "anpr")
        img_html = f'<br><img src="{img_rel}" width="200">' if img_rel else ""

        conf = p.get("ocr_confidence", 0) or 0

        popup_html = f"""
        <b>{p.get('plate_text')}</b><br>
        Vehicle: {p.get('vehicle_class')}<br>
        OCR confidence: {conf:.0%}<br>
        Reads: {p.get('read_count')} (votes: {p.get('plate_vote_count')})<br>
        Bus: {p.get('bus_id')}<br>
        Time: {p.get('timestamp')}{img_html}
        """

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="blue", icon="car", prefix="fa"),
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=p.get("plate_text"),
        ).add_to(anpr_layer)

    anpr_layer.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)

    # ------------------------------------------------------
    # Floating stats box
    # ------------------------------------------------------
    stats_html = f"""
    <div style="position: fixed; top: 12px; left: 60px; z-index: 9999;
                background: white; padding: 12px 20px; border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.35); font-family: Arial, sans-serif;
                font-size: 13px; line-height: 1.6;">
        <b style="font-size: 15px;">AI Urban Intelligence \u2014 Live Report</b><br>
        Road Damage Points: <b>{len(road_damage)}</b><br>
        Accident Events: <b>{len(accidents)}</b><br>
        Unique Plates Read: <b>{len(anpr_plates)}</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(stats_html))

    return m


# ============================================================
# SUMMARY CHART
# ============================================================

def build_summary_chart(road_damage, accidents, anpr_plates):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    type_counts = Counter(
        d.get("damage_name", d.get("damage_type", "UNKNOWN")) for d in road_damage
    )
    if type_counts:
        axes[0].bar(list(type_counts.keys()), list(type_counts.values()), color="#D97B29")
    axes[0].set_title("Road Damage by Type")
    axes[0].tick_params(axis="x", rotation=25)

    sev_counts = Counter(a.get("severity", "UNKNOWN") for a in accidents)
    if sev_counts:
        axes[1].bar(list(sev_counts.keys()), list(sev_counts.values()), color="#B23A3A")
    axes[1].set_title("Accidents by Severity")

    class_counts = Counter(p.get("vehicle_class", "unknown") for p in anpr_plates)
    if class_counts:
        axes[2].bar(list(class_counts.keys()), list(class_counts.values()), color="#1F6FBF")
    axes[2].set_title("Plates Read by Vehicle Class")

    plt.tight_layout()

    out_path = REPORT_DIR / "summary.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[REPORT] Summary chart saved to {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    road_damage = normalize_to_list(load_json(ROAD_DAMAGE_FILE))

    accidents = normalize_to_list(load_json(ACCIDENT_FILE))

    anpr_data = load_json(ANPR_FILE)
    anpr_plates = (anpr_data or {}).get("plates", [])

    print(f"[REPORT] Road damage points: {len(road_damage)}")
    print(f"[REPORT] Accident events: {len(accidents)}")
    print(f"[REPORT] Unique plates: {len(anpr_plates)}")

    m = build_map(road_damage, accidents, anpr_plates)
    map_path = REPORT_DIR / "map.html"
    m.save(str(map_path))
    print(f"[REPORT] Map saved to {map_path}")

    build_summary_chart(road_damage, accidents, anpr_plates)

    print("\nOpen this file in any browser:")
    print(map_path)


if __name__ == "__main__":
    main()
