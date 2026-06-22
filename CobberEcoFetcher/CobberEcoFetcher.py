"""
CobberEcoFetcher.py

A standalone PyQt6 application for exploring ecological data from:
- GBIF biodiversity occurrence records
- iNaturalist community observations

Rebuilt classroom version for the ML for Ecology data-access chapter.

Major classroom design choices:
- Launch directly into the main workspace.
- Use one Build your query panel.
- Include a Scientific question field.
- Use source-specific filters for GBIF and iNaturalist.
- Use student-facing result tabs: Query summary, Returned records,
  Map of records, and Media preview.
- Keep raw metadata in the exported JSON file, but do not display a raw JSON tab
  in the main student workflow.
- Export a CSV file, a JSON metadata file, and a TXT query-summary file.

Dependencies:
    pip install PyQt6 requests
Optional for live map support:
    pip install PyQt6-WebEngine
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PyQt6.QtCore import QObject, QRunnable, Qt, QThreadPool, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QSplitter,
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView

    WEBENGINE_AVAILABLE = True
except Exception:
    QWebEngineView = None  # type: ignore
    WEBENGINE_AVAILABLE = False

APP_TITLE = "CobberEcoFetcher"
APP_VERSION = "5.0 classroom rebuild"
USER_AGENT = "CobberEcoFetcher/5.0 (educational prototype; contact instructor)"
GBIF_BASE = "https://api.gbif.org/v1"
INAT_BASE = "https://api.inaturalist.org/v1"
REQUEST_TIMEOUT = 20

SAMPLE_QUERIES = {
    "GBIF - Broad bur oak search": {
        "source": "GBIF",
        "scientific_question": "Where has bur oak been recorded?",
        "taxon": "Quercus macrocarpa",
        "place": "",
        "start": "2000-01-01",
        "end": datetime.now().strftime("%Y-%m-%d"),
        "coords_only": True,
        "photos_only": True,
        "research_grade": True,
        "limit": 100,
    },
    "GBIF - Bur oak in Minnesota": {
        "source": "GBIF",
        "scientific_question": "Where has bur oak been recorded in Minnesota since 2000?",
        "taxon": "Quercus macrocarpa",
        "place": "Minnesota",
        "start": "2000-01-01",
        "end": datetime.now().strftime("%Y-%m-%d"),
        "coords_only": True,
        "photos_only": True,
        "research_grade": True,
        "limit": 100,
    },
    "iNaturalist - Older monarch observations in Iowa": {
        "source": "iNaturalist",
        "scientific_question": "How many older monarch butterfly observations are available in Iowa?",
        "taxon": "Danaus plexippus",
        "place": "Iowa",
        "start": "2008-01-01",
        "end": "2012-12-31",
        "coords_only": True,
        "photos_only": True,
        "research_grade": True,
        "limit": 50,
    },
    "iNaturalist - Recent monarch observations in Iowa": {
        "source": "iNaturalist",
        "scientific_question": "How many recent monarch butterfly observations are available in Iowa?",
        "taxon": "Danaus plexippus",
        "place": "Iowa",
        "start": "2020-01-01",
        "end": "2025-12-31",
        "coords_only": True,
        "photos_only": True,
        "research_grade": True,
        "limit": 50,
    },
}


@dataclass
class ResultPackage:
    source: str
    title: str
    scientific_question: str
    query_summary: Dict[str, Any]
    fetched_at: str
    preview_rows: List[Dict[str, Any]]
    raw_metadata: Dict[str, Any]
    total_matching: Any = "unknown"
    map_points: List[Dict[str, Any]] = field(default_factory=list)
    media_items: List[Dict[str, Any]] = field(default_factory=list)
    summary_details: Dict[str, Any] = field(default_factory=dict)
    export_name: str = "dataset"


def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    response = requests.get(url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def bool_word(value: bool) -> str:
    return "yes" if value else "no"


def clean_filename(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value.strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "dataset"


def normalize_date_for_query(date_str: str) -> str:
    return date_str[:10]


class GBIFAdapter:
    @staticmethod
    def fetch_preview(query: Dict[str, Any], progress_cb=None) -> ResultPackage:
        taxon = (query.get("taxon") or "").strip()
        if not taxon:
            raise ValueError("Please enter an organism or taxon for GBIF.")

        if progress_cb:
            progress_cb(f"Matching taxon '{taxon}' in GBIF...")
        match_json = safe_get_json(f"{GBIF_BASE}/species/match", {"name": taxon})
        usage_key = match_json.get("usageKey")
        accepted_name = match_json.get("scientificName") or match_json.get("canonicalName") or taxon

        limit = min(int(query.get("limit", 100)), 200)
        params: Dict[str, Any] = {"limit": limit, "offset": 0}
        if usage_key:
            params["taxonKey"] = usage_key
        else:
            params["scientificName"] = taxon

        place = (query.get("place") or "").strip()
        if place:
            if len(place) == 2:
                params["country"] = place.upper()
            else:
                params["stateProvince"] = place

        start = query.get("start") or ""
        end = query.get("end") or ""
        if start or end:
            start_year = start[:4] if start else "0000"
            end_year = end[:4] if end else "9999"
            params["year"] = f"{start_year},{end_year}"

        if query.get("coords_only"):
            params["hasCoordinate"] = "true"

        if progress_cb:
            progress_cb("Fetching occurrence records from GBIF...")
        occ_json = safe_get_json(f"{GBIF_BASE}/occurrence/search", params)
        results = occ_json.get("results", [])

        preview_rows: List[Dict[str, Any]] = []
        points: List[Dict[str, Any]] = []
        media_items: List[Dict[str, Any]] = []
        countries: Dict[str, int] = {}

        for item in results:
            country = item.get("country") or item.get("countryCode") or ""
            if country:
                countries[country] = countries.get(country, 0) + 1

            row = {
                "source": "GBIF",
                "record_id": item.get("key", ""),
                "scientific_name": item.get("scientificName") or item.get("species") or "",
                "accepted_name": accepted_name,
                "basis_of_record": item.get("basisOfRecord", ""),
                "event_date": item.get("eventDate") or item.get("year") or "",
                "country": country,
                "state_province": item.get("stateProvince", ""),
                "decimalLatitude": item.get("decimalLatitude", ""),
                "decimalLongitude": item.get("decimalLongitude", ""),
                "license": item.get("license", ""),
                "dataset": item.get("datasetName", ""),
            }
            preview_rows.append(row)

            lat = item.get("decimalLatitude")
            lon = item.get("decimalLongitude")
            if lat is not None and lon is not None:
                points.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "label": row["scientific_name"] or accepted_name,
                        "popup": (
                            f"GBIF occurrence #{item.get('key')}<br>"
                            f"{html_escape(row['scientific_name'] or accepted_name)}<br>"
                            f"{html_escape(country)}"
                        ),
                    }
                )

            for media in (item.get("media") or [])[:1]:
                media_items.append(
                    {
                        "title": row["scientific_name"] or accepted_name,
                        "url": media.get("identifier") or media.get("references") or "",
                        "link": media.get("references") or media.get("identifier") or "",
                        "type": media.get("type", "media"),
                    }
                )

        top_countries = ", ".join(
            f"{k} ({v})" for k, v in sorted(countries.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ) or "None in preview"

        summary_details = {
            "taxon_matched": accepted_name,
            "taxon_key": usage_key or "not provided",
            "preview_count": len(preview_rows),
            "total_matching": occ_json.get("count", "unknown"),
            "common_regions": top_countries,
        }

        return ResultPackage(
            source="GBIF",
            title=f"GBIF: {accepted_name}",
            scientific_question=query.get("scientific_question", ""),
            query_summary=query,
            fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            preview_rows=preview_rows,
            raw_metadata={"match": match_json, "occurrence_search": occ_json},
            total_matching=occ_json.get("count", "unknown"),
            map_points=points,
            media_items=media_items,
            summary_details=summary_details,
            export_name=f"gbif_{clean_filename(str(accepted_name))}",
        )


class INaturalistAdapter:
    @staticmethod
    def _resolve_place(place_text: str) -> Tuple[Optional[int], Optional[str], Dict[str, Any]]:
        if not place_text.strip():
            return None, None, {}
        data = safe_get_json(f"{INAT_BASE}/places/autocomplete", {"q": place_text, "per_page": 1})
        results = data.get("results", [])
        if not results:
            return None, None, data
        place = results[0]
        return place.get("id"), place.get("display_name") or place.get("name"), data

    @staticmethod
    def fetch_preview(query: Dict[str, Any], progress_cb=None) -> ResultPackage:
        taxon = (query.get("taxon") or "").strip()
        if not taxon:
            raise ValueError("Please enter an organism or taxon for iNaturalist.")

        place_text = query.get("place") or ""
        place_id = None
        place_name = None
        place_lookup: Dict[str, Any] = {}
        if place_text.strip():
            if progress_cb:
                progress_cb(f"Resolving iNaturalist place '{place_text}'...")
            place_id, place_name, place_lookup = INaturalistAdapter._resolve_place(place_text)

        limit = min(int(query.get("limit", 50)), 200)
        params: Dict[str, Any] = {
            "taxon_name": taxon,
            "per_page": limit,
            "page": 1,
            "order_by": "observed_on",
            "order": "desc",
        }
        if query.get("photos_only"):
            params["photos"] = "true"
        if query.get("research_grade"):
            params["quality_grade"] = "research"
        if place_id is not None:
            params["place_id"] = place_id

        start = normalize_date_for_query(query.get("start") or "")
        end = normalize_date_for_query(query.get("end") or "")
        if start:
            params["d1"] = start
        if end:
            params["d2"] = end

        if progress_cb:
            progress_cb("Fetching observations from iNaturalist...")
        data = safe_get_json(f"{INAT_BASE}/observations", params)
        results = data.get("results", [])

        preview_rows: List[Dict[str, Any]] = []
        points: List[Dict[str, Any]] = []
        media_items: List[Dict[str, Any]] = []
        places_seen: Dict[str, int] = {}
        photo_count = 0
        taxon_matched = taxon

        for item in results:
            taxon_obj = item.get("taxon") or {}
            if isinstance(taxon_obj, dict):
                taxon_matched = taxon_obj.get("name") or taxon_matched
            user = item.get("user") or {}
            place_guess = item.get("place_guess") or ""
            if place_guess:
                places_seen[place_guess] = places_seen.get(place_guess, 0) + 1
            photos = item.get("photos") or []
            if photos:
                photo_count += 1

            coords = item.get("geojson", {}).get("coordinates", [None, None])
            lon = coords[0] if len(coords) > 0 else None
            lat = coords[1] if len(coords) > 1 else None

            row = {
                "source": "iNaturalist",
                "observation_id": item.get("id", ""),
                "scientific_name": taxon_matched,
                "common_name": taxon_obj.get("preferred_common_name", "") if isinstance(taxon_obj, dict) else "",
                "observer": user.get("login") or "",
                "observed_on": item.get("observed_on") or item.get("time_observed_at") or "",
                "place_guess": place_guess,
                "quality_grade": item.get("quality_grade", ""),
                "photo_count": len(photos),
                "latitude": lat,
                "longitude": lon,
                "license": item.get("license_code", ""),
                "uri": item.get("uri", ""),
            }
            preview_rows.append(row)

            if lat is not None and lon is not None:
                points.append(
                    {
                        "lat": lat,
                        "lon": lon,
                        "label": row["scientific_name"],
                        "popup": (
                            f"iNaturalist observation #{item.get('id')}<br>"
                            f"{html_escape(row['scientific_name'])}<br>"
                            f"{html_escape(place_guess)}"
                        ),
                    }
                )

            if photos:
                first = photos[0]
                thumb = first.get("url") or ""
                media_items.append(
                    {
                        "title": f"Observation {item.get('id')}",
                        "url": thumb.replace("square", "medium") if thumb else "",
                        "link": item.get("uri", ""),
                        "type": "photo",
                    }
                )

        top_places = ", ".join(
            f"{k} ({v})" for k, v in sorted(places_seen.items(), key=lambda kv: kv[1], reverse=True)[:5]
        ) or "None in preview"

        summary_details = {
            "taxon_matched": taxon_matched,
            "place_matched": place_name or place_text or "not specified",
            "preview_count": len(preview_rows),
            "total_matching": data.get("total_results", "unknown"),
            "photos_in_preview": photo_count,
            "common_places": top_places,
        }

        return ResultPackage(
            source="iNaturalist",
            title=f"iNaturalist: {taxon}",
            scientific_question=query.get("scientific_question", ""),
            query_summary=query,
            fetched_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            preview_rows=preview_rows,
            raw_metadata={"place_lookup": place_lookup, "observations": data},
            total_matching=data.get("total_results", "unknown"),
            map_points=points,
            media_items=media_items,
            summary_details=summary_details,
            export_name=f"inat_{clean_filename(taxon)}",
        )


class WorkerSignals(QObject):
    finished = pyqtSignal()
    result = pyqtSignal(object)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)


class FetchWorker(QRunnable):
    def __init__(self, source: str, query: Dict[str, Any]):
        super().__init__()
        self.source = source
        self.query = query
        self.signals = WorkerSignals()

    def run(self):
        try:
            def progress(message: str):
                self.signals.progress.emit(message)

            if self.source == "GBIF":
                result = GBIFAdapter.fetch_preview(self.query, progress)
            else:
                result = INaturalistAdapter.fetch_preview(self.query, progress)
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class HtmlPanel(QWidget):
    def __init__(self, html: str = ""):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(html)
        layout.addWidget(self.browser)


class SimpleTablePanel(QWidget):
    def __init__(self, rows: List[Dict[str, Any]]):
        super().__init__()
        layout = QVBoxLayout(self)
        self.table = QTableWidget()
        layout.addWidget(self.table)
        self.populate(rows)

    def populate(self, rows: List[Dict[str, Any]]):
        self.table.clear()
        if not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return
        columns = list(rows[0].keys())
        self.table.setColumnCount(len(columns))
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, col in enumerate(columns):
                value = "" if row.get(col) is None else str(row.get(col))
                self.table.setItem(r, c, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)


class CoordinatePlotFallback(QWidget):
    def __init__(self, points: List[Dict[str, Any]]):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(self._build_html(points))
        layout.addWidget(browser)

    def _build_html(self, points: List[Dict[str, Any]]) -> str:
        valid = []
        for point in points:
            try:
                lat = float(point.get("lat"))
                lon = float(point.get("lon"))
                valid.append({**point, "lat": lat, "lon": lon})
            except Exception:
                continue

        if not valid:
            return "<h3>Map of records</h3><p>No coordinates were available for this result.</p>"

        lats = [p["lat"] for p in valid]
        lons = [p["lon"] for p in valid]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)

        lat_pad = max(1.0, (max_lat - min_lat) * 0.08)
        lon_pad = max(1.0, (max_lon - min_lon) * 0.08)
        min_lat -= lat_pad
        max_lat += lat_pad
        min_lon -= lon_pad
        max_lon += lon_pad

        width = 880
        height = 520
        left = 78
        top = 28
        plot_w = width - 140
        plot_h = height - 86

        def x_of(lon: float) -> float:
            return left + (lon - min_lon) / max(max_lon - min_lon, 1e-9) * plot_w

        def y_of(lat: float) -> float:
            return top + plot_h - (lat - min_lat) / max(max_lat - min_lat, 1e-9) * plot_h

        grid_lines: List[str] = []
        for i in range(6):
            x = left + i * plot_w / 5
            lon_val = min_lon + i * (max_lon - min_lon) / 5
            grid_lines.append(
                f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{top + plot_h}' stroke='#d9d9d9' stroke-width='1'/>"
            )
            grid_lines.append(
                f"<text x='{x:.1f}' y='{top + plot_h + 24}' text-anchor='middle' font-size='12' fill='#555'>{lon_val:.1f}°</text>"
            )
        for j in range(6):
            y = top + j * plot_h / 5
            lat_val = max_lat - j * (max_lat - min_lat) / 5
            grid_lines.append(
                f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' stroke='#d9d9d9' stroke-width='1'/>"
            )
            grid_lines.append(
                f"<text x='{left - 10}' y='{y + 4:.1f}' text-anchor='end' font-size='12' fill='#555'>{lat_val:.1f}°</text>"
            )

        circles: List[str] = []
        for point in valid[:500]:
            x = x_of(point["lon"])
            y = y_of(point["lat"])
            popup = html_escape(point.get("popup", point.get("label", "Record")))
            circles.append(
                f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5.5' fill='#2b7bbb' fill-opacity='0.78' stroke='white' stroke-width='1.2'>"
                f"<title>{popup}</title></circle>"
            )

        return f"""
        <html>
        <body style='font-family:Arial, sans-serif; margin:0; padding:10px; background:#fafafa;'>
          <div style='margin-bottom:8px;'><b>Map of records</b> - coordinate plot fallback.</div>
          <svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' style='background:white; border:1px solid #d0d0d0;'>
            <rect x='{left}' y='{top}' width='{plot_w}' height='{plot_h}' fill='#f6fbff' stroke='#888' stroke-width='1.2'/>
            {''.join(grid_lines)}
            {''.join(circles)}
            <text x='{left + plot_w / 2:.1f}' y='{height - 12}' text-anchor='middle' font-size='13' fill='#444'>Longitude</text>
            <text x='18' y='{top + plot_h / 2:.1f}' text-anchor='middle' font-size='13' fill='#444' transform='rotate(-90 18 {top + plot_h / 2:.1f})'>Latitude</text>
          </svg>
          <div style='font-size:12px; color:#555; margin-top:8px;'>
            Showing {len(valid)} mapped point(s). Hover over a point to see its record label.
          </div>
        </body>
        </html>
        """


class LazyMapPanel(QWidget):
    def __init__(self, points: List[Dict[str, Any]]):
        super().__init__()
        self.points = points
        self.built = False
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QLabel("Select the Map of records tab to load the map.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.placeholder)

    def ensure_built(self):
        if self.built:
            return
        self.built = True
        self.layout.removeWidget(self.placeholder)
        self.placeholder.deleteLater()
        if WEBENGINE_AVAILABLE:
            view = QWebEngineView()
            view.setHtml(self._build_leaflet_html(self.points))
            self.layout.addWidget(view)
        else:
            self.layout.addWidget(CoordinatePlotFallback(self.points))

    def _build_leaflet_html(self, points: List[Dict[str, Any]]) -> str:
        valid = []
        for point in points:
            try:
                lat = float(point.get("lat"))
                lon = float(point.get("lon"))
                valid.append({**point, "lat": lat, "lon": lon})
            except Exception:
                continue

        markers_js = []
        bounds_entries = []
        for point in valid[:500]:
            markers_js.append(
                f"L.marker([{point['lat']}, {point['lon']}]).addTo(map).bindPopup({json.dumps(point.get('popup', point.get('label', 'Record')))});"
            )
            bounds_entries.append(f"[{point['lat']}, {point['lon']}]")

        if len(valid) == 1:
            center_script = f"map.setView([{valid[0]['lat']}, {valid[0]['lon']}], 8);"
        elif len(valid) > 1:
            center_script = (
                f"var bounds = L.latLngBounds([{', '.join(bounds_entries)}]);"
                "map.fitBounds(bounds.pad(0.12));"
            )
        else:
            center_script = "map.setView([39.5, -98.35], 3);"

        count_note = f"{len(valid)} mapped point(s)" if valid else "No coordinates available"

        return f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body {{ height: 100%; margin: 0; padding: 0; background: #f4f4f4; }}
    #map {{ height: 100%; width: 100%; }}
    .badge {{ position:absolute; top:12px; right:12px; z-index:9999; background:white; padding:8px 10px; border-radius:6px; border:1px solid #ddd; font-family:Arial,sans-serif; color:#555; }}
  </style>
</head>
<body>
  <div class="badge">{html_escape(count_note)}</div>
  <div id="map"></div>
  <script>
    var map = L.map('map', {{zoomControl: true, worldCopyJump: false}});
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      subdomains: 'abcd',
      maxZoom: 19,
      noWrap: true,
      attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
    }}).addTo(map);
    {''.join(markers_js)}
    {center_script}
  </script>
</body>
</html>
        """


class MediaPanel(QWidget):
    def __init__(self, items: List[Dict[str, Any]]):
        super().__init__()
        layout = QVBoxLayout(self)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        if not items:
            browser.setHtml(
                "<h3>Media preview</h3>"
                "<p>No media preview is available for this result.</p>"
                "<p>For GBIF, some occurrence records include images and some do not. "
                "For iNaturalist, requiring photos usually makes this tab more useful.</p>"
            )
        else:
            cards = []
            for item in items[:30]:
                url = item.get("url", "")
                link = item.get("link", url)
                title = html_escape(item.get("title", "Item"))
                if url:
                    cards.append(
                        f"<div style='margin-bottom:14px; padding-bottom:10px; border-bottom:1px solid #ddd;'>"
                        f"<b>{title}</b><br>"
                        f"<a href='{html_escape(link)}'><img src='{html_escape(url)}' width='220'></a>"
                        f"</div>"
                    )
                else:
                    cards.append(f"<div><b>{title}</b></div>")
            browser.setHtml("<h3>Media preview</h3>" + "".join(cards))
        layout.addWidget(browser)


class SampleQueryDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load sample query")
        self.resize(560, 380)
        layout = QVBoxLayout(self)
        self.combo = QComboBox()
        self.combo.addItems(list(SAMPLE_QUERIES.keys()))
        self.details = QTextBrowser()
        self.details.setHtml(self._details_html(self.combo.currentText()))
        self.combo.currentTextChanged.connect(lambda text: self.details.setHtml(self._details_html(text)))
        layout.addWidget(QLabel("Choose a sample query:"))
        layout.addWidget(self.combo)
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _details_html(self, name: str) -> str:
        payload = SAMPLE_QUERIES.get(name, {})
        bits = "".join(f"<li><b>{html_escape(k)}:</b> {html_escape(v)}</li>" for k, v in payload.items())
        return f"<h3>{html_escape(name)}</h3><ul>{bits}</ul>"

    def get_payload(self) -> Optional[Dict[str, Any]]:
        if self.exec() == QDialog.DialogCode.Accepted:
            return SAMPLE_QUERIES.get(self.combo.currentText())
        return None


class CobberEcoFetcherApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.threadpool = QThreadPool()
        self.results: List[ResultPackage] = []

        self.cobber_maroon = QColor(108, 29, 69)
        self.eco_green = QColor(38, 102, 66)
        self.base_font = QFont("Lato", 10)
        self.setFont(self.base_font)

        self.setWindowTitle(APP_TITLE)
        self._set_laptop_friendly_geometry()
        self._build_ui()
        self._build_menu()
        self.statusBar().showMessage("Ready. Build a query and fetch records.")

    def _set_laptop_friendly_geometry(self):
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1180, 720)
            return
        geom = screen.availableGeometry()
        width = min(1220, max(1040, int(geom.width() * 0.92)))
        height = min(730, max(640, int(geom.height() * 0.88)))
        self.resize(width, height)
        x = geom.x() + max(0, (geom.width() - width) // 2)
        y = geom.y() + max(0, (geom.height() - height) // 2)
        self.move(x, y)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("File")
        sample_action = QAction("Load sample query", self)
        sample_action.triggered.connect(self.load_sample_query)
        export_action = QAction("Export dataset", self)
        export_action.triggered.connect(self.export_current_result)
        file_menu.addAction(sample_action)
        file_menu.addAction(export_action)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.addWidget(self._build_main_page())
        self.setStatusBar(QStatusBar())

    def _build_main_page(self) -> QWidget:
        page = QWidget()
        layout = QHBoxLayout(page)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left = QWidget()
        left.setMinimumWidth(360)
        left.setMaximumWidth(430)
        left_layout = QVBoxLayout(left)

        form_box = QGroupBox("Build your query")
        form_layout = QVBoxLayout(form_box)

        self.scientific_question = QLineEdit()
        self.scientific_question.setPlaceholderText("Example: Where have bur oaks been recorded in Minnesota since 2000?")
        form_layout.addWidget(QLabel("Scientific question:"))
        form_layout.addWidget(self.scientific_question)

        self.source_combo = QComboBox()
        self.source_combo.addItems(["GBIF", "iNaturalist"])
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        form_layout.addWidget(QLabel("Data source:"))
        form_layout.addWidget(self.source_combo)

        self.taxon_input = QLineEdit()
        self.taxon_input.setPlaceholderText("scientific or common name")
        form_layout.addWidget(QLabel("Organism or taxon:"))
        form_layout.addWidget(self.taxon_input)

        self.place_input = QLineEdit()
        self.place_input.setPlaceholderText("country, state, region, or place name")
        form_layout.addWidget(QLabel("Place filter:"))
        form_layout.addWidget(self.place_input)

        date_row = QHBoxLayout()
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(date(2000, 1, 1))
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(date.today())
        date_row.addWidget(self.start_date)
        date_row.addWidget(self.end_date)
        form_layout.addWidget(QLabel("Start date and end date:"))
        form_layout.addLayout(date_row)

        filter_box = QGroupBox("Source-specific filters")
        filter_layout = QVBoxLayout(filter_box)

        self.gbif_label = QLabel("GBIF filters")
        self.gbif_label.setStyleSheet("font-weight: bold;")
        self.gbif_coords = QCheckBox("Require coordinates")
        self.gbif_coords.setChecked(True)
        filter_layout.addWidget(self.gbif_label)
        filter_layout.addWidget(self.gbif_coords)

        self.inat_label = QLabel("iNaturalist filters")
        self.inat_label.setStyleSheet("font-weight: bold;")
        self.inat_photos = QCheckBox("Require photos")
        self.inat_photos.setChecked(True)
        self.inat_research = QCheckBox("Research grade only")
        self.inat_research.setChecked(True)
        filter_layout.addWidget(self.inat_label)
        filter_layout.addWidget(self.inat_photos)
        filter_layout.addWidget(self.inat_research)
        form_layout.addWidget(filter_box)

        self.limit_input = QSpinBox()
        self.limit_input.setRange(10, 200)
        self.limit_input.setValue(100)
        form_layout.addWidget(QLabel("Preview size:"))
        form_layout.addWidget(self.limit_input)

        btn_row = QHBoxLayout()
        self.fetch_btn = QPushButton("Fetch records")
        self.fetch_btn.clicked.connect(self.start_fetch)
        self.fetch_btn.setStyleSheet(
            f"QPushButton {{ background-color: {self.cobber_maroon.name()}; color: white; padding: 8px; font-weight: bold; border-radius: 5px; }}"
        )
        self.export_btn = QPushButton("Export dataset")
        self.export_btn.clicked.connect(self.export_current_result)
        self.clear_btn = QPushButton("Clear query")
        self.clear_btn.clicked.connect(self.clear_query)
        btn_row.addWidget(self.fetch_btn)
        btn_row.addWidget(self.export_btn)
        btn_row.addWidget(self.clear_btn)

        self.log_console = QTextBrowser()
        self.log_console.setOpenExternalLinks(True)

        left_layout.addWidget(form_box)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(QLabel("Fetch report"))
        left_layout.addWidget(self.log_console, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.results_tabs = QTabWidget()
        self.results_tabs.setTabsClosable(True)
        self.results_tabs.tabCloseRequested.connect(self.close_result_tab)
        right_layout.addWidget(self.results_tabs)
        self._add_start_tab()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 820])
        self.on_source_changed(self.source_combo.currentText())
        return page

    def _add_start_tab(self):
        empty = QTextBrowser()
        empty.setHtml(
            "<h2>Welcome to CobberEcoFetcher</h2>"
            "<p>Build a database query, fetch biodiversity records, inspect the results, and export a reproducible dataset.</p>"
            "<p><b>GBIF</b> returns biodiversity occurrence records. "
            "<b>iNaturalist</b> returns community observations, often with photographs.</p>"
            "<p>The preview shows a limited number of returned rows. The query summary reports the total number of matching records or observations found by the source.</p>"
        )
        self.results_tabs.addTab(empty, "Start here")

    def on_source_changed(self, source: str):
        is_gbif = source == "GBIF"
        self.gbif_coords.setEnabled(is_gbif)
        self.gbif_label.setEnabled(is_gbif)
        self.inat_photos.setEnabled(not is_gbif)
        self.inat_research.setEnabled(not is_gbif)
        self.inat_label.setEnabled(not is_gbif)
        self.limit_input.setValue(100 if is_gbif else 50)
        if is_gbif:
            self.start_date.setDate(date(2000, 1, 1))
        else:
            self.start_date.setDate(date(2020, 1, 1))

    def _collect_query(self) -> Tuple[str, Dict[str, Any]]:
        source = self.source_combo.currentText()
        query = {
            "scientific_question": self.scientific_question.text().strip(),
            "taxon": self.taxon_input.text().strip(),
            "place": self.place_input.text().strip(),
            "start": self.start_date.date().toString("yyyy-MM-dd"),
            "end": self.end_date.date().toString("yyyy-MM-dd"),
            "coords_only": self.gbif_coords.isChecked(),
            "photos_only": self.inat_photos.isChecked(),
            "research_grade": self.inat_research.isChecked(),
            "limit": self.limit_input.value(),
        }
        return source, query

    def start_fetch(self):
        source, query = self._collect_query()
        self.fetch_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.log_console.clear()
        self._write_fetch_start(source, query)
        worker = FetchWorker(source, query)
        worker.signals.progress.connect(self.log_progress)
        worker.signals.result.connect(self.add_result_tab)
        worker.signals.error.connect(self.log_error)
        worker.signals.finished.connect(self.on_fetch_finished)
        self.threadpool.start(worker)

    def _write_fetch_start(self, source: str, query: Dict[str, Any]):
        if source == "GBIF":
            lines = [
                "<b>Fetching records from GBIF...</b>",
                "<br><br><b>Sending query:</b>",
                f"<br>Taxon: {html_escape(query.get('taxon'))}",
                f"<br>Place filter: {html_escape(query.get('place') or 'none')}",
                f"<br>Date filter: {html_escape(query.get('start'))} to {html_escape(query.get('end'))}",
                f"<br>Coordinate filter: {'required' if query.get('coords_only') else 'not required'}",
                f"<br>Preview size: {html_escape(query.get('limit'))}",
            ]
        else:
            lines = [
                "<b>Fetching observations from iNaturalist...</b>",
                "<br><br><b>Sending query:</b>",
                f"<br>Taxon: {html_escape(query.get('taxon'))}",
                f"<br>Place filter: {html_escape(query.get('place') or 'none')}",
                f"<br>Date filter: {html_escape(query.get('start'))} to {html_escape(query.get('end'))}",
                f"<br>Photo filter: {'required' if query.get('photos_only') else 'not required'}",
                f"<br>Quality filter: {'research grade only' if query.get('research_grade') else 'all quality grades'}",
                f"<br>Preview size: {html_escape(query.get('limit'))}",
            ]
        self.log_console.setHtml("".join(lines))

    def on_fetch_finished(self):
        self.fetch_btn.setEnabled(True)
        self.export_btn.setEnabled(True)
        self.statusBar().showMessage("Fetch complete.", 5000)

    def log_progress(self, message: str):
        self.statusBar().showMessage(message)
        self.log_console.append(f"<br><span style='color:#555;'>{html_escape(message)}</span>")

    def log_error(self, message: str):
        self.statusBar().showMessage("Error.", 5000)
        self.log_console.append(f"<br><span style='color:red;'><b>Error:</b> {html_escape(message)}</span>")
        QMessageBox.warning(self, APP_TITLE, message)

    def log_success(self, result: ResultPackage):
        if result.source == "GBIF":
            text = (
                "<br><br><span style='color:green;'><b>Success.</b> "
                "GBIF returned occurrence records for your query.</span>"
                f"<br><br>Occurrence records shown in preview: {len(result.preview_rows)}"
                f"<br>Total matching occurrence records found: {html_escape(result.total_matching)}"
                "<br><br>Next step: Open Query summary to check the matched taxon, filters, and record counts."
            )
        else:
            text = (
                "<br><br><span style='color:green;'><b>Success.</b> "
                "iNaturalist returned observations for your query.</span>"
                f"<br><br>Observations shown in preview: {len(result.preview_rows)}"
                f"<br>Total matching observations found: {html_escape(result.total_matching)}"
                "<br><br>Next step: Open Query summary to check the matched taxon, filters, and observation counts."
            )
        self.log_console.append(text)

    def add_result_tab(self, result: ResultPackage):
        if self.results_tabs.count() == 1 and self.results_tabs.tabText(0) == "Start here":
            self.results_tabs.removeTab(0)
        self.results.append(result)

        container = QWidget()
        layout = QVBoxLayout(container)
        subtabs = QTabWidget()
        map_panel = LazyMapPanel(result.map_points)

        subtabs.addTab(HtmlPanel(self._query_summary_html(result)), "Query summary")
        subtabs.addTab(SimpleTablePanel(result.preview_rows), "Returned records")
        subtabs.addTab(map_panel, "Map of records")
        subtabs.addTab(MediaPanel(result.media_items), "Media preview")

        def maybe_build_map(index: int, tabs=subtabs, panel=map_panel):
            if tabs.tabText(index) == "Map of records":
                panel.ensure_built()

        subtabs.currentChanged.connect(maybe_build_map)

        export_row = QHBoxLayout()
        export_now = QPushButton("Export dataset")
        export_now.clicked.connect(lambda: self.export_result(result))
        export_row.addStretch(1)
        export_row.addWidget(export_now)

        layout.addWidget(subtabs)
        layout.addLayout(export_row)
        self.results_tabs.addTab(container, result.title)
        self.results_tabs.setCurrentWidget(container)
        self.log_success(result)

    def _query_summary_html(self, result: ResultPackage) -> str:
        q = result.query_summary
        details = result.summary_details
        if result.source == "GBIF":
            return f"""
            <h2>GBIF query summary</h2>
            <h3>Scientific question</h3>
            <p>{html_escape(result.scientific_question or 'No scientific question entered.')}</p>

            <h3>Filters used</h3>
            <ul>
              <li><b>Organism or taxon:</b> {html_escape(q.get('taxon'))}</li>
              <li><b>Place filter:</b> {html_escape(q.get('place') or 'none')}</li>
              <li><b>Date filter:</b> {html_escape(q.get('start'))} to {html_escape(q.get('end'))}</li>
              <li><b>Coordinate required:</b> {bool_word(bool(q.get('coords_only')))}</li>
              <li><b>Preview size:</b> {html_escape(q.get('limit'))}</li>
            </ul>

            <h3>Database match</h3>
            <ul>
              <li><b>Taxon matched by GBIF:</b> {html_escape(details.get('taxon_matched'))}</li>
              <li><b>GBIF taxon key:</b> {html_escape(details.get('taxon_key'))}</li>
            </ul>

            <h3>Records returned</h3>
            <ul>
              <li><b>Occurrence records shown in preview:</b> {html_escape(details.get('preview_count'))}</li>
              <li><b>Total matching occurrence records found:</b> {html_escape(details.get('total_matching'))}</li>
              <li><b>Common countries or regions in preview:</b> {html_escape(details.get('common_regions'))}</li>
            </ul>

            <p><b>Fetched at:</b> {html_escape(result.fetched_at)}</p>
            <p><i>No records does not necessarily mean the organism is absent. It may mean that no one recorded it there, the record was not shared with GBIF, or the filters were too narrow.</i></p>
            """
        return f"""
            <h2>iNaturalist query summary</h2>
            <h3>Scientific question</h3>
            <p>{html_escape(result.scientific_question or 'No scientific question entered.')}</p>

            <h3>Filters used</h3>
            <ul>
              <li><b>Organism or taxon:</b> {html_escape(q.get('taxon'))}</li>
              <li><b>Place filter:</b> {html_escape(q.get('place') or 'none')}</li>
              <li><b>Date filter:</b> {html_escape(q.get('start'))} to {html_escape(q.get('end'))}</li>
              <li><b>Photo filter:</b> {'required' if q.get('photos_only') else 'not required'}</li>
              <li><b>Quality filter:</b> {'research grade only' if q.get('research_grade') else 'all quality grades'}</li>
              <li><b>Preview size:</b> {html_escape(q.get('limit'))}</li>
            </ul>

            <h3>Database match</h3>
            <ul>
              <li><b>Taxon matched by iNaturalist:</b> {html_escape(details.get('taxon_matched'))}</li>
              <li><b>Place matched by iNaturalist:</b> {html_escape(details.get('place_matched'))}</li>
            </ul>

            <h3>Observations returned</h3>
            <ul>
              <li><b>Observations shown in preview:</b> {html_escape(details.get('preview_count'))}</li>
              <li><b>Total matching observations found:</b> {html_escape(details.get('total_matching'))}</li>
              <li><b>Observations with photos in preview:</b> {html_escape(details.get('photos_in_preview'))}</li>
              <li><b>Common places in preview:</b> {html_escape(details.get('common_places'))}</li>
            </ul>

            <p><b>Fetched at:</b> {html_escape(result.fetched_at)}</p>
            <p><i>iNaturalist observations reflect both organisms and observer activity. A map of observations is not the same thing as a complete map of where the organism lives.</i></p>
            """

    def close_result_tab(self, index: int):
        self.results_tabs.removeTab(index)
        if self.results_tabs.count() == 0:
            self._add_start_tab()

    def current_result(self) -> Optional[ResultPackage]:
        current_index = self.results_tabs.currentIndex()
        if current_index < 0:
            return None
        tab_title = self.results_tabs.tabText(current_index)
        for result in reversed(self.results):
            if result.title == tab_title:
                return result
        return None

    def export_current_result(self):
        result = self.current_result()
        if result is None:
            QMessageBox.information(self, APP_TITLE, "There is no result tab selected to export.")
            return
        self.export_result(result)

    def export_result(self, result: ResultPackage):
        target_dir = QFileDialog.getExistingDirectory(self, "Choose export folder")
        if not target_dir:
            return
        export_dir = Path(target_dir)
        base = result.export_name or "dataset"
        csv_path = export_dir / f"{base}.csv"
        json_path = export_dir / f"{base}_metadata.json"
        txt_path = export_dir / f"{base}_query_summary.txt"

        try:
            if result.preview_rows:
                columns = list(result.preview_rows[0].keys())
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(result.preview_rows)
            else:
                with csv_path.open("w", newline="", encoding="utf-8") as f:
                    f.write("No preview rows were returned.\n")

            with json_path.open("w", encoding="utf-8") as f:
                json.dump(result.raw_metadata, f, indent=2, ensure_ascii=False)

            with txt_path.open("w", encoding="utf-8") as f:
                f.write(self._plain_text_query_summary(result))

            self.log_console.append(f"<br><span style='color:green;'><b>Exported files to:</b> {html_escape(export_dir)}</span>")
            QMessageBox.information(
                self,
                APP_TITLE,
                f"Exported:\n{csv_path.name}\n{json_path.name}\n{txt_path.name}",
            )
        except Exception as exc:
            self.log_error(f"Export failed: {exc}")

    def _plain_text_query_summary(self, result: ResultPackage) -> str:
        q = result.query_summary
        details = result.summary_details
        if result.source == "GBIF":
            lines = [
                "GBIF query summary",
                "",
                f"Source: {result.source}",
                f"Dataset title: GBIF occurrence records for {details.get('taxon_matched', q.get('taxon'))}",
                f"Fetched at: {result.fetched_at}",
                "",
                "Scientific question",
                result.scientific_question or "No scientific question entered.",
                "",
                "Filters used",
                f"Organism or taxon: {q.get('taxon')}",
                f"Place filter: {q.get('place') or 'none'}",
                f"Date filter: {q.get('start')} to {q.get('end')}",
                f"Coordinate filter: {'required' if q.get('coords_only') else 'not required'}",
                f"Preview size: {q.get('limit')}",
                "",
                "Database match",
                f"Taxon matched by GBIF: {details.get('taxon_matched')}",
                f"GBIF taxon key: {details.get('taxon_key')}",
                "",
                "Records returned",
                f"Occurrence records shown in preview: {details.get('preview_count')}",
                f"Total matching occurrence records found: {details.get('total_matching')}",
                f"Common countries or regions in preview: {details.get('common_regions')}",
            ]
        else:
            lines = [
                "iNaturalist query summary",
                "",
                f"Source: {result.source}",
                f"Dataset title: iNaturalist observations for {details.get('taxon_matched', q.get('taxon'))}",
                f"Fetched at: {result.fetched_at}",
                "",
                "Scientific question",
                result.scientific_question or "No scientific question entered.",
                "",
                "Filters used",
                f"Organism or taxon: {q.get('taxon')}",
                f"Place filter: {q.get('place') or 'none'}",
                f"Date filter: {q.get('start')} to {q.get('end')}",
                f"Photo filter: {'required' if q.get('photos_only') else 'not required'}",
                f"Quality filter: {'research grade only' if q.get('research_grade') else 'all quality grades'}",
                f"Preview size: {q.get('limit')}",
                "",
                "Database match",
                f"Taxon matched by iNaturalist: {details.get('taxon_matched')}",
                f"Place matched by iNaturalist: {details.get('place_matched')}",
                "",
                "Observations returned",
                f"Observations shown in preview: {details.get('preview_count')}",
                f"Total matching observations found: {details.get('total_matching')}",
                f"Observations with photos in preview: {details.get('photos_in_preview')}",
                f"Common places in preview: {details.get('common_places')}",
            ]

        lines.extend(
            [
                "",
                "Exported files",
                "CSV file: returned records or observations shown in the preview table",
                "JSON file: source metadata and raw database response",
                "TXT file: query summary and export notes",
                "",
                "Notes",
                f"This export was created by CobberEcoFetcher {APP_VERSION}.",
                "Review source licensing and citation requirements before publication.",
            ]
        )
        return "\n".join(str(line) for line in lines) + "\n"

    def clear_query(self):
        self.scientific_question.clear()
        self.taxon_input.clear()
        self.place_input.clear()
        self.log_console.clear()
        self.results_tabs.clear()
        self._add_start_tab()
        self.statusBar().showMessage("Ready. Query cleared.")

    def load_sample_query(self):
        dialog = SampleQueryDialog(self)
        payload = dialog.get_payload()
        if not payload:
            return
        self.source_combo.setCurrentText(payload.get("source", "GBIF"))
        self.scientific_question.setText(payload.get("scientific_question", ""))
        self.taxon_input.setText(payload.get("taxon", ""))
        self.place_input.setText(payload.get("place", ""))
        self.gbif_coords.setChecked(bool(payload.get("coords_only", True)))
        self.inat_photos.setChecked(bool(payload.get("photos_only", True)))
        self.inat_research.setChecked(bool(payload.get("research_grade", True)))
        self.limit_input.setValue(int(payload.get("limit", 100 if payload.get("source") == "GBIF" else 50)))
        try:
            self.start_date.setDate(datetime.strptime(payload.get("start", "2000-01-01"), "%Y-%m-%d").date())
            self.end_date.setDate(datetime.strptime(payload.get("end", datetime.now().strftime("%Y-%m-%d")), "%Y-%m-%d").date())
        except Exception:
            pass
        self.log_console.setHtml(f"<span style='color:green;'><b>Loaded sample query for {html_escape(payload.get('source', 'GBIF'))}.</b></span>")


def main():
    app = QApplication(sys.argv)
    window = CobberEcoFetcherApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
