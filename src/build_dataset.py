#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

# -------------------------
# Status vocabulary
# -------------------------
IDENTITY_OK = "ok"
IDENTITY_AMBIGUOUS = "ambiguous"
IDENTITY_NOT_FOUND = "not_found"

PLACE_OK = "ok"
PLACE_NO_PLACE = "no_place"
PLACE_GEOCODE_FAILED = "geocode_failed"

# -------------------------
# Place candidate model
# -------------------------
@dataclass
class PlaceCandidate:
    place_label: str
    place_type: str          # base/residence/work_location/formation/hq/birthplace/unknown
    confidence: str          # high/medium/low
    source_kind: str         # bandcamp/wikidata/nominatim/...
    evidence_url: str
    evidence_note: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None


# -------------------------
# SQLite cache (HTTP + geocode)
# -------------------------
def db_connect(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS http_cache (
        url TEXT PRIMARY KEY,
        status INTEGER,
        fetched_at TEXT,
        body BLOB
    )
    """)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS geocode_cache (
        query TEXT PRIMARY KEY,
        lat REAL,
        lon REAL,
        fetched_at TEXT
    )
    """)
    conn.commit()
    return conn

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def http_get_cached(
    conn: sqlite3.Connection,
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    min_delay_s: float,
    ttl_days: int = 30,
    last_request_time: Optional[List[float]] = None,
) -> Tuple[int, str]:
    """
    Returns (status_code, text_body). Uses a simple SQLite cache with TTL.
    """
    cur = conn.execute("SELECT status, fetched_at, body FROM http_cache WHERE url = ?", (url,))
    row = cur.fetchone()
    if row:
        status, fetched_at, body = row
        try:
            fetched_dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - fetched_dt).total_seconds() / 86400.0
        except Exception:
            age_days = 1e9
        if age_days <= ttl_days:
            return int(status), (body.decode("utf-8", errors="replace") if isinstance(body, (bytes, bytearray)) else str(body))

    # throttle
    if last_request_time is not None:
        dt = time.time() - last_request_time[0]
        if dt < min_delay_s:
            time.sleep(min_delay_s - dt)

    resp = session.get(url, headers=headers, timeout=30)
    if last_request_time is not None:
        last_request_time[0] = time.time()

    text = resp.text
    conn.execute(
        "INSERT OR REPLACE INTO http_cache(url, status, fetched_at, body) VALUES (?, ?, ?, ?)",
        (url, int(resp.status_code), utc_now_iso(), text.encode("utf-8")),
    )
    conn.commit()
    return int(resp.status_code), text

def geocode_nominatim(
    conn: sqlite3.Connection,
    session: requests.Session,
    query: str,
    headers: Dict[str, str],
    min_delay_s: float,
    last_request_time: List[float],
) -> Optional[Tuple[float, float]]:
    query = query.strip()
    if not query:
        return None

    cur = conn.execute("SELECT lat, lon FROM geocode_cache WHERE query = ?", (query,))
    row = cur.fetchone()
    if row and row[0] is not None and row[1] is not None:
        return float(row[0]), float(row[1])

    dt = time.time() - last_request_time[0]
    if dt < min_delay_s:
        time.sleep(min_delay_s - dt)

    url = "https://nominatim.openstreetmap.org/search"
    params = {"format": "jsonv2", "q": query, "limit": 1}
    resp = session.get(url, params=params, headers=headers, timeout=30)
    last_request_time[0] = time.time()

    if resp.status_code != 200:
        conn.execute(
            "INSERT OR REPLACE INTO geocode_cache(query, lat, lon, fetched_at) VALUES (?, ?, ?, ?)",
            (query, None, None, utc_now_iso()),
        )
        conn.commit()
        return None

    data = resp.json()
    if not data:
        conn.execute(
            "INSERT OR REPLACE INTO geocode_cache(query, lat, lon, fetched_at) VALUES (?, ?, ?, ?)",
            (query, None, None, utc_now_iso()),
        )
        conn.commit()
        return None

    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(query, lat, lon, fetched_at) VALUES (?, ?, ?, ?)",
        (query, lat, lon, utc_now_iso()),
    )
    conn.commit()
    return lat, lon


# -------------------------
# Bandcamp extraction
# -------------------------
def extract_bandcamp_location(html: str, artist_input: str) -> Optional[str]:
    """
    Tries multiple strategies to extract 'City, Country' from a Bandcamp artist page.
    Returns a location string or None.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Common header block: <p id="band-name-location"> ... <span class="location">Lyon, France</span>
    p = soup.find(id="band-name-location")
    if p:
        loc = p.get_text(" ", strip=True)
        # Often "ArtistName Lyon, France" -> remove artist name prefix if present
        if artist_input:
            loc = re.sub(r"^\s*" + re.escape(artist_input) + r"\s+", "", loc, flags=re.IGNORECASE)
        # Try to keep only the last "City, Country" segment
        m = re.search(r"([A-Za-zÀ-ÿ0-9 .'\-]+,\s*[A-Za-zÀ-ÿ .'\-]+)\s*$", loc)
        if m:
            return m.group(1).strip()
        if loc:
            return loc.strip()

    # Fallback: look for any span.location
    sp = soup.find("span", class_=re.compile(r"\blocation\b"))
    if sp:
        loc = sp.get_text(" ", strip=True)
        if loc:
            return loc.strip()

    # Fallback: regex in raw HTML for "location":"Lyon, France"
    m = re.search(r'"location"\s*:\s*"([^"]{2,80})"', html)
    if m:
        loc = m.group(1).strip()
        if loc:
            return loc

    return None


# -------------------------
# MusicBrainz identity pivot
# -------------------------
def mb_request(
    session: requests.Session,
    url: str,
    headers: Dict[str, str],
    min_delay_s: float,
    last_request_time: List[float],
    params: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    dt = time.time() - last_request_time[0]
    if dt < min_delay_s:
        time.sleep(min_delay_s - dt)
    resp = session.get(url, params=params, headers=headers, timeout=30)
    last_request_time[0] = time.time()
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None

def mb_search_artist(
    session: requests.Session,
    name: str,
    headers: Dict[str, str],
    min_delay_s: float,
    last_request_time: List[float],
    limit: int = 3,

) -> List[Dict[str, Any]]:
    url = "https://musicbrainz.org/ws/2/artist/"
    q = f'artist:"{name}"'
    data = mb_request(session, url, headers, min_delay_s, last_request_time, params={"query": q, "fmt": "json", "limit": str(limit)})
    if not data:
        return []
    return data.get("artists", []) or []

def mb_get_artist(
    session: requests.Session,
    mbid: str,
    headers: Dict[str, str],
    min_delay_s: float,
    last_request_time: List[float],
) -> Optional[Dict[str, Any]]:
    url = f"https://musicbrainz.org/ws/2/artist/{mbid}"
    return mb_request(session, url, headers, min_delay_s, last_request_time, params={"fmt": "json", "inc": "url-rels+aliases"})

def mb_extract_urls(mb_artist: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    rels = mb_artist.get("relations", []) or []
    for r in rels:
        if r.get("target-type") != "url":
            continue
        u = (r.get("url") or {}).get("resource") or ""
        t = (r.get("type") or "").lower()
        if not u:
            continue
        if "wikidata.org" in u:
            out.setdefault("wikidata_url", u)
        if "bandcamp.com" in u:
            out.setdefault("bandcamp_url", u)
        if "soundcloud.com" in u:
            out.setdefault("soundcloud_url", u)
        if t in ("official homepage", "homepage") or ("http" in u and "bandcamp.com" not in u and "wikidata.org" not in u and "soundcloud.com" not in u):
            out.setdefault("official_url", u)
    return out

def qid_from_wikidata_url(url: str) -> Optional[str]:
    m = re.search(r"/wiki/(Q\d+)", url)
    return m.group(1) if m else None


# -------------------------
# Wikidata places (EntityData JSON)
# -------------------------
def wd_entity_json(
    session: requests.Session,
    qid: str,
    headers: Dict[str, str],
    min_delay_s: float,
    last_request_time: List[float],
) -> Optional[Dict[str, Any]]:
    dt = time.time() - last_request_time[0]
    if dt < min_delay_s:
        time.sleep(min_delay_s - dt)
    url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
    resp = session.get(url, headers=headers, timeout=30)
    last_request_time[0] = time.time()
    if resp.status_code != 200:
        return None
    try:
        return resp.json()
    except Exception:
        return None

def wd_label_and_coords(entity_data: Dict[str, Any], qid: str) -> Tuple[Optional[str], Optional[Tuple[float, float]]]:
    ent = (entity_data.get("entities") or {}).get(qid) or {}
    labels = ent.get("labels") or {}
    label = None
    if "en" in labels:
        label = labels["en"].get("value")
    elif labels:
        label = next(iter(labels.values())).get("value")

    claims = ent.get("claims") or {}
    coords = None
    if "P625" in claims:
        snaks = claims["P625"]
        if snaks:
            dv = (((snaks[0].get("mainsnak") or {}).get("datavalue") or {}).get("value") or {})
            if isinstance(dv, dict) and "latitude" in dv and "longitude" in dv:
                coords = (float(dv["latitude"]), float(dv["longitude"]))
    return label, coords

def wd_extract_place_qids(entity_data: Dict[str, Any], subject_qid: str) -> List[Tuple[str, str]]:
    """
    Returns list of (place_qid, place_type) in priority order.
    """
    ent = (entity_data.get("entities") or {}).get(subject_qid) or {}
    claims = ent.get("claims") or {}

    # priority order (your hierarchy for structured sources)
    props = [
        ("P551", "residence"),
        ("P937", "work_location"),
        ("P740", "formation"),
        ("P159", "hq"),
        ("P19", "birthplace"),
    ]

    out: List[Tuple[str, str]] = []
    for pid, ptype in props:
        for c in claims.get(pid, []) or []:
            ms = c.get("mainsnak") or {}
            dv = (ms.get("datavalue") or {}).get("value") or {}
            if isinstance(dv, dict) and "id" in dv and str(dv["id"]).startswith("Q"):
                out.append((str(dv["id"]), ptype))
    return out


# -------------------------
# Main pipeline
# -------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="artists.csv")
    ap.add_argument("--out", default="public/data")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--user-agent", default="artist-map/0.2 (contact: unknown)")
    ap.add_argument("--mb-delay", type=float, default=1.05)
    ap.add_argument("--nominatim-delay", type=float, default=1.05)
    ap.add_argument("--wd-delay", type=float, default=0.3)
    ap.add_argument("--cache-db", default="src/geocode_cache.sqlite")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()

def main() -> int:
    args = parse_args()

    headers = {
        "User-Agent": args.user_agent,
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
    }

    conn = db_connect(args.cache_db)
    session = requests.Session()

    mb_last = [0.0]
    nom_last = [0.0]
    wd_last = [0.0]
    http_last = [0.0]

    # read input
    with open(args.input, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if args.max and args.max > 0:
        rows = rows[: args.max]

    ensure_dir(args.out)

    features: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []

    for row in rows:
        artist_input = (row.get("artist_input") or "").strip()
        qid = (row.get("qid") or "").strip()
        mbid = (row.get("mbid") or "").strip()

        # optional pivots
        bandcamp_url = (row.get("bandcamp_url") or "").strip()
        official_url = (row.get("official_url") or "").strip()
        soundcloud_url = (row.get("soundcloud_url") or "").strip()
 
         # evidence-driven base (manual but proof-backed)
         base_place = (row.get("base_place") or "").strip()
         base_source_url = (row.get("base_source_url") or "").strip()


        identity_status = IDENTITY_NOT_FOUND
        identity_reason = ""
        identity_confidence = "low"
        chosen_qid = qid
        chosen_mbid = mbid

        evidence_urls: List[str] = []

        # ---- Identity resolution
        mb_artist = None
        candidates_brief = ""

        if chosen_qid:
            identity_status = IDENTITY_OK
            identity_confidence = "high"
            identity_reason = "qid provided"
        else:
            if chosen_mbid:
                mb_artist = mb_get_artist(session, chosen_mbid, headers, args.mb_delay, mb_last)
                if mb_artist:
                    identity_status = IDENTITY_OK
                    identity_confidence = "high"
                    identity_reason = "mbid provided"
                else:
                    identity_status = IDENTITY_NOT_FOUND
                    identity_reason = "mbid provided but MusicBrainz fetch failed"
            else:
                # name search on MusicBrainz (may be ambiguous)
                if artist_input:
                    cands = mb_search_artist(session, artist_input, headers, args.mb_delay, mb_last, limit=3)
                    if cands:
                        # check best candidate
                        cands_sorted = sorted(cands, key=lambda x: int(x.get("score", 0)), reverse=True)
                        best = cands_sorted[0]
                        best_score = int(best.get("score", 0))
                        second_score = int(cands_sorted[1].get("score", 0)) if len(cands_sorted) > 1 else -1
                        best_name = (best.get("name") or "").strip()
                        best_id = (best.get("id") or "").strip()
                        candidates_brief = "; ".join([f'{(c.get("name") or "").strip()}|{c.get("id")}|{c.get("score")}' for c in cands_sorted])

                        # strict auto-pick only if strong
                        if best_id and best_score >= 95 and (best_score - second_score) >= 5 and best_name.lower() == artist_input.lower():
                            chosen_mbid = best_id
                            mb_artist = mb_get_artist(session, chosen_mbid, headers, args.mb_delay, mb_last)
                            if mb_artist:
                                identity_status = IDENTITY_OK
                                identity_confidence = "medium"
                                identity_reason = "MusicBrainz name search strong match"
                            else:
                                identity_status = IDENTITY_NOT_FOUND
                                identity_reason = "MusicBrainz candidate fetch failed"
                        else:
                            identity_status = IDENTITY_AMBIGUOUS if len(cands_sorted) > 1 else IDENTITY_NOT_FOUND
                            identity_reason = "MusicBrainz name search ambiguous; provide mbid/qid or pivot URL"
                    else:
                        identity_status = IDENTITY_NOT_FOUND
                        identity_reason = "MusicBrainz name search returned nothing"
                else:
                    identity_status = IDENTITY_NOT_FOUND
                    identity_reason = "empty artist_input"

        # If we have MB artist data, extract pivots (bandcamp/wikidata/etc.)
        if mb_artist:
            url_pivots = mb_extract_urls(mb_artist)
            if not bandcamp_url:
                bandcamp_url = url_pivots.get("bandcamp_url", "") or bandcamp_url
            if not official_url:
                official_url = url_pivots.get("official_url", "") or official_url
            if not soundcloud_url:
                soundcloud_url = url_pivots.get("soundcloud_url", "") or soundcloud_url
            if not chosen_qid:
                wdu = url_pivots.get("wikidata_url", "")
                q = qid_from_wikidata_url(wdu) if wdu else None
                if q:
                    chosen_qid = q

        # Pivot URLs (if present) are identity locks
        if bandcamp_url:
            evidence_urls.append(bandcamp_url)
        if official_url:
            evidence_urls.append(official_url)
        if soundcloud_url:
            evidence_urls.append(soundcloud_url)

        if identity_status != IDENTITY_OK and (bandcamp_url or official_url or soundcloud_url):
            # A provided pivot URL is treated as a practical identity lock
            identity_status = IDENTITY_OK
            identity_confidence = "medium"
            identity_reason = "pivot URL provided (bandcamp/official/soundcloud)"

        # ---- Place extraction (hierarchy: Wikidata then Bandcamp as per your strict execution, but for niche we prioritize Bandcamp if provided)
        place_candidates: List[PlaceCandidate] = []

 
         # 0) Evidence-driven base: explicit place string + proof URL (no inference)
         if base_place:
             place_candidates.append(
                 PlaceCandidate(
                     place_label=base_place,
                     place_type="base",
                     confidence="medium",
                     source_kind="evidence",
                     evidence_url=(base_source_url or bandcamp_url or official_url or ""),
                     evidence_note="base_place provided (proof-backed manual entry)",
                 )
             )


        # 1) Wikidata (structured) if QID available
        if chosen_qid and identity_status == IDENTITY_OK:
            wd = wd_entity_json(session, chosen_qid, headers, args.wd_delay, wd_last)
            if wd:
                for pqid, ptype in wd_extract_place_qids(wd, chosen_qid):
                    wd_place = wd_entity_json(session, pqid, headers, args.wd_delay, wd_last)
                    if not wd_place:
                        continue
                    label, coords = wd_label_and_coords(wd_place, pqid)
                    if not label:
                        continue
                    latlon = coords
                    place_candidates.append(
                        PlaceCandidate(
                            place_label=label,
                            place_type=ptype,
                            confidence="medium",
                            source_kind="wikidata",
                            evidence_url=f"https://www.wikidata.org/wiki/{chosen_qid}",
                            evidence_note=f"{ptype} -> {pqid}",
                            lat=(latlon[0] if latlon else None),
                            lon=(latlon[1] if latlon else None),
                        )
                    )

        # 2) Bandcamp profile (niche-coverage, explicit city/country shown on profile)
        if bandcamp_url and identity_status == IDENTITY_OK:
            status, html = http_get_cached(conn, session, bandcamp_url, headers, min_delay_s=0.5, ttl_days=14, last_request_time=http_last)
            if status == 200 and html:
                loc = extract_bandcamp_location(html, artist_input)
                if loc:
                    place_candidates.append(
                        PlaceCandidate(
                            place_label=loc,
                            place_type="base",
                            confidence="medium",
                            source_kind="bandcamp",
                            evidence_url=bandcamp_url,
                            evidence_note="profile location",
                        )
                    )

        # pick best candidate:
        # - Prefer Bandcamp base (explicit profile location) when present
        # - Else prefer Wikidata in priority order already appended (residence/work/formation/hq/birthplace)
        chosen: Optional[PlaceCandidate] = None
        for c in place_candidates:
            if c.source_kind == "bandcamp" and c.place_type == "base":
                chosen = c
                break
        if chosen is None and place_candidates:
            chosen = place_candidates[0]

        place_status = PLACE_NO_PLACE
        place_reason = ""
        lat = lon = None

        if not chosen:
            place_status = PLACE_NO_PLACE
            place_reason = "No place candidate (need qid/mbid or pivot URL like bandcamp_url)."
        else:
            # ensure coords
            if chosen.lat is not None and chosen.lon is not None:
                lat, lon = chosen.lat, chosen.lon
                place_status = PLACE_OK
            else:
                latlon = geocode_nominatim(conn, session, chosen.place_label, headers, args.nominatim_delay, nom_last)
                if latlon:
                    lat, lon = latlon
                    place_status = PLACE_OK
                else:
                    place_status = PLACE_GEOCODE_FAILED
                    place_reason = f"Geocoding failed for {chosen.place_label!r}"

        # build feature if ok
        if place_status == PLACE_OK and lat is not None and lon is not None:
            feat = {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "artist_input": artist_input,
                    "qid": chosen_qid,
                    "mbid": chosen_mbid,
                    "place_label": chosen.place_label if chosen else "",
                    "place_type": chosen.place_type if chosen else "unknown",
                    "confidence": chosen.confidence if chosen else "low",
                    "source_kind": chosen.source_kind if chosen else "",
                    "evidence_url": chosen.evidence_url if chosen else "",
                    "identity_confidence": identity_confidence,
                    "identity_reason": identity_reason,
                },
            }
            features.append(feat)

        audit_rows.append({
            "artist_input": artist_input,
            "qid": chosen_qid,
            "mbid": chosen_mbid,
            "bandcamp_url": bandcamp_url,
            "official_url": official_url,
            "soundcloud_url": soundcloud_url,
            "base_place": base_place,
            "base_source_url": base_source_url,
            "identity_status": identity_status,
            "identity_confidence": identity_confidence,
            "identity_reason": identity_reason,
            "mb_candidates": candidates_brief,
            "place_status": place_status,
            "place_reason": place_reason,
            "chosen_place_label": chosen.place_label if chosen else "",
            "chosen_place_type": chosen.place_type if chosen else "",
            "chosen_source_kind": chosen.source_kind if chosen else "",
            "chosen_evidence_url": chosen.evidence_url if chosen else "",
            "lat": lat if lat is not None else "",
            "lon": lon if lon is not None else "",
        })

        if args.verbose:
            print(f"[{identity_status}/{place_status}] {artist_input} | qid={chosen_qid or '-'} mbid={chosen_mbid or '-'} | place={chosen.place_label if chosen else '-'} | {place_reason or identity_reason}")

    # write outputs
    geo = {"type": "FeatureCollection", "features": features}
    out_geo = os.path.join(args.out, "artists.geojson")
    out_csv = os.path.join(args.out, "artists.csv")

    with open(out_geo, "w", encoding="utf-8") as f:
        json.dump(geo, f, ensure_ascii=False, indent=2)

    # audit csv
    fieldnames = list(audit_rows[0].keys()) if audit_rows else []
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in audit_rows:
            w.writerow(r)

    counts_id = Counter(r["identity_status"] for r in audit_rows)
    counts_pl = Counter(r["place_status"] for r in audit_rows)
    print("[OK] Wrote:", out_geo)
    print("[OK] Wrote:", out_csv)
    print("[SUMMARY] identity_status:", dict(counts_id))
    print("[SUMMARY] place_status:", dict(counts_pl))
    print("[SUMMARY] features:", len(features))
    return 0

