#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
MUSICBRAINZ_WS = "https://musicbrainz.org/ws/2"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

DEFAULT_UA = "artist-map/0.1 (https://github.com/zb4xh4p8pk-droid/artist-map)"
REQ_SLEEP_S = 1.05

WKT_POINT_RE = re.compile(r"Point\(([-0-9.]+)\s+([-0-9.]+)\)")

PLACE_PRIORITY = [
    ("residence", "P551"),   # place of residence
    ("base",      "P159"),   # headquarters location
    ("base",      "P276"),   # location
    ("origin",    "P740"),   # location of formation
    ("birthplace","P19"),    # place of birth
]

ACCEPTABLE_PLACE_INSTANCES = {
    "Q515",       # city
    "Q486972",    # human settlement
    "Q3957",      # town
    "Q532",       # village
    "Q6256",      # country
    "Q3624078",   # sovereign state
    "Q5119",      # capital
}

@dataclass
class ArtistRow:
    artist_input: str
    mbid: str
    qid: str


def http_get_json(url: str, params: Dict[str, Any], ua: str) -> Dict[str, Any]:
    r = requests.get(url, params=params, headers={"User-Agent": ua}, timeout=30)
    r.raise_for_status()
    return r.json()


def sparql_json(query: str, ua: str) -> Dict[str, Any]:
    r = requests.get(
        WIKIDATA_SPARQL,
        params={"format": "json", "query": query},
        headers={"User-Agent": ua, "Accept": "application/sparql-results+json"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def parse_wkt_point(wkt: str) -> Optional[Tuple[float, float]]:
    m = WKT_POINT_RE.search(wkt)
    if not m:
        return None
    lon = float(m.group(1))
    lat = float(m.group(2))
    return lat, lon


def init_cache(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS geocode_cache (
            query TEXT PRIMARY KEY,
            lat REAL,
            lon REAL,
            display_name TEXT,
            updated_at TEXT
        )
        """
    )
    conn.commit()
    return conn


def cache_get(conn: sqlite3.Connection, query: str) -> Optional[Tuple[float, float, str]]:
    cur = conn.execute("SELECT lat, lon, display_name FROM geocode_cache WHERE query = ?", (query,))
    row = cur.fetchone()
    if not row:
        return None
    return float(row[0]), float(row[1]), str(row[2])


def cache_put(conn: sqlite3.Connection, query: str, lat: float, lon: float, display_name: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO geocode_cache(query, lat, lon, display_name, updated_at) VALUES(?,?,?,?,?)",
        (query, lat, lon, display_name, dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")),
    )
    conn.commit()


def nominatim_geocode(query: str, ua: str, conn: sqlite3.Connection) -> Optional[Tuple[float, float, str, str]]:
    cached = cache_get(conn, query)
    if cached:
        lat, lon, display_name = cached
        return lat, lon, display_name, "cache"

    params = {"q": query, "format": "json", "limit": 1}
    r = requests.get(NOMINATIM_SEARCH, params=params, headers={"User-Agent": ua}, timeout=30)
    r.raise_for_status()
    hits = r.json()
    time.sleep(REQ_SLEEP_S)

    if not hits:
        return None
    hit = hits[0]
    lat = float(hit["lat"])
    lon = float(hit["lon"])
    display_name = hit.get("display_name", query)
    cache_put(conn, query, lat, lon, display_name)
    return lat, lon, display_name, "nominatim"


def mb_artist_urls(mbid: str, ua: str) -> List[str]:
    url = f"{MUSICBRAINZ_WS}/artist/{mbid}"
    params = {"fmt": "json", "inc": "url-rels"}
    data = http_get_json(url, params=params, ua=ua)
    rels = data.get("relations") or []
    out: List[str] = []
    for rel in rels:
        u = (rel.get("url") or {}).get("resource")
        if u:
            out.append(u)
    return out


def extract_wikidata_qid(urls: List[str]) -> str:
    for u in urls:
        m = re.search(r"wikidata\.org/wiki/(Q\d+)", u)
        if m:
            return m.group(1)
    return ""


def wikidata_search_qid(label: str, ua: str, lang: str = "fr") -> str:
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": lang,
        "format": "json",
        "limit": 5,
        "type": "item",
    }
    data = http_get_json(WIKIDATA_API, params=params, ua=ua)
    hits = data.get("search") or []
    if not hits:
        return ""
    return hits[0].get("id") or ""


def qid_url(qid: str) -> str:
    return f"https://www.wikidata.org/wiki/{qid}"


def wd_get_best_place(qid: str, ua: str) -> Tuple[str, str, str]:
    for place_type, prop in PLACE_PRIORITY:
        query = f"""
        SELECT ?place ?placeLabel WHERE {{
          wd:{qid} wdt:{prop} ?place .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "fr,en". }}
        }} LIMIT 1
        """
        data = sparql_json(query, ua=ua)
        time.sleep(REQ_SLEEP_S)
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            place_uri = bindings[0]["place"]["value"]
            place_qid = place_uri.rsplit("/", 1)[-1]
            place_label = bindings[0]["placeLabel"]["value"]
            return place_qid, place_label, place_type
    return "", "", "unknown"


def wd_place_details(place_qid: str, ua: str) -> Tuple[Optional[Tuple[float, float]], List[str], str]:
    query = f"""
    SELECT ?coord ?inst ?label WHERE {{
      OPTIONAL {{ wd:{place_qid} wdt:P625 ?coord. }}
      OPTIONAL {{ wd:{place_qid} wdt:P31 ?inst. }}
      SERVICE wikibase:label {{
        bd:serviceParam wikibase:language "fr,en".
        wd:{place_qid} rdfs:label ?label.
      }}
    }}
    """
    data = sparql_json(query, ua=ua)
    time.sleep(REQ_SLEEP_S)
    bindings = data.get("results", {}).get("bindings", [])

    coords = None
    insts: List[str] = []
    label = place_qid

    for b in bindings:
        if "coord" in b and coords is None:
            coords = parse_wkt_point(b["coord"]["value"])
        if "inst" in b:
            inst_uri = b["inst"]["value"]
            insts.append(inst_uri.rsplit("/", 1)[-1])
        if "label" in b:
            label = b["label"]["value"]

    return coords, sorted(set(insts)), label


def wd_admin_fallback(place_qid: str, ua: str, max_hops: int = 4) -> str:
    current = place_qid
    for _ in range(max_hops):
        coords, insts, _ = wd_place_details(current, ua=ua)
        if coords is not None or any(i in ACCEPTABLE_PLACE_INSTANCES for i in insts):
            return current

        query = f"SELECT ?admin WHERE {{ wd:{current} wdt:P131 ?admin . }} LIMIT 1"
        data = sparql_json(query, ua=ua)
        time.sleep(REQ_SLEEP_S)
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            return current
        admin_uri = bindings[0]["admin"]["value"]
        current = admin_uri.rsplit("/", 1)[-1]
    return current


def read_input_csv(path: Path) -> List[ArtistRow]:
    rows: List[ArtistRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for line in reader:
            artist_input = (line.get("artist_input") or "").strip()
            if not artist_input:
                continue
            mbid = (line.get("mbid") or "").strip()
            qid = (line.get("qid") or "").strip()
            rows.append(ArtistRow(artist_input=artist_input, mbid=mbid, qid=qid))
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="", help="CSV input (défaut: artists.csv puis artistes.csv)")
    ap.add_argument("--outdir", default="public/data", help="dossier de sortie")
    ap.add_argument("--ua", default=DEFAULT_UA, help="User-Agent HTTP")
    ap.add_argument("--max", type=int, default=0, help="limiter à N artistes (debug)")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    if args.input:
        input_csv = (repo_root / args.input).resolve()
    else:
        c1 = repo_root / "artists.csv"
        c2 = repo_root / "artistes.csv"
        input_csv = c1 if c1.exists() else c2

    outdir = (repo_root / args.outdir).resolve()
    ensure_dir(outdir)

    cache_path = (repo_root / "src" / "geocode_cache.sqlite").resolve()
    conn = init_cache(cache_path)

    rows = read_input_csv(input_csv)
    if args.max and args.max > 0:
        rows = rows[: args.max]

    checked_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    audit_rows: List[Dict[str, Any]] = []
    features: List[Dict[str, Any]] = []

    for row in rows:
        sources: List[str] = []
        confidence = "faible"

        qid = row.qid.strip()

        if not qid and row.mbid:
            try:
                urls = mb_artist_urls(row.mbid, ua=args.ua)
                sources.append(f"https://musicbrainz.org/artist/{row.mbid}")
                qid = extract_wikidata_qid(urls)
                time.sleep(REQ_SLEEP_S)
                if qid:
                    confidence = "moyen"
            except Exception:
                pass

        if not qid:
            try:
                qid = wikidata_search_qid(row.artist_input, ua=args.ua, lang="fr")
                time.sleep(REQ_SLEEP_S)
                if qid:
                    confidence = "faible"
            except Exception:
                qid = ""

        if qid:
            sources.append(qid_url(qid))
            if row.qid:
                confidence = "élevé"

        place_qid, place_label, place_type = ("", "", "unknown")
        if qid:
            try:
                place_qid, place_label, place_type = wd_get_best_place(qid, ua=args.ua)
            except Exception:
                place_qid, place_label, place_type = ("", "", "unknown")

        lat = lon = None
        place_source = ""
        final_place_qid = ""

        if place_qid:
            try:
                final_place_qid = wd_admin_fallback(place_qid, ua=args.ua)
                coords, _, final_label = wd_place_details(final_place_qid, ua=args.ua)
                place_label = final_label or place_label
                if coords:
                    lat, lon = coords
                    place_source = "wikidata"
            except Exception:
                pass

        if (lat is None or lon is None) and place_label:
            try:
                geo = nominatim_geocode(place_label, ua=args.ua, conn=conn)
                if geo:
                    lat, lon, disp, src = geo
                    place_source = src
                    place_label = disp
                    sources.append(
                        f"{NOMINATIM_SEARCH}?q={requests.utils.quote(place_label)}&format=json&limit=1"
                    )
            except Exception:
                pass

        if lat is None or lon is None:
            place_type = "unknown"
            confidence = "faible"

        props = {
            "artist_input": row.artist_input,
            "qid": qid,
            "mbid": row.mbid,
            "place_label": place_label,
            "place_qid": final_place_qid,
            "place_type": place_type,
            "confidence": confidence,
            "place_source": place_source,
            "sources": sources,
            "checked_at": checked_at,
        }
        audit_rows.append(props)

        if lat is not None and lon is not None:
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": props,
                }
            )

    geojson = {"type": "FeatureCollection", "features": features}
    (outdir / "artists.geojson").write_text(
        json.dumps(geojson, ensure_ascii=False), encoding="utf-8"
    )

    audit_path = outdir / "artists.csv"
    fieldnames = [
        "artist_input", "qid", "mbid",
        "place_label", "place_qid", "place_type",
        "confidence", "place_source", "checked_at", "sources",
    ]
    with audit_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in audit_rows:
            rr = dict(r)
            rr["sources"] = " | ".join(r.get("sources", []))
            w.writerow({k: rr.get(k, "") for k in fieldnames})

    print(f"[OK] Wrote: {outdir / 'artists.geojson'}")
    print(f"[OK] Wrote: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
