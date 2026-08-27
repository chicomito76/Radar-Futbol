from __future__ import annotations

import asyncio
import base64
import html
import json
import math
import os
import re
import socket
import unicodedata
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from dateutil import parser as dateparser
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

APP_NAME = "Radar Fútbol"
APP_VERSION = "4.7"
TZ = ZoneInfo("America/Santiago")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT, "frontend")

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Caché en memoria. Evita descargar la misma página varias veces.
CACHE: dict[str, tuple[float, Any]] = {}
HTML_TTL = 15 * 60
SEARCH_TTL = 30 * 60
ANALYSIS_TTL = 12 * 60
CONTEXT_TTL = 20 * 60

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151 Safari/537.36 "
    "RadarFutbol/4.7"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "es-CL,es;q=0.9,en;q=0.7",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def now_chile() -> datetime:
    return datetime.now(TZ)


def cache_get(key: str, ttl: int):
    item = CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > ttl:
        CACHE.pop(key, None)
        return None
    return value


def cache_set(key: str, value: Any):
    CACHE[key] = (time.time(), value)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize3(a: float, b: float, c: float) -> tuple[float, float, float]:
    total = a + b + c
    if total <= 0:
        return 33.34, 33.33, 33.33
    return a / total * 100, b / total * 100, c / total * 100


def poisson_cdf(k: int, lam: float) -> float:
    lam = max(0.01, lam)
    return sum(math.exp(-lam) * (lam**i) / math.factorial(i) for i in range(k + 1))


def best_line(lam: float, lines: list[float]) -> tuple[str, int]:
    choices: list[tuple[str, float, float]] = []
    for line in lines:
        choices.append(("+", line, 1 - poisson_cdf(math.floor(line), lam)))
        choices.append(("−", line, poisson_cdf(math.floor(line), lam)))
    sign, line, prob = max(choices, key=lambda x: x[2])
    return f"{sign}{str(line).replace('.', ',')}", round(prob * 100)


CONFIDENCE_LEVEL = {"Baja": 0, "Media": 1, "Alta": 2}

def confidence(probability: float, quality: float) -> str:
    score = 0.60 * quality + 0.40 * abs(probability - 50) * 2
    if score >= 76:
        return "Alta"
    if score >= 56:
        return "Media"
    return "Baja"

def cap_confidence(value: str, maximum: str) -> str:
    if CONFIDENCE_LEVEL.get(value, 0) <= CONFIDENCE_LEVEL.get(maximum, 0):
        return value
    return maximum

def global_confidence_cap(quality: float) -> str:
    if quality >= 70:
        return "Alta"
    if quality >= 50:
        return "Media"
    return "Baja"

def reliability_level(score: float) -> str:
    if score >= 75:
        return "Alta"
    if score >= 55:
        return "Media"
    return "Baja"


def safe_external_url(url: str) -> bool:
    """Bloquea destinos locales/privados. No intenta saltar logins, CAPTCHA ni bloqueos."""
    try:
        p = urlparse(url)
        if p.scheme not in {"http", "https"} or not p.hostname:
            return False
        host = p.hostname.lower()
        if host in {"localhost", "127.0.0.1", "::1"} or host.endswith(".local"):
            return False
        try:
            ip = socket.gethostbyname(host)
            octets = [int(x) for x in ip.split(".")]
            if octets[0] in {10, 127}:
                return False
            if octets[0] == 192 and octets[1] == 168:
                return False
            if octets[0] == 172 and 16 <= octets[1] <= 31:
                return False
            if octets[0] == 169 and octets[1] == 254:
                return False
        except Exception:
            # DNS puede variar; httpx hará la resolución final.
            pass
        return True
    except Exception:
        return False


async def fetch_html(url: str, ttl: int = HTML_TTL) -> str:
    if not safe_external_url(url):
        raise ValueError("URL externa no permitida")
    key = f"html:{url}"
    cached = cache_get(key, ttl)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(18.0, connect=8.0),
            headers=HEADERS,
            follow_redirects=True,
        ) as client:
            r = await client.get(url)
        if r.status_code in {401, 403, 429}:
            raise RuntimeError(f"Fuente no accesible ({r.status_code})")
        r.raise_for_status()
        content_type = (r.headers.get("content-type") or "").lower()
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            raise RuntimeError("La fuente no devolvió HTML")
        text = r.text[:2_500_000]
        cache_set(key, text)
        return text
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer {urlparse(url).netloc}: {exc}") from exc


async def fetch_json(url: str, ttl: int = HTML_TTL) -> dict[str, Any]:
    """Lee JSON público que la propia web utiliza. No requiere clave ni suscripción."""
    if not safe_external_url(url):
        raise ValueError("URL externa no permitida")
    key = f"json:{url}"
    cached = cache_get(key, ttl)
    if cached is not None:
        return cached

    try:
        headers = dict(HEADERS)
        headers["Accept"] = "application/json,text/plain,*/*"
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(18.0, connect=8.0),
            headers=headers,
            follow_redirects=True,
        ) as client:
            r = await client.get(url)
        if r.status_code in {401, 403, 429}:
            raise RuntimeError(f"Fuente no accesible ({r.status_code})")
        r.raise_for_status()
        data = r.json()
        cache_set(key, data)
        return data
    except Exception as exc:
        raise RuntimeError(f"No se pudo leer JSON de {urlparse(url).netloc}: {exc}") from exc


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str = ""
    engine: str = ""


def unwrap_ddg(url: str) -> str:
    try:
        p = urlparse(url)
        qs = parse_qs(p.query)
        if "uddg" in qs:
            return unquote(qs["uddg"][0])
    except Exception:
        pass
    return url


def unwrap_bing(url: str) -> str:
    try:
        p = urlparse(url)
        if "bing.com" not in (p.netloc or "").lower():
            return url
        qs = parse_qs(p.query)
        raw = (qs.get("u") or [None])[0]
        if not raw:
            return url
        raw = unquote(raw)
        if raw.startswith("a1"):
            raw = raw[2:]
        padded = raw + "=" * (-len(raw) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode()).decode("utf-8", "ignore").strip()
        if decoded.startswith(("http://", "https://")):
            return decoded
    except Exception:
        pass
    return url


def clean_search_url(url: str) -> str:
    return unwrap_bing(unwrap_ddg(url))


def norm_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str) -> str:
    return norm_text(value).replace(" ", "-")


def _first_link(item: dict[str, Any]) -> str | None:
    links = item.get("links")
    if isinstance(links, list):
        for link in links:
            if isinstance(link, dict) and isinstance(link.get("href"), str):
                href = link["href"]
                if "/soccer/" in href and "/team" in href:
                    return href
    for key in ("link", "href", "url"):
        val = item.get(key)
        if isinstance(val, str) and val.startswith("http"):
            return val
        if isinstance(val, dict) and isinstance(val.get("href"), str):
            return val["href"]
    return None


def _walk_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_dicts(value)


def _team_from_search_item(item: dict[str, Any], query: str) -> dict[str, Any] | None:
    raw_id = item.get("id")
    uid = str(item.get("uid") or "")
    item_type = str(item.get("type") or "").lower()
    sport = str(item.get("sport") or item.get("sportSlug") or "").lower()

    # En fútbol ESPN usa uid s:600~t:ID. Aceptamos además type=team + sport=soccer.
    looks_team = item_type in {"team", "club"} or "~t:" in uid
    looks_soccer = sport in {"soccer", "football"} or uid.startswith("s:600") or "/soccer/" in str(item)
    if not (looks_team and looks_soccer):
        return None
    try:
        team_id = int(str(raw_id))
    except Exception:
        m = re.search(r"~t:(\d+)", uid)
        if not m:
            return None
        team_id = int(m.group(1))

    name = str(
        item.get("displayName")
        or item.get("name")
        or item.get("shortDisplayName")
        or item.get("shortName")
        or ""
    ).strip()
    if not name:
        return None

    nq = norm_text(query)
    nn = norm_text(name)
    q_tokens = set(nq.split())
    n_tokens = set(nn.split())
    if nq and nq not in nn and not q_tokens.intersection(n_tokens):
        return None

    href = _first_link(item) or f"https://www.espn.com/soccer/team/_/id/{team_id}/{slugify(name)}"
    league = str(item.get("league") or item.get("defaultLeagueSlug") or item.get("leagueSlug") or "").strip()
    return {
        "id": team_id,
        "nombre": name,
        "pais": league or "ESPN fútbol",
        "logo": f"https://a.espncdn.com/i/teamlogos/soccer/500/{team_id}.png",
        "fuente": href,
    }


async def search_espn_public_json(query: str) -> list[dict[str, Any]]:
    """Buscador principal: datos públicos cargados por la propia web de ESPN."""
    urls = [
        f"https://site.web.api.espn.com/apis/common/v3/search?region=us&lang=en&query={quote(query)}&limit=30&mode=prefix",
        f"https://site.web.api.espn.com/apis/search/v2?query={quote(query)}&limit=30",
    ]
    payloads = await asyncio.gather(*(fetch_json(u, SEARCH_TTL) for u in urls), return_exceptions=True)
    found: list[dict[str, Any]] = []
    seen: set[int] = set()
    for payload in payloads:
        if not isinstance(payload, (dict, list)):
            continue
        for item in _walk_dicts(payload):
            candidate = _team_from_search_item(item, query)
            if not candidate or candidate["id"] in seen:
                continue
            seen.add(candidate["id"])
            found.append(candidate)

    nq = norm_text(query)
    def score(c: dict[str, Any]) -> tuple[int, int]:
        nn = norm_text(c["nombre"])
        if nn == nq:
            return (0, len(nn))
        if nn.startswith(nq):
            return (1, len(nn))
        if nq in nn:
            return (2, len(nn))
        return (3, len(nn))
    found.sort(key=score)
    return found[:12]


# Respaldo mínimo para clubes probados. Sólo se usa si el buscador público no responde.
KNOWN_TEAMS = {
    "river plate": {"id": 16, "nombre": "River Plate"},
    "colo colo": {"id": 2688, "nombre": "Colo Colo"},
    "union espanola": {"id": 4132, "nombre": "Unión Española"},
}

def known_team_candidates(query: str) -> list[dict[str, Any]]:
    nq = norm_text(query)
    out = []
    for key, item in KNOWN_TEAMS.items():
        if nq == key or nq in key or key in nq:
            tid = item["id"]
            name = item["nombre"]
            out.append({
                "id": tid,
                "nombre": name,
                "pais": "Respaldo local",
                "logo": f"https://a.espncdn.com/i/teamlogos/soccer/500/{tid}.png",
                "fuente": f"https://www.espn.com/soccer/team/_/id/{tid}/{slugify(name)}",
            })
    return out


async def search_espn_direct(query: str) -> list[SearchResult]:
    url = f"https://www.espn.com/search/_/q/{quote(query)}"
    try:
        raw = await fetch_html(url, SEARCH_TTL)
    except Exception:
        return []
    soup = BeautifulSoup(raw, "html.parser")
    out: list[SearchResult] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if re.search(r"/(?:soccer/team|futbol/equipo)/_/id/\d+", href):
            if href.startswith("/"):
                href = "https://www.espn.com" + href
            title = " ".join(a.stripped_strings).strip()
            if title:
                out.append(SearchResult(title=title, url=href, engine="ESPN"))
    return out[:15]


async def search_ddg(query: str) -> list[SearchResult]:
    url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
    try:
        raw = await fetch_html(url, SEARCH_TTL)
    except Exception:
        return []
    soup = BeautifulSoup(raw, "html.parser")
    out: list[SearchResult] = []
    for node in soup.select(".result"):
        a = node.select_one("a.result__a") or node.find("a", href=True)
        if not a:
            continue
        href = unwrap_ddg(a.get("href", ""))
        if not href.startswith("http"):
            continue
        sn = node.select_one(".result__snippet")
        out.append(
            SearchResult(
                title=" ".join(a.stripped_strings),
                url=href,
                snippet=" ".join(sn.stripped_strings) if sn else "",
                engine="DuckDuckGo",
            )
        )
    return out[:10]


async def search_bing(query: str) -> list[SearchResult]:
    url = f"https://www.bing.com/search?q={quote(query)}&setlang=es"
    try:
        raw = await fetch_html(url, SEARCH_TTL)
    except Exception:
        return []
    soup = BeautifulSoup(raw, "html.parser")
    out: list[SearchResult] = []
    for node in soup.select("li.b_algo"):
        a = node.select_one("h2 a")
        if not a or not a.get("href"):
            continue
        sn = node.select_one(".b_caption p")
        href = clean_search_url(a["href"])
        out.append(
            SearchResult(
                title=" ".join(a.stripped_strings),
                url=href,
                snippet=" ".join(sn.stripped_strings) if sn else "",
                engine="Bing",
            )
        )
    return out[:10]


async def web_search(query: str, limit: int = 8) -> list[SearchResult]:
    key = f"websearch:{query.lower().strip()}"
    cached = cache_get(key, SEARCH_TTL)
    if cached is not None:
        return [SearchResult(**x) for x in cached]

    ddg, bing = await asyncio.gather(search_ddg(query), search_bing(query))
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for item in ddg + bing:
        clean = clean_search_url(item.url).split("#")[0]
        if clean in seen or not clean.startswith("http"):
            continue
        seen.add(clean)
        item.url = clean
        merged.append(item)
        if len(merged) >= limit:
            break
    cache_set(key, [asdict(x) for x in merged])
    return merged


def clean_team_title(title: str) -> str:
    title = html.unescape(title)
    title = re.sub(r"\s*[-|–]\s*ESPN.*$", "", title, flags=re.I)
    title = re.sub(r"\s+(Scores|Resultados|Fixtures|Schedule|Stats|Estadísticas|Noticias).*", "", title, flags=re.I)
    return re.sub(r"\s+", " ", title).strip(" -|")


def candidate_from_url(title: str, url: str) -> dict[str, Any] | None:
    m = re.search(r"/(?:soccer/team|football/team|futbol/equipo)/_/id/(\d+)(?:/([^?#]+))?", url)
    if not m:
        return None
    team_id = int(m.group(1))
    name = clean_team_title(title)
    if not name or len(name) > 80:
        slug = (m.group(2) or "").replace("%20", " ").replace("-", " ")
        name = slug.title() or f"Equipo {team_id}"
    return {
        "id": team_id,
        "nombre": name,
        "pais": "Fuente web ESPN",
        "logo": f"https://a.espncdn.com/i/teamlogos/soccer/500/{team_id}.png",
        "fuente": url,
    }


async def discover_teams(query: str) -> list[dict[str, Any]]:
    key = f"teams:{norm_text(query)}"
    cached = cache_get(key, SEARCH_TTL)
    if cached is not None:
        return cached

    # 1) Respaldo local inmediato para clubes ya verificados.
    known = known_team_candidates(query)
    if known:
        cache_set(key, known)
        return known

    # 2) Buscador público que utiliza la propia web de ESPN (sin clave ni pago).
    public = await search_espn_public_json(query)
    if public:
        cache_set(key, public)
        return public

    # 3) Último respaldo: páginas HTML / motores de búsqueda públicos.
    direct = await search_espn_direct(query)
    search_query = f'site:espn.com/soccer/team/_/id "{query}" fútbol'
    web = await web_search(search_query, 12)
    web_es = await web_search(f'site:espndeportes.espn.com/futbol/equipo/_/id "{query}"', 8)

    query_tokens = {t for t in norm_text(query).split() if len(t) > 1}
    candidates: list[dict[str, Any]] = []
    seen: set[int] = set()
    for r in direct + web + web_es:
        c = candidate_from_url(r.title, r.url)
        if not c or c["id"] in seen:
            continue
        name_tokens = set(norm_text(c["nombre"]).split())
        if query_tokens and not (query_tokens & name_tokens):
            continue
        seen.add(c["id"])
        candidates.append(c)
        if len(candidates) >= 12:
            break

    cache_set(key, candidates)
    return candidates


def text_of(node) -> str:
    return " ".join(node.stripped_strings) if node else ""


def extract_team_links(row) -> list[dict[str, Any]]:
    out = []
    seen = set()
    for a in row.find_all("a", href=True):
        href = a.get("href", "")
        m = re.search(r"/(?:soccer/team|football/team|futbol/equipo)/_/id/(\d+)(?:/([^?#]+))?", href)
        if not m:
            continue
        tid = int(m.group(1))
        if tid in seen:
            continue
        seen.add(tid)
        name = text_of(a)
        if not name:
            name = (m.group(2) or "").replace("-", " ").title()
        if href.startswith("/"):
            href = "https://www.espn.com" + href
        out.append({"id": tid, "name": name, "url": href})
    return out


def extract_game_url(row) -> str | None:
    for a in row.find_all("a", href=True):
        href = a.get("href", "")
        if re.search(r"/(?:soccer|football|futbol)/(?:match|partido)/_/gameId/\d+", href):
            if href.startswith("/"):
                href = "https://www.espn.com" + href
            return href
    return None


def extract_iso_from_node(node) -> str | None:
    if not node:
        return None
    for tag in [node] + list(node.find_all(True)):
        for attr in ("datetime", "data-date", "data-time", "data-dt"):
            val = tag.attrs.get(attr)
            if isinstance(val, str) and re.match(r"20\d\d-\d\d-\d\d[T ]", val):
                return val
    # ESPN suele incrustar timestamps ISO en atributos/JSON del documento.
    m = re.search(r"20\d\d-\d\d-\d\dT\d\d:\d\d(?::\d\d)?(?:Z|[+-]\d\d:?\d\d)", str(node))
    return m.group(0) if m else None


def parse_event_start_from_html(raw: str) -> datetime | None:
    soup = BeautifulSoup(raw, "html.parser")
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or script.get_text() or "{}")
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            start = item.get("startDate")
            if start:
                try:
                    dt = dateparser.isoparse(start)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=TZ)
                    return dt.astimezone(TZ)
                except Exception:
                    pass
    # Fallback: timestamps UTC explícitos del contenido.
    iso_candidates = re.findall(r"20\d\d-\d\d-\d\dT\d\d:\d\d(?::\d\d)?(?:\.\d+)?Z", raw)
    now = now_chile()
    for iso in iso_candidates:
        try:
            dt = dateparser.isoparse(iso).astimezone(TZ)
            if now - timedelta(days=2) <= dt <= now + timedelta(days=370):
                return dt
        except Exception:
            continue
    return None


def parse_fixture_date_text(date_text: str, time_text: str) -> datetime | None:
    if not date_text:
        return None
    now = now_chile()
    cleaned = re.sub(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s*", "", date_text, flags=re.I)
    year = now.year
    combo = f"{cleaned} {year} {time_text}".strip()
    try:
        dt = dateparser.parse(combo, fuzzy=True, default=now.replace(month=1, day=1, hour=12, minute=0, second=0, microsecond=0))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=TZ)
        dt = dt.astimezone(TZ)
        # En diciembre, una fecha de enero/febrero probablemente corresponde al año siguiente.
        if dt < now - timedelta(days=30):
            dt = dt.replace(year=dt.year + 1)
        return dt
    except Exception:
        return None


async def get_next_fixture(team_id: int) -> dict[str, Any]:
    url = f"https://www.espn.com/soccer/team/fixtures/_/id/{team_id}"
    raw = await fetch_html(url)
    soup = BeautifulSoup(raw, "html.parser")
    now = now_chile()
    fixtures: list[dict[str, Any]] = []

    for row in soup.find_all("tr"):
        teams = extract_team_links(row)
        if len(teams) < 2 or team_id not in {t["id"] for t in teams}:
            continue
        cells = [text_of(td) for td in row.find_all(["td", "th"])]
        if not cells:
            continue
        row_text = " | ".join(cells)
        if re.search(r"\bFT\b|FT-Pens|Final", row_text, flags=re.I):
            continue
        game_url = extract_game_url(row)
        iso = extract_iso_from_node(row)
        dt = None
        if iso:
            try:
                dt = dateparser.isoparse(iso)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=TZ)
                dt = dt.astimezone(TZ)
            except Exception:
                dt = None

        date_text = cells[0] if cells else ""
        time_text = ""
        for c in cells[1:]:
            if re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM)\b|\bTBD\b", c, flags=re.I):
                time_text = c
                break
        if not dt and "TBD" not in time_text.upper():
            dt = parse_fixture_date_text(date_text, time_text)

        match_competition = ""
        # Si tenemos página del partido, intentamos una hora absoluta y la competición.
        if game_url:
            try:
                match_raw = await fetch_html(game_url)
                precise = parse_event_start_from_html(match_raw)
                if precise:
                    dt = precise
                match_competition = parse_competition_from_match_html(match_raw)
            except Exception:
                pass

        if dt and dt < now - timedelta(hours=3):
            continue

        # El primer enlace suele ser local y el segundo visita.
        home = teams[0]
        away = teams[1]
        competition = (
            sanitize_competition(match_competition, home["name"], away["name"])
            or sanitize_competition(competition_cell(row), home["name"], away["name"])
            or "Competición no identificada"
        )

        fixtures.append(
            {
                "home": home,
                "away": away,
                "dt": dt,
                "date_text": date_text,
                "time_text": time_text,
                "competition": competition,
                "game_url": game_url,
                "source_url": url,
            }
        )

    if not fixtures:
        raise RuntimeError("La fuente web no entregó un próximo partido identificable para este equipo.")

    # Priorizamos fecha absoluta; los TBD quedan al final conservando orden de ESPN.
    fixtures.sort(key=lambda x: x["dt"] or now + timedelta(days=365))
    return fixtures[0]


@dataclass
class ResultRow:
    home_id: int | None
    away_id: int | None
    home_name: str
    away_name: str
    home_goals: int
    away_goals: int
    competition: str


async def get_results(team_id: int, limit: int = 15) -> list[ResultRow]:
    url = f"https://www.espn.com/soccer/team/results/_/id/{team_id}"
    raw = await fetch_html(url)
    soup = BeautifulSoup(raw, "html.parser")
    out: list[ResultRow] = []
    for row in soup.find_all("tr"):
        teams = extract_team_links(row)
        if len(teams) < 2 or team_id not in {t["id"] for t in teams}:
            continue
        cells = [text_of(td) for td in row.find_all(["td", "th"])]
        row_text = " | ".join(cells)
        score = re.search(r"(?<!\d)(\d{1,2})\s*[-–]\s*(\d{1,2})(?!\d)", row_text)
        if not score or not re.search(r"FT|Final", row_text, flags=re.I):
            continue
        hg, ag = int(score.group(1)), int(score.group(2))
        competition = competition_cell(row)
        out.append(
            ResultRow(
                home_id=teams[0]["id"],
                away_id=teams[1]["id"],
                home_name=teams[0]["name"],
                away_name=teams[1]["name"],
                home_goals=hg,
                away_goals=ag,
                competition=competition,
            )
        )
        if len(out) >= limit:
            break
    return out


def summarize_results(rows: list[ResultRow], team_id: int, venue: str | None = None, n: int = 10) -> dict[str, Any]:
    chosen = []
    for r in rows:
        is_home = r.home_id == team_id
        is_away = r.away_id == team_id
        if not (is_home or is_away):
            continue
        if venue == "home" and not is_home:
            continue
        if venue == "away" and not is_away:
            continue
        chosen.append(r)
        if len(chosen) >= n:
            break

    if not chosen:
        return {"played": 0, "ppg": 1.25, "gf": 1.2, "ga": 1.2, "form": "", "wins": 0, "draws": 0, "losses": 0}

    pts = gf = ga = wins = draws = losses = 0
    form = []
    for r in chosen:
        if r.home_id == team_id:
            a, b = r.home_goals, r.away_goals
        else:
            a, b = r.away_goals, r.home_goals
        gf += a
        ga += b
        if a > b:
            wins += 1
            pts += 3
            form.append("G")
        elif a == b:
            draws += 1
            pts += 1
            form.append("E")
        else:
            losses += 1
            form.append("P")
    played = len(chosen)
    return {
        "played": played,
        "ppg": pts / played,
        "gf": gf / played,
        "ga": ga / played,
        "form": "".join(form),
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }


async def get_standing(team_id: int, team_name: str) -> dict[str, Any]:
    url = f"https://www.espn.com/soccer/team/_/id/{team_id}"
    raw = await fetch_html(url)
    soup = BeautifulSoup(raw, "html.parser")
    target = re.sub(r"\s+", " ", team_name).lower().strip()
    for table in soup.find_all("table"):
        headers = [text_of(th).upper() for th in table.find_all("th")]
        header_text = " ".join(headers)
        if not any(x in header_text for x in ["GP", "PTS", " P ", "J "]):
            continue
        rows = table.find_all("tr")
        parsed_rows = []
        for row in rows:
            cells = [text_of(td) for td in row.find_all(["td", "th"])]
            if len(cells) < 5:
                continue
            parsed_rows.append(cells)
        for idx, cells in enumerate(parsed_rows, start=1):
            joined = " ".join(cells).lower()
            if target not in joined and not all(tok in joined for tok in target.split()[:2]):
                continue
            nums = []
            for c in cells:
                if re.fullmatch(r"[+-]?\d+", c.strip()):
                    nums.append(int(c))
            # ESPN: GP W D L GD P. Tomamos valores desde el final cuando están disponibles.
            played = wins = draws = losses = gd = points = None
            if len(nums) >= 6:
                played, wins, draws, losses, gd, points = nums[-6:]
            return {
                "rank": idx,
                "total": len(parsed_rows),
                "played": played,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "gd": gd,
                "points": points,
                "source_url": url,
            }
    # Fallback al encabezado: "2nd in ..."
    page_text = text_of(soup)
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\s+in\s+([^\n]{3,80})", page_text, re.I)
    if m:
        return {"rank": int(m.group(1)), "total": None, "source_url": url}
    return {"rank": None, "total": None, "source_url": url}


async def get_discipline(team_id: int, played_hint: int | None = None) -> dict[str, Any]:
    urls = [
        f"https://www.espn.com/soccer/team/stats/_/id/{team_id}/view/discipline",
        f"https://www.espn.com/soccer/team/stats?id={team_id}&view=discipline",
    ]
    for url in urls:
        try:
            raw = await fetch_html(url)
        except Exception:
            continue
        soup = BeautifulSoup(raw, "html.parser")
        for table in soup.find_all("table"):
            headers = [text_of(th).strip().lower() for th in table.find_all("th")]
            h = " ".join(headers)
            if "yc" not in h or "rc" not in h:
                continue
            yc_total = rc_total = 0
            max_p = 0
            for row in table.find_all("tr"):
                cells = [text_of(td).strip() for td in row.find_all("td")]
                if len(cells) < 4:
                    continue
                ints = []
                for c in cells:
                    if re.fullmatch(r"\d+", c):
                        ints.append(int(c))
                # Normalmente termina en P, YC, RC, Pts; tomamos de forma conservadora.
                if len(ints) >= 4:
                    p, yc, rc = ints[-4], ints[-3], ints[-2]
                    max_p = max(max_p, p)
                    yc_total += yc
                    rc_total += rc
            played = played_hint or max_p or 1
            return {
                "yellow_total": yc_total,
                "red_total": rc_total,
                "yellow_pg": yc_total / max(1, played),
                "red_pg": rc_total / max(1, played),
                "source_url": url,
            }
    return {"yellow_pg": None, "red_pg": None, "source_url": None}


async def discover_transfermarkt_injuries(team_name: str, match_dt: datetime | None) -> dict[str, Any]:
    query = f'site:transfermarkt.com "{team_name}" "Suspensions and injuries"'
    results = await web_search(query, 6)
    target = None
    for r in results:
        if "transfermarkt." in urlparse(r.url).netloc and "sperrenundverletzungen" in r.url:
            target = r
            break
    if not target:
        return {"count": None, "players": [], "source_url": None}
    try:
        raw = await fetch_html(target.url, CONTEXT_TTL)
        soup = BeautifulSoup(raw, "html.parser")
        players: list[str] = []
        # Transfermarkt pone el cuadro de bajas antes del bloque "Risk of suspension".
        risk = soup.find(string=re.compile(r"Risk of suspension", re.I))
        stop_table = risk.find_parent("table") if risk else None
        for table in soup.find_all("table"):
            if stop_table is not None and table == stop_table:
                break
            header = text_of(table).lower()
            if "player" not in header or not any(k in header for k in ["reason", "expected return", "since"]):
                continue
            for row in table.find_all("tr"):
                cells = [text_of(td) for td in row.find_all("td")]
                if len(cells) < 3:
                    continue
                # El nombre suele estar en el primer/segundo campo textual del jugador.
                name = ""
                a = row.find("a", href=re.compile(r"/profil/spieler/|/spieler/"))
                if a:
                    name = text_of(a)
                if not name:
                    name = cells[0]
                name = re.sub(r"\s+", " ", name).strip()
                if name and name.lower() not in {"player", "injuries", "suspensions"} and name not in players:
                    players.append(name)
            if players:
                break
        if players:
            return {"count": len(players), "players": players[:8], "source_url": target.url}
        return {"count": None, "players": [], "source_url": target.url}
    except Exception:
        return {"count": None, "players": [], "source_url": target.url}



ABSENCE_KEYWORDS = re.compile(
    r"\b(?:lesi[oó]n|lesionado|injury|injured|suspendido|suspension|sanci[oó]n|"
    r"cruciate|ligament|muscle|muscular|hamstring|knee|ankle|calf|thigh|"
    r"illness|sick|tear|strain|sprain|fracture|problema f[ií]sico)\b",
    re.I,
)

def clean_absence_name(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip(" -|")
    if not text:
        return ""
    m = ABSENCE_KEYWORDS.search(text)
    if m:
        text = text[:m.start()].strip(" -|")
    text = re.sub(r"^(?:Image:\s*)+", "", text, flags=re.I).strip()
    # Evitar encabezados o frases enteras.
    words = text.split()
    if not (1 <= len(words) <= 6):
        return ""
    if any(ch.isdigit() for ch in text):
        return ""
    if norm_text(text) in {"estado actual", "current status", "player", "jugador", "injuries", "lesiones", "suspensions", "sanciones"}:
        return ""
    return text

def parse_besoccer_current_absences(raw: str) -> list[str]:
    soup = BeautifulSoup(raw, "html.parser")
    heading = soup.find(
        lambda tag: tag.name in {"h2", "h3", "h4"} and
        re.search(r"Estado actual|Current status", text_of(tag), re.I)
    )
    if not heading:
        return []

    players: list[str] = []
    for node in heading.find_all_next():
        if node is not heading and node.name in {"h2", "h3", "h4"}:
            # El siguiente encabezado mensual marca el fin del estado actual.
            break
        if node.name not in {"li", "div", "tr"}:
            continue
        txt = text_of(node)
        if not txt or not ABSENCE_KEYWORDS.search(txt):
            continue

        name = ""
        for a in node.find_all("a", href=True):
            href = a.get("href", "").lower()
            if any(k in href for k in ["/jugador/", "/player/", "/perfil/", "/profil/"]):
                candidate = clean_absence_name(text_of(a))
                if candidate:
                    name = candidate
                    break
        if not name:
            name = clean_absence_name(txt)

        if name and name not in players:
            players.append(name)
        if len(players) >= 12:
            break
    return players

async def discover_besoccer_injuries(team_name: str) -> dict[str, Any]:
    queries = [
        f'site:besoccer.es/equipo/lesionados-sancionados "{team_name}"',
        f'site:es.besoccer.com/equipo/lesionados-sancionados "{team_name}"',
        f'site:besoccer.com/team/injuries-suspensions "{team_name}"',
    ]
    batches = await asyncio.gather(*(web_search(q, 5) for q in queries))
    results = [r for batch in batches for r in batch]
    seen = set()
    for r in results:
        r.url = clean_search_url(r.url)
        host = (urlparse(r.url).netloc or "").lower()
        if "besoccer" not in host or r.url in seen:
            continue
        seen.add(r.url)
        try:
            raw = await fetch_html(r.url, CONTEXT_TTL)
            players = parse_besoccer_current_absences(raw)
            if players:
                return {
                    "count": len(players),
                    "players": players[:8],
                    "source_url": r.url,
                    "source_name": "BeSoccer",
                }
        except Exception:
            continue
    return {"count": None, "players": [], "source_url": None, "source_name": None}

async def discover_absences(team_name: str, match_dt: datetime | None) -> dict[str, Any]:
    primary = await discover_transfermarkt_injuries(team_name, match_dt)
    if primary.get("count") is not None and primary.get("count", 0) > 0:
        primary["source_name"] = "Transfermarkt"
        return primary
    fallback = await discover_besoccer_injuries(team_name)
    if fallback.get("count") is not None:
        return fallback
    # Si Transfermarkt respondió con cero de forma estructurada, conservarlo.
    if primary.get("count") == 0 and primary.get("source_url"):
        primary["source_name"] = "Transfermarkt"
        return primary
    return fallback

def _clean_official_name(value: str | None) -> str | None:
    if not value:
        return None
    value = re.split(r"\(|\bVAR\b|\bAssistant\b|\bAsistente\b|[.;]", value, maxsplit=1, flags=re.I)[0]
    value = re.sub(r"\s+", " ", value).strip(" -,:")
    words = value.split()
    if not (2 <= len(words) <= 6):
        return None
    return value

def parse_referee_var_from_text(text: str) -> tuple[str | None, str | None]:
    text = re.sub(r"\s+", " ", html.unescape(text or ""))
    referee = None
    var = None
    patterns_ref = [
        r"(?:Árbitro|Arbitro|Referee|Árbitro principal|Main referee)\s*[:\-]\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ.' -]{3,55})",
        r"(?:será arbitrado por|will be refereed by)\s+([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ.' -]{3,55})",
    ]
    for p in patterns_ref:
        m = re.search(p, text, re.I)
        if m:
            referee = _clean_official_name(m.group(1))
            if referee:
                break

    patterns_var = [
        r"\bVAR\s*[:\-]\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ.' -]{3,55})",
        r"(?:Video Assistant Referee)\s*[:\-]\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ.' -]{3,55})",
    ]
    for p in patterns_var:
        m = re.search(p, text, re.I)
        if m:
            var = _clean_official_name(m.group(1))
            if var:
                break
    return referee, var

def parse_referee_stats_from_text(text: str) -> tuple[float | None, float | None]:
    text = re.sub(r"\s+", " ", html.unescape(text or ""))
    yellow = red = None

    yellow_patterns = [
        r"(?:yellow cards?(?: per game| per match)?|amarillas(?: por partido)?|tarjetas amarillas(?: por partido)?)[^0-9]{0,35}(\d{1,2}(?:[.,]\d{1,2})?)",
        r"(\d{1,2}(?:[.,]\d{1,2})?)\s*(?:yellow cards?|amarillas|tarjetas amarillas)\s*(?:per game|per match|por partido)?",
    ]
    red_patterns = [
        r"(?:red cards?(?: per game| per match)?|rojas(?: por partido)?|tarjetas rojas(?: por partido)?)[^0-9]{0,35}(\d(?:[.,]\d{1,2})?)",
        r"(\d(?:[.,]\d{1,2})?)\s*(?:red cards?|rojas|tarjetas rojas)\s*(?:per game|per match|por partido)?",
    ]

    for p in yellow_patterns:
        m = re.search(p, text, re.I)
        if m:
            try:
                val = float(m.group(1).replace(",", "."))
                if 1.0 <= val <= 12.0:
                    yellow = val
                    break
            except Exception:
                pass

    for p in red_patterns:
        m = re.search(p, text, re.I)
        if m:
            try:
                val = float(m.group(1).replace(",", "."))
                if 0.0 <= val <= 2.5:
                    red = val
                    break
            except Exception:
                pass
    return yellow, red

async def discover_referee_stats(referee: str | None) -> dict[str, Any]:
    if not referee:
        return {"yellow_pg": None, "red_pg": None, "source_url": None}
    key = f"refstats:{norm_text(referee)}"
    cached = cache_get(key, CONTEXT_TTL)
    if cached is not None:
        return cached

    queries = [
        f'site:valuestats.com "{referee}" referee yellow cards',
        f'site:footymetrics.com "{referee}" referee cards',
        f'"{referee}" "yellow cards per match" referee',
        f'"{referee}" "tarjetas amarillas" árbitro',
    ]
    batches = await asyncio.gather(*(web_search(q, 5) for q in queries))
    results = [r for batch in batches for r in batch]
    seen = set()

    # Primero intentar los snippets de búsqueda, sólo desde fuentes permitidas.
    for r in results:
        r.url = clean_search_url(r.url)
        if r.url in seen or not is_trusted_context_source(r.url):
            continue
        seen.add(r.url)
        y, rr = parse_referee_stats_from_text(f"{r.title} {r.snippet}")
        if y is not None or rr is not None:
            out = {"yellow_pg": y, "red_pg": rr, "source_url": r.url}
            cache_set(key, out)
            return out

    # Luego páginas públicas legibles, siempre de dominios permitidos.
    trusted_results = []
    seen_trusted = set()
    for r in results:
        r.url = clean_search_url(r.url)
        if r.url in seen_trusted or not is_trusted_context_source(r.url):
            continue
        seen_trusted.add(r.url)
        trusted_results.append(r)

    for r in trusted_results[:8]:
        try:
            raw = await fetch_html(r.url, CONTEXT_TTL)
            visible = text_of(BeautifulSoup(raw, "html.parser"))[:180_000]
            y, rr = parse_referee_stats_from_text(visible)
            if y is not None or rr is not None:
                out = {"yellow_pg": y, "red_pg": rr, "source_url": r.url}
                cache_set(key, out)
                return out
        except Exception:
            continue

    out = {"yellow_pg": None, "red_pg": None, "source_url": None}
    cache_set(key, out)
    return out


OFFICIAL_DOMAIN_HINTS = [
    "conmebol.com", "gol.conmebol.com", "uefa.com", "fifa.com", "anfp.cl", "afa.com.ar",
    "premierleague.com", "laliga.com", "bundesliga.com", "legaseriea.it",
    "ligue1.com", "cbf.com.br", "auf.org.uy", "dimayor.com.co", "ligamx.net",
    "fpf.org.pe", "fef.ec", "apf.org.py", "fpf.pt", "rfef.es", "thefa.com",
    "fff.fr", "dfb.de", "figc.it", "knvb.nl", "rbfa.be", "concacaf.com",
]

SPORT_DATA_DOMAINS = [
    "espn.com", "espn.com.ar", "espn.cl", "espn.com.co", "espn.com.mx",
    "sofascore.com", "fotmob.com", "flashscore.com", "besoccer.com",
    "es.besoccer.com", "transfermarkt.com", "transfermarkt.us",
    "worldfootball.net", "soccerway.com", "footystats.org", "valuestats.com",
    "footymetrics.com",
]

TRUSTED_NEWS_DOMAINS = [
    # Argentina
    "tycsports.com", "ole.com.ar", "clarin.com", "lanacion.com.ar",
    "infobae.com", "pagina12.com.ar",
    # Chile
    "emol.com", "latercera.com", "biobiochile.cl", "cooperativa.cl",
    "adnradio.cl", "t13.cl", "24horas.cl",
    # Colombia
    "eltiempo.com", "elespectador.com", "caracol.com.co", "futbolred.com",
    "antena2.com", "wradio.com.co",
    # Brasil
    "ge.globo.com", "globo.com", "uol.com.br", "lance.com.br",
    # Uruguay / Paraguay / Perú / Ecuador
    "ovaciondigital.com.uy", "elpais.com.uy", "abc.com.py", "ultimahora.com",
    "elcomercio.pe", "depor.com", "libero.pe", "eluniverso.com", "primicias.ec",
    # México / USA latino
    "mediotiempo.com", "record.com.mx", "eluniversal.com.mx", "tudn.com",
    # España
    "marca.com", "as.com", "mundodeportivo.com", "sport.es", "elpais.com",
    # Inglaterra
    "bbc.com", "theguardian.com", "skysports.com", "independent.co.uk",
    # Italia / Alemania / Francia / Portugal
    "gazzetta.it", "corrieredellosport.it", "tuttosport.com",
    "kicker.de", "bild.de", "lequipe.fr", "rmcsport.bfmtv.com",
    "abola.pt", "record.pt", "ojogo.pt",
]

BETTING_DOMAINS = [
    "oddsportal.com", "oddschecker.com", "betexplorer.com", "sportingbet.com",
    "stake.bet.ar", "caliente.mx", "sports.caliente.mx",
]

BLOCKED_SOURCE_DOMAINS = [
    "reddit.com", "facebook.com", "instagram.com", "tiktok.com", "twitter.com",
    "x.com", "youtube.com", "youtu.be", "xvideos.com", "pornhub.com",
    "pinterest.com", "quora.com", "threads.net", "telegram.me", "t.me",
]

def _host_matches(host: str, domain: str) -> bool:
    host = (host or "").lower().split(":")[0]
    domain = domain.lower()
    return host == domain or host.endswith("." + domain)

def host_in_domains(url: str, domains: list[str]) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(_host_matches(host, d) for d in domains)

def is_blocked_source(url: str) -> bool:
    return host_in_domains(url, BLOCKED_SOURCE_DOMAINS)

def is_official_domain(url: str) -> bool:
    return host_in_domains(url, OFFICIAL_DOMAIN_HINTS)

def source_trust_type(url: str) -> str | None:
    if not url or is_blocked_source(url):
        return None
    if is_official_domain(url):
        return "Oficial"
    if host_in_domains(url, SPORT_DATA_DOMAINS):
        return "Estadística deportiva"
    if host_in_domains(url, TRUSTED_NEWS_DOMAINS):
        return "Medio confiable"
    if host_in_domains(url, BETTING_DOMAINS):
        return "Mercado"
    return None

def is_trusted_context_source(url: str) -> bool:
    kind = source_trust_type(url)
    return kind in {"Oficial", "Estadística deportiva", "Medio confiable"}

COMPETITION_HINTS = [
    ("CONMEBOL Sudamericana", [r"\bsudamericana\b", r"\bconmebol sudamericana\b"]),
    ("CONMEBOL Libertadores", [r"\blibertadores\b", r"\bconmebol libertadores\b"]),
    ("Liga Profesional Argentina", [r"\bliga profesional\b", r"\bprimera nacional\b", r"\bargentine primera\b"]),
    ("Primera División de Chile", [r"\bprimera división de chile\b", r"\bchilean primera\b", r"\bliga de primera\b"]),
    ("Copa Chile", [r"\bcopa chile\b"]),
    ("Copa Argentina", [r"\bcopa argentina\b"]),
    ("UEFA Champions League", [r"\bchampions league\b"]),
    ("UEFA Europa League", [r"\beuropa league\b"]),
    ("UEFA Conference League", [r"\bconference league\b"]),
    ("LaLiga", [r"\blaliga\b", r"\bspanish laliga\b"]),
    ("Premier League", [r"\bpremier league\b"]),
    ("Serie A", [r"\bitalian serie a\b"]),
    ("Bundesliga", [r"\bbundesliga\b"]),
    ("Ligue 1", [r"\bligue 1\b"]),
    ("Liga MX", [r"\bliga mx\b"]),
    ("Brasileirão", [r"\bbrasileir[aã]o\b", r"\bbrazilian serie a\b"]),
]

def competition_from_text(value: str) -> str:
    clean = re.sub(r"\s+", " ", value or "").strip()
    low = clean.lower()
    for canonical, patterns in COMPETITION_HINTS:
        for pattern in patterns:
            if re.search(pattern, low, re.I):
                return canonical
    return ""

def parse_competition_from_match_html(raw: str) -> str:
    # 1) JSON/metadatos comunes de ESPN.
    candidates = []
    patterns = [
        r'"leagueName"\s*:\s*"([^"]{3,100})"',
        r'"competitionName"\s*:\s*"([^"]{3,100})"',
        r'"tournamentName"\s*:\s*"([^"]{3,100})"',
        r'"league"\s*:\s*\{[^{}]{0,1200}?"name"\s*:\s*"([^"]{3,100})"',
        r'"competition"\s*:\s*\{[^{}]{0,1200}?"name"\s*:\s*"([^"]{3,100})"',
    ]
    for p in patterns:
        for m in re.finditer(p, raw, re.I | re.S):
            candidates.append(html.unescape(m.group(1)))
    for c in candidates:
        found = competition_from_text(c)
        if found:
            return found

    # 2) Fallback restringido a nombres reconocibles de competiciones.
    text = BeautifulSoup(raw, "html.parser").get_text(" ", strip=True)[:250_000]
    return competition_from_text(text)

def sanitize_competition(value: str, home_name: str = "", away_name: str = "") -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    if not value:
        return ""
    nv = norm_text(value)
    if nv in {norm_text(home_name), norm_text(away_name)}:
        return ""
    # No aceptar como competición un nombre de equipo incrustado.
    if home_name and norm_text(home_name) == nv:
        return ""
    if away_name and norm_text(away_name) == nv:
        return ""
    recognized = competition_from_text(value)
    return recognized or ""

def competition_cell(row) -> str:
    table = row.find_parent("table")
    cells = [text_of(td).strip() for td in row.find_all("td")]

    # Sólo usar una columna si el encabezado realmente está alineado con las celdas.
    if table:
        header_row = table.find("tr")
        headers = [text_of(th).strip().upper() for th in header_row.find_all(["th", "td"])] if header_row else []
        if headers and len(headers) == len(cells):
            for i, h in enumerate(headers):
                if "COMPETITION" in h or "COMPETICIÓN" in h:
                    candidate = competition_from_text(cells[i])
                    if candidate:
                        return candidate

    # Fallback seguro: buscar exclusivamente nombres conocidos de competición.
    for c in cells:
        candidate = competition_from_text(c)
        if candidate:
            return candidate
    return ""

async def discover_match_context(home: str, away: str, competition: str, dt: datetime | None) -> dict[str, Any]:
    date_key = dt.strftime("%Y-%m-%d") if dt else "próximo partido"
    key = f"context:{home}:{away}:{date_key}"
    cached = cache_get(key, CONTEXT_TTL)
    if cached is not None:
        return cached

    nq = norm_text(competition)
    targeted = []
    if "conmebol" in nq or "sudamericana" in nq or "libertadores" in nq:
        targeted.append(f'site:gol.conmebol.com "{home}" "{away}" "Ficha del partido"')
        targeted.append(f'site:gol.conmebol.com "{home}" "{away}" árbitro VAR')
        targeted.append(f'site:conmebol.com "{home}" "{away}" árbitro VAR')
    elif "uefa" in nq or "champions" in nq or "europa league" in nq or "conference" in nq:
        targeted.append(f'site:uefa.com "{home}" "{away}" referee VAR')
    elif "primera division de chile" in nq or "copa chile" in nq:
        targeted.append(f'site:anfp.cl "{home}" "{away}" árbitro')
    elif "liga profesional" in nq or "copa argentina" in nq:
        targeted.append(f'site:afa.com.ar "{home}" "{away}" árbitro')

    queries = targeted + [
        f'"{home}" "{away}" "{competition}" árbitro VAR {date_key}',
        f'"{home}" "{away}" referee VAR {date_key}',
    ]

    batches = await asyncio.gather(*(web_search(q, 8) for q in queries))
    merged = []
    seen_urls = set()
    for batch in batches:
        for r in batch:
            if r.url in seen_urls:
                continue
            seen_urls.add(r.url)
            merged.append(r)

    cleaned_results = []
    seen_clean = set()
    for r in merged:
        r.url = clean_search_url(r.url)
        if r.url in seen_clean:
            continue
        if not is_trusted_context_source(r.url):
            continue
        seen_clean.add(r.url)
        cleaned_results.append(r)
    merged = cleaned_results

    # Prioridad: oficial -> estadística deportiva -> prensa confiable.
    priority = {"Oficial": 0, "Estadística deportiva": 1, "Medio confiable": 2}
    merged.sort(key=lambda x: priority.get(source_trust_type(x.url) or "", 9))
    referee = var = None
    sources = []

    # Snippets primero.
    for r in merged[:15]:
        rr, vv = parse_referee_var_from_text(f"{r.title} {r.snippet}")
        if rr or vv:
            sources.append({
                "titulo": r.title,
                "url": r.url,
                "tipo": source_trust_type(r.url) or "Fuente deportiva",
            })
            referee = referee or rr
            var = var or vv
        if referee and var:
            break

    # Si falta algo, leer solamente páginas de fuentes autorizadas.
    if not (referee and var):
        for r in merged[:12]:
            try:
                raw = await fetch_html(r.url, CONTEXT_TTL)
                visible = text_of(BeautifulSoup(raw, "html.parser"))[:180_000]
                rr, vv = parse_referee_var_from_text(visible)
            except Exception:
                continue

            if rr or vv:
                if not any(s.get("url") == r.url for s in sources):
                    sources.append({
                        "titulo": r.title,
                        "url": r.url,
                        "tipo": source_trust_type(r.url) or "Fuente deportiva",
                    })
                referee = referee or rr
                var = var or vv
            if referee and var:
                break

    out = {"referee": referee, "var": var, "sources": sources[:5]}
    cache_set(key, out)
    return out


def _valid_odds_trio(values: list[float]) -> tuple[float, float, float] | None:
    if len(values) < 3:
        return None
    trio = values[:3]
    if all(1.01 <= x <= 30.0 for x in trio):
        return trio[0], trio[1], trio[2]
    return None

def parse_decimal_odds_from_table(raw: str) -> tuple[float, float, float] | None:
    soup = BeautifulSoup(raw, "html.parser")
    for table in soup.find_all("table"):
        headers = [text_of(x).strip().lower() for x in table.find_all("th")]
        h = " ".join(headers)
        if not (re.search(r"\b1\b", h) and re.search(r"\bx\b", h) and re.search(r"\b2\b", h)):
            continue
        for row in table.find_all("tr"):
            vals = []
            for td in row.find_all("td"):
                txt = text_of(td).replace(",", ".")
                for m in re.findall(r"(?<!\d)(\d{1,2}\.\d{1,2})(?!\d)", txt):
                    try:
                        vals.append(float(m))
                    except Exception:
                        pass
            trio = _valid_odds_trio(vals)
            if trio:
                return trio
    return None

def _team_text_variants(name: str) -> list[str]:
    name = re.sub(r"\s+", " ", name or "").strip()
    variants = [name]
    stripped = re.sub(r"^(?:CA|CF|FC|CD|Club|Independiente)\s+", "", name, flags=re.I).strip()
    if stripped and stripped not in variants:
        variants.append(stripped)
    parts = name.split()
    if len(parts) >= 3:
        short = " ".join(parts[-2:])
        if short not in variants:
            variants.append(short)
    return variants

def parse_main_1x2_market(text: str, home: str, away: str) -> tuple[float, float, float] | None:
    text = re.sub(r"\s+", " ", html.unescape(text or ""))
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    anchors = ["Resultado del Partido", "Match Result", "Full Time Result", "1X2"]
    windows = []
    low = text.lower()
    for anchor in anchors:
        pos = low.find(anchor.lower())
        if pos >= 0:
            windows.append(text[pos:pos + 3500])
    if not windows:
        windows = [text[:12000]]

    def price_after(window: str, labels: list[str]) -> float | None:
        for label in labels:
            m = re.search(rf"{re.escape(label)}.{{0,140}}?(\d{{1,2}}\.\d{{2,3}})", window, re.I)
            if m:
                try:
                    val = float(m.group(1))
                    if 1.01 <= val <= 30:
                        return val
                except Exception:
                    pass
        return None

    for window in windows:
        oh = price_after(window, _team_text_variants(home))
        od = price_after(window, ["Empate", "Draw", "X"])
        oa = price_after(window, _team_text_variants(away))
        if oh is not None and od is not None and oa is not None:
            return _valid_odds_trio([oh, od, oa])
    return None


def parse_decimal_odds_from_text(text: str, home: str, away: str) -> tuple[float, float, float] | None:
    text = re.sub(r"\s+", " ", html.unescape(text or ""))
    # Normalizar sólo comas decimales.
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)

    for hv in _team_text_variants(home):
        for av in _team_text_variants(away):
            hp = re.escape(hv)
            ap = re.escape(av)
            patterns = [
                rf"{hp}.{{0,90}}?(\d{{1,2}}\.\d{{1,2}}).{{0,120}}?(?:Empate|Draw|\bX\b).{{0,50}}?(\d{{1,2}}\.\d{{1,2}}).{{0,120}}?{ap}.{{0,90}}?(\d{{1,2}}\.\d{{1,2}})",
                rf"(?:1x2|Ganador|Win Market|Resultado del Partido).{{0,180}}?{hp}.{{0,80}}?(\d{{1,2}}\.\d{{1,2}}).{{0,120}}?(?:Empate|Draw|\bX\b).{{0,50}}?(\d{{1,2}}\.\d{{1,2}}).{{0,120}}?{ap}.{{0,80}}?(\d{{1,2}}\.\d{{1,2}})",
            ]
            for p in patterns:
                m = re.search(p, text, re.I)
                if not m:
                    continue
                try:
                    trio = [float(m.group(1)), float(m.group(2)), float(m.group(3))]
                except Exception:
                    continue
                valid = _valid_odds_trio(trio)
                if valid:
                    return valid
    return None

async def discover_odds(home: str, away: str) -> dict[str, Any]:
    key = f"odds:{norm_text(home)}:{norm_text(away)}"
    cached = cache_get(key, CONTEXT_TTL)
    if cached is not None:
        return cached

    queries = [
        f'site:sports.caliente.mx "{home}" "{away}" "Resultado del Partido"',
        f'site:stake.bet.ar "{home}" "{away}" "Resultado del Partido"',
        f'site:oddschecker.com "{home}" "{away}" odds',
        f'site:sportingbet.com "{home}" "{away}"',
        f'site:betexplorer.com "{home}" "{away}" odds',
        f'site:oddsportal.com "{home}" "{away}" odds',
    ]
    batches = await asyncio.gather(*(web_search(q, 5) for q in queries))
    all_results = []
    seen = set()
    for batch in batches:
        for r in batch:
            if r.url in seen:
                continue
            seen.add(r.url)
            all_results.append(r)

    allowed = ["oddschecker", "sportingbet", "stake.bet", "caliente", "betexplorer", "oddsportal"]
    candidates = []
    seen_candidate_urls = set()
    for r in all_results:
        r.url = clean_search_url(r.url)
        host = (urlparse(r.url).netloc or "").lower()
        if not any(x in host for x in allowed) or r.url in seen_candidate_urls:
            continue
        seen_candidate_urls.add(r.url)
        candidates.append(r)

    # 1) Snippets/resultados de búsqueda.
    for r in candidates:
        snippet_text = f"{r.title} {r.snippet}"
        trio = parse_main_1x2_market(snippet_text, home, away) or parse_decimal_odds_from_text(snippet_text, home, away)
        if trio:
            oh, od, oa = trio
            ph, pd, pa = 1 / oh, 1 / od, 1 / oa
            s = ph + pd + pa
            out = {
                "raw": [oh, od, oa],
                "prob": [ph / s * 100, pd / s * 100, pa / s * 100],
                "source_url": r.url,
                "source_title": r.title,
            }
            cache_set(key, out)
            return out

    # 2) Leer hasta 8 páginas públicas en paralelo.
    page_candidates = candidates[:8]
    payloads = await asyncio.gather(
        *(fetch_html(r.url, CONTEXT_TTL) for r in page_candidates),
        return_exceptions=True,
    )
    for r, raw in zip(page_candidates, payloads):
        if not isinstance(raw, str):
            continue
        visible = text_of(BeautifulSoup(raw, "html.parser"))[:300_000]
        trio = parse_main_1x2_market(visible, home, away)
        if not trio:
            trio = parse_decimal_odds_from_table(raw)
        if not trio:
            trio = parse_decimal_odds_from_text(visible, home, away)
        if trio:
            oh, od, oa = trio
            ph, pd, pa = 1 / oh, 1 / od, 1 / oa
            s = ph + pd + pa
            out = {
                "raw": [oh, od, oa],
                "prob": [ph / s * 100, pd / s * 100, pa / s * 100],
                "source_url": r.url,
                "source_title": r.title,
            }
            cache_set(key, out)
            return out

    out = {"raw": None, "prob": None, "source_url": None, "source_title": None}
    cache_set(key, out)
    return out


def is_continental_competition(name: str) -> bool:
    n = norm_text(name)
    return any(token in n for token in [
        "conmebol sudamericana",
        "conmebol libertadores",
        "uefa champions league",
        "uefa europa league",
        "uefa conference league",
        "club world cup",
        "mundial de clubes",
    ])


def table_score(standing: dict[str, Any]) -> float:
    rank, total = standing.get("rank"), standing.get("total")
    if rank and total and total > 1:
        return 1 - 2 * ((rank - 1) / (total - 1))
    played, points = standing.get("played"), standing.get("points")
    if played and points is not None:
        return clamp((points / played - 1.4) / 1.4, -1, 1)
    return 0.0


def injury_penalty(count: int | None) -> float:
    if count is None:
        return 0.0
    return 0.035 * min(count, 6)


def metric(
    selection: str,
    probability: float,
    quality: float,
    maximum_confidence: str = "Alta",
    reason: str | None = None,
    show_confidence: bool = True,
) -> dict[str, Any]:
    computed = confidence(probability, quality)
    global_cap = global_confidence_cap(quality)
    final = cap_confidence(cap_confidence(computed, global_cap), maximum_confidence)
    return {
        "seleccion": selection,
        "probabilidad": round(probability),
        "confianza": final if show_confidence else None,
        "mostrar_confianza": show_confidence,
        "confianza_calculada": computed,
        "confianza_limitada": show_confidence and final != computed,
        "motivo_confianza": reason if show_confidence and final != computed else None,
    }


def source_item(title: str, url: str | None, tipo: str) -> dict[str, str] | None:
    if not url:
        return None
    url = clean_search_url(url)
    host = (urlparse(url).netloc or "").lower()
    if not host or is_blocked_source(url):
        return None
    # Evitar mostrar links de buscadores como si fueran fuentes.
    if _host_matches(host, "bing.com") or _host_matches(host, "duckduckgo.com"):
        return None
    return {"titulo": title, "url": url, "tipo": tipo}


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "version": APP_VERSION,
        "hora_chile": now_chile().strftime("%Y-%m-%d %H:%M"),
        "buscador": "ESPN-web-publica",
        "version_motor": "4.7",
        "api_deportiva": False,
        "openai_api": False,
        "requiere_claves": False,
    }


@app.get("/api/search")
async def search(q: str = Query(min_length=2, max_length=80)):
    teams = await discover_teams(q.strip())
    if not teams:
        raise HTTPException(
            status_code=502,
            detail="No pude localizar el club. El buscador web público no respondió; inténtalo nuevamente en unos segundos.",
        )
    return {"response": teams, "modo": "web-directa"}


@app.get("/api/team/{team_id}/next")
async def analyze_next(team_id: int, name: str = Query(default="", max_length=100)):
    key = f"analysis:{team_id}"
    cached = cache_get(key, ANALYSIS_TTL)
    if cached is not None:
        result = dict(cached)
        result["cache"] = True
        return result

    try:
        fixture = await get_next_fixture(team_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    home = fixture["home"]
    away = fixture["away"]
    home_id, away_id = int(home["id"]), int(away["id"])
    home_name, away_name = home["name"], away["name"]
    dt: datetime | None = fixture.get("dt")

    # Lecturas principales en paralelo, todas desde páginas web públicas.
    home_results_c, away_results_c, home_stand_c, away_stand_c = await asyncio.gather(
        get_results(home_id),
        get_results(away_id),
        get_standing(home_id, home_name),
        get_standing(away_id, away_name),
        return_exceptions=True,
    )

    home_results = home_results_c if isinstance(home_results_c, list) else []
    away_results = away_results_c if isinstance(away_results_c, list) else []
    home_stand = home_stand_c if isinstance(home_stand_c, dict) else {"rank": None, "total": None}
    away_stand = away_stand_c if isinstance(away_stand_c, dict) else {"rank": None, "total": None}

    h10 = summarize_results(home_results, home_id, n=10)
    a10 = summarize_results(away_results, away_id, n=10)
    h5 = summarize_results(home_results, home_id, n=5)
    a5 = summarize_results(away_results, away_id, n=5)
    hvenue = summarize_results(home_results, home_id, venue="home", n=8)
    avenue = summarize_results(away_results, away_id, venue="away", n=8)

    home_played_hint = home_stand.get("played") or h10.get("played")
    away_played_hint = away_stand.get("played") or a10.get("played")

    home_disc_c, away_disc_c, home_inj_c, away_inj_c, context_c, odds_c = await asyncio.gather(
        get_discipline(home_id, home_played_hint),
        get_discipline(away_id, away_played_hint),
        discover_absences(home_name, dt),
        discover_absences(away_name, dt),
        discover_match_context(home_name, away_name, fixture.get("competition") or "", dt),
        discover_odds(home_name, away_name),
        return_exceptions=True,
    )

    home_disc = home_disc_c if isinstance(home_disc_c, dict) else {"yellow_pg": None, "red_pg": None}
    away_disc = away_disc_c if isinstance(away_disc_c, dict) else {"yellow_pg": None, "red_pg": None}
    home_inj = home_inj_c if isinstance(home_inj_c, dict) else {"count": None, "players": [], "source_url": None}
    away_inj = away_inj_c if isinstance(away_inj_c, dict) else {"count": None, "players": [], "source_url": None}
    context = context_c if isinstance(context_c, dict) else {"referee": None, "var": None, "sources": []}
    odds = odds_c if isinstance(odds_c, dict) else {"prob": None, "source_url": None, "raw": None}

    try:
        referee_stats = await discover_referee_stats(context.get("referee"))
    except Exception:
        referee_stats = {"yellow_pg": None, "red_pg": None, "source_url": None}

    # Fuerza actual. Prestigio histórico = 0%.
    # En competiciones continentales no se comparan directamente posiciones
    # de ligas nacionales diferentes: el peso de tabla baja a 3%.
    continental = is_continental_competition(fixture.get("competition") or "")
    table_weight = 0.03 if continental else 0.11
    recent10_weight = 0.32 if continental else 0.28
    recent5_weight = 0.23 if continental else 0.20
    venue_weight = 0.16 if continental else 0.15

    h_strength = (
        recent10_weight * h10["ppg"]
        + recent5_weight * h5["ppg"]
        + 0.16 * (h10["gf"] - h10["ga"] + 1.4)
        + venue_weight * hvenue["ppg"]
        + table_weight * (table_score(home_stand) + 1)
        + 0.18
        - injury_penalty(home_inj.get("count"))
    )
    a_strength = (
        recent10_weight * a10["ppg"]
        + recent5_weight * a5["ppg"]
        + 0.16 * (a10["gf"] - a10["ga"] + 1.4)
        + venue_weight * avenue["ppg"]
        + table_weight * (table_score(away_stand) + 1)
        - injury_penalty(away_inj.get("count"))
    )

    z = h_strength - a_strength
    mh = 100 / (1 + math.exp(-1.08 * z))
    md = clamp(28.5 - abs(z) * 5.0, 18, 31)
    ma = 100 - mh
    mh, md, ma = normalize3(mh * (100 - md) / 100, md, ma * (100 - md) / 100)

    if odds.get("prob"):
        oh, od, oa = odds["prob"]
        # Las cuotas son factor actual, no autoridad absoluta.
        fh = 0.78 * mh + 0.22 * oh
        fd = 0.78 * md + 0.22 * od
        fa = 0.78 * ma + 0.22 * oa
        fh, fd, fa = normalize3(fh, fd, fa)
    else:
        fh, fd, fa = mh, md, ma

    winner = max({"1": fh, "X": fd, "2": fa}.items(), key=lambda x: x[1])
    double = max({"1X": fh + fd, "X2": fd + fa, "12": fh + fa}.items(), key=lambda x: x[1])

    # Goles: forma reciente + rendimiento local/visita.
    lambda_home = clamp(
        0.52 * ((h10["gf"] + a10["ga"]) / 2)
        + 0.48 * ((hvenue["gf"] + avenue["ga"]) / 2)
        + 0.10,
        0.25,
        3.5,
    )
    lambda_away = clamp(
        0.52 * ((a10["gf"] + h10["ga"]) / 2)
        + 0.48 * ((avenue["gf"] + hvenue["ga"]) / 2),
        0.20,
        3.5,
    )
    lambda_total = lambda_home + lambda_away
    btts_yes = (1 - math.exp(-lambda_home)) * (1 - math.exp(-lambda_away))
    if btts_yes >= 0.5:
        btts_sel, btts_prob = "Sí", btts_yes * 100
    else:
        btts_sel, btts_prob = "No", (1 - btts_yes) * 100
    goals_sel, goals_prob = best_line(lambda_total, [1.5, 2.5, 3.5])

    # Disciplina. Baseline prudente cuando falta una fuente.
    hy = home_disc.get("yellow_pg") if home_disc.get("yellow_pg") is not None else 2.1
    ay = away_disc.get("yellow_pg") if away_disc.get("yellow_pg") is not None else 2.1
    hr = home_disc.get("red_pg") if home_disc.get("red_pg") is not None else 0.08
    ar = away_disc.get("red_pg") if away_disc.get("red_pg") is not None else 0.08

    team_yellow_lambda = clamp(hy + ay, 1.5, 9.0)
    team_red_lambda = clamp(hr + ar, 0.03, 1.4)

    ref_y = referee_stats.get("yellow_pg")
    ref_r = referee_stats.get("red_pg")
    yellow_lambda = 0.72 * team_yellow_lambda + 0.28 * ref_y if ref_y is not None else team_yellow_lambda
    red_lambda = 0.78 * team_red_lambda + 0.22 * ref_r if ref_r is not None else team_red_lambda

    yellow_sel, yellow_prob = best_line(clamp(yellow_lambda, 1.5, 9.0), [2.5, 3.5, 4.5, 5.5])
    red_sel, red_prob = best_line(clamp(red_lambda, 0.03, 1.4), [0.5, 1.5])

    quality = 20  # próximo partido identificado
    if len(home_results) >= 5 and len(away_results) >= 5:
        quality += 28
    elif home_results or away_results:
        quality += 14
    if home_stand.get("rank") is not None and away_stand.get("rank") is not None:
        quality += 14
    if home_disc.get("yellow_pg") is not None and away_disc.get("yellow_pg") is not None:
        quality += 10
    if home_inj.get("count") is not None and away_inj.get("count") is not None:
        quality += 10
    if context.get("referee"):
        quality += 5
    if context.get("var"):
        quality += 2
    if referee_stats.get("yellow_pg") is not None:
        quality += 3
    if referee_stats.get("red_pg") is not None:
        quality += 2
    if odds.get("prob"):
        quality += 10
    quality = int(clamp(quality, 0, 100))

    # Confiabilidad efectiva: el análisis SIEMPRE se entrega, pero la puntuación
    # baja explícitamente cuando faltan datos críticos.
    reliability = quality
    missing_critical = []

    # Faltantes reducen la confiabilidad de forma moderada; nunca bloquean el análisis.
    if not odds.get("prob"):
        reliability -= 3
        missing_critical.append("cuotas 1X2")

    if not context.get("referee"):
        reliability -= 3
        missing_critical.append("árbitro")
    elif referee_stats.get("yellow_pg") is None:
        reliability -= 2
        missing_critical.append("estadísticas del árbitro")

    if not context.get("var"):
        reliability -= 1
        missing_critical.append("VAR")

    if home_inj.get("count") is None:
        reliability -= 2
        missing_critical.append(f"bajas {home_name}")
    if away_inj.get("count") is None:
        reliability -= 2
        missing_critical.append(f"bajas {away_name}")

    if home_disc.get("yellow_pg") is None or away_disc.get("yellow_pg") is None:
        reliability -= 4
        missing_critical.append("estadísticas de tarjetas")

    if len(home_results) < 5 or len(away_results) < 5:
        reliability -= 7
        missing_critical.append("forma reciente")

    reliability = int(clamp(reliability, 0, 100))
    reliability_label = reliability_level(reliability)

    sources = [
        source_item("Próximo partido — ESPN", fixture.get("source_url"), "Partido"),
        source_item(f"Resultados recientes — {home_name}", f"https://www.espn.com/soccer/team/results/_/id/{home_id}", "Forma"),
        source_item(f"Resultados recientes — {away_name}", f"https://www.espn.com/soccer/team/results/_/id/{away_id}", "Forma"),
        source_item(f"Tabla/estadísticas — {home_name}", home_stand.get("source_url"), "Tabla"),
        source_item(f"Tabla/estadísticas — {away_name}", away_stand.get("source_url"), "Tabla"),
        source_item(f"Disciplina — {home_name}", home_disc.get("source_url"), "Tarjetas"),
        source_item(f"Disciplina — {away_name}", away_disc.get("source_url"), "Tarjetas"),
        source_item(f"Bajas — {home_name}", home_inj.get("source_url"), "Bajas"),
        source_item(f"Bajas — {away_name}", away_inj.get("source_url"), "Bajas"),
        source_item("Cuotas 1X2", odds.get("source_url"), "Mercado"),
        source_item(f"Estadísticas del árbitro — {context.get('referee') or 'no confirmado'}", referee_stats.get("source_url"), "Árbitro"),
    ]
    sources.extend(context.get("sources") or [])
    clean_sources = []
    seen_urls = set()
    for s in sources:
        if not s or not s.get("url") or s["url"] in seen_urls:
            continue
        seen_urls.add(s["url"])
        clean_sources.append(s)
        if len(clean_sources) >= 10:
            break

    if dt:
        fecha = dt.strftime("%d/%m/%Y")
        hora = dt.strftime("%H:%M")
        iso = dt.isoformat()
    else:
        fecha = fixture.get("date_text") or "Por confirmar"
        hora = "Por confirmar"
        iso = None

    warnings = []
    if missing_critical:
        warnings.append(
            "Faltan algunos datos; el análisis se realizó igualmente con una confiabilidad ligeramente menor."
        )

    # Topes de confianza según datos críticos de cada mercado.
    winner_cap = "Alta"
    winner_reason_parts = []
    missing_odds = not odds.get("prob")
    missing_injuries = home_inj.get("count") is None or away_inj.get("count") is None

    if missing_odds:
        winner_cap = "Media"
        winner_reason_parts.append("sin cuotas 1X2")
    if missing_injuries:
        winner_cap = "Media"
        winner_reason_parts.append("sin bajas confirmadas")
    if len(home_results) < 5 or len(away_results) < 5:
        winner_cap = "Baja"
        winner_reason_parts.append("forma reciente insuficiente")

    goals_cap = "Alta" if len(home_results) >= 5 and len(away_results) >= 5 else "Baja"
    goals_reason = None if goals_cap == "Alta" else "forma reciente insuficiente"

    cards_cap = "Alta"
    cards_reason_parts = []
    if home_disc.get("yellow_pg") is None or away_disc.get("yellow_pg") is None:
        cards_cap = "Baja"
        cards_reason_parts.append("sin estadísticas de tarjetas")
    elif not context.get("referee") or referee_stats.get("yellow_pg") is None:
        cards_cap = "Media"
        cards_reason_parts.append("sin datos arbitrales completos")

    reds_cap = cards_cap
    if cards_cap == "Alta" and referee_stats.get("red_pg") is None:
        reds_cap = "Media"
        cards_reason_parts.append("sin promedio de rojas del árbitro")

    # La probabilidad de tarjetas se sigue mostrando, pero si falta Árbitro o VAR
    # no se publica ningún color/nivel de confianza para esos mercados.
    show_cards_confidence = bool(context.get("referee") and context.get("var"))

    result = {
        "local": home_name,
        "visita": away_name,
        "fecha_chile": fecha,
        "hora_chile": hora,
        "fecha_hora_chile": iso,
        "competicion": fixture.get("competition") or "Competición no identificada",
        "ganador": metric(winner[0], winner[1], reliability, winner_cap, ", ".join(winner_reason_parts) or None),
        "doble_oportunidad": metric(double[0], double[1], reliability, winner_cap, ", ".join(winner_reason_parts) or None),
        "ambos_marcan": metric(btts_sel, btts_prob, reliability, goals_cap, goals_reason),
        "tarjetas_amarillas": metric(yellow_sel, yellow_prob, reliability, cards_cap, ", ".join(cards_reason_parts) or None, show_cards_confidence),
        "tarjetas_rojas": metric(red_sel, red_prob, reliability, reds_cap, ", ".join(cards_reason_parts) or None, show_cards_confidence),
        "goles_totales": metric(goals_sel, goals_prob, reliability, goals_cap, goals_reason),
        "forma_local_5": h5["form"],
        "forma_visita_5": a5["form"],
        "forma_local_10": h10["form"],
        "forma_visita_10": a10["form"],
        "tabla_local": home_stand.get("rank"),
        "tabla_visita": away_stand.get("rank"),
        "bajas_local": home_inj.get("count"),
        "bajas_visita": away_inj.get("count"),
        "bajas_local_nombres": home_inj.get("players") or [],
        "bajas_visita_nombres": away_inj.get("players") or [],
        "arbitro": context.get("referee"),
        "var": context.get("var"),
        "arbitro_amarillas_pg": referee_stats.get("yellow_pg"),
        "arbitro_rojas_pg": referee_stats.get("red_pg"),
        "cuotas_1x2": odds.get("raw"),
        "confiabilidad_analisis": reliability,
        "nivel_confiabilidad_analisis": reliability_label,
        "datos_no_encontrados": missing_critical,
        "actualizado_chile": now_chile().strftime("%d/%m/%Y %H:%M"),
        "fuentes": clean_sources,
        "cache": False,
    }
    cache_set(key, result)
    return result


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
