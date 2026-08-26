from __future__ import annotations

import asyncio
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
APP_VERSION = "4.2-web-directa"
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
    "RadarFutbol/4.0"
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


def confidence(probability: float, quality: float) -> str:
    score = 0.60 * quality + 0.40 * abs(probability - 50) * 2
    if score >= 76:
        return "Alta"
    if score >= 56:
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
        out.append(
            SearchResult(
                title=" ".join(a.stripped_strings),
                url=a["href"],
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
        clean = item.url.split("#")[0]
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
        return {"count": len(players), "players": players[:8], "source_url": target.url}
    except Exception:
        return {"count": None, "players": [], "source_url": target.url}


def parse_referee_var_from_text(text: str) -> tuple[str | None, str | None]:
    text = re.sub(r"\s+", " ", text)
    referee = None
    var = None
    patterns_ref = [
        r"(?:Árbitro|Arbitro|Referee|Árbitro principal)\s*[:\-]\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ.' -]{3,45})",
        r"(?:será arbitrado por|will be refereed by)\s+([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ.' -]{3,45})",
    ]
    for p in patterns_ref:
        m = re.search(p, text, re.I)
        if m:
            referee = re.split(r"\b(?:VAR|assistant|asistente|y el|with)\b|[.;]", m.group(1), maxsplit=1, flags=re.I)[0].strip(" -")
            break
    m = re.search(r"\bVAR\s*[:\-]\s*([A-ZÁÉÍÓÚÑ][A-Za-zÁÉÍÓÚÑáéíóúñ.' -]{3,45})", text, re.I)
    if m:
        var = re.split(r"[.;]", m.group(1), maxsplit=1)[0].strip(" -")
    return referee, var


OFFICIAL_DOMAIN_HINTS = [
    "conmebol.com", "uefa.com", "fifa.com", "anfp.cl", "afa.com.ar",
    "premierleague.com", "laliga.com", "bundesliga.com", "legaseriea.it",
    "ligue1.com", "cbf.com.br", "auf.org.uy", "dimayor.com.co", "ligamx.net",
]

def is_official_domain(url: str) -> bool:
    host = (urlparse(url).netloc or "").lower()
    return any(h in host for h in OFFICIAL_DOMAIN_HINTS)

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

    official_q = f'"{home}" "{away}" {competition} {date_key} sitio oficial'
    generic_q = f'"{home}" "{away}" árbitro VAR {competition} {date_key}'
    official_results, generic_results = await asyncio.gather(
        web_search(official_q, 8), web_search(generic_q, 8)
    )
    merged = []
    seen_urls = set()
    for r in sorted(official_results + generic_results, key=lambda x: (0 if is_official_domain(x.url) else 1)):
        if r.url in seen_urls:
            continue
        seen_urls.add(r.url)
        merged.append(r)
    results = merged
    referee = var = None
    sources = []
    for r in results[:6]:
        sources.append({"titulo": r.title, "url": r.url, "tipo": "Oficial" if is_official_domain(r.url) else "Árbitro/VAR"})
        combined = f"{r.title} {r.snippet}"
        rr, vv = parse_referee_var_from_text(combined)
        referee = referee or rr
        var = var or vv
        if referee and var:
            break
        # Leemos sólo páginas públicas que respondan normalmente.
        try:
            raw = await fetch_html(r.url, CONTEXT_TTL)
            visible = text_of(BeautifulSoup(raw, "html.parser"))[:120_000]
            rr, vv = parse_referee_var_from_text(visible)
            referee = referee or rr
            var = var or vv
        except Exception:
            continue
        if referee and var:
            break

    out = {"referee": referee, "var": var, "sources": sources[:4]}
    cache_set(key, out)
    return out


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
                for m in re.findall(r"(?<!\d)(1\.\d{2}|[2-9]\.\d{2}|1\d\.\d{2})(?!\d)", txt):
                    try:
                        vals.append(float(m))
                    except Exception:
                        pass
            if len(vals) >= 3:
                trio = vals[:3]
                if all(1.01 <= x <= 25 for x in trio):
                    return trio[0], trio[1], trio[2]
    return None


async def discover_odds(home: str, away: str) -> dict[str, Any]:
    queries = [
        f'site:betexplorer.com "{home}" "{away}" odds',
        f'site:oddsportal.com "{home}" "{away}" odds',
    ]
    all_results: list[SearchResult] = []
    for q in queries:
        all_results.extend(await web_search(q, 5))
    seen = set()
    for r in all_results:
        if r.url in seen:
            continue
        seen.add(r.url)
        host = urlparse(r.url).netloc.lower()
        if not any(x in host for x in ["betexplorer", "oddsportal"]):
            continue
        try:
            raw = await fetch_html(r.url, CONTEXT_TTL)
            odds = parse_decimal_odds_from_table(raw)
            if odds:
                oh, od, oa = odds
                ph, pd, pa = 1 / oh, 1 / od, 1 / oa
                s = ph + pd + pa
                return {
                    "raw": [oh, od, oa],
                    "prob": [ph / s * 100, pd / s * 100, pa / s * 100],
                    "source_url": r.url,
                }
        except Exception:
            continue
    return {"raw": None, "prob": None, "source_url": None}


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


def metric(selection: str, probability: float, quality: float) -> dict[str, Any]:
    return {
        "seleccion": selection,
        "probabilidad": round(probability),
        "confianza": confidence(probability, quality),
    }


def source_item(title: str, url: str | None, tipo: str) -> dict[str, str] | None:
    if not url:
        return None
    return {"titulo": title, "url": url, "tipo": tipo}


@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "version": APP_VERSION,
        "hora_chile": now_chile().strftime("%Y-%m-%d %H:%M"),
        "modo": "web-directa",
        "buscador": "ESPN-web-publica",
        "version_motor": "4.2",
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
        discover_transfermarkt_injuries(home_name, dt),
        discover_transfermarkt_injuries(away_name, dt),
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

    # Fuerza actual. Prestigio histórico = 0%.
    h_strength = (
        0.28 * h10["ppg"]
        + 0.20 * h5["ppg"]
        + 0.16 * (h10["gf"] - h10["ga"] + 1.4)
        + 0.15 * hvenue["ppg"]
        + 0.11 * (table_score(home_stand) + 1)
        + 0.18  # localía actual
        - injury_penalty(home_inj.get("count"))
    )
    a_strength = (
        0.28 * a10["ppg"]
        + 0.20 * a5["ppg"]
        + 0.16 * (a10["gf"] - a10["ga"] + 1.4)
        + 0.15 * avenue["ppg"]
        + 0.11 * (table_score(away_stand) + 1)
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

    # Disciplina. Si una fuente no entrega datos, usamos un baseline prudente y reducimos calidad.
    hy = home_disc.get("yellow_pg") if home_disc.get("yellow_pg") is not None else 2.1
    ay = away_disc.get("yellow_pg") if away_disc.get("yellow_pg") is not None else 2.1
    hr = home_disc.get("red_pg") if home_disc.get("red_pg") is not None else 0.08
    ar = away_disc.get("red_pg") if away_disc.get("red_pg") is not None else 0.08
    yellow_sel, yellow_prob = best_line(clamp(hy + ay, 1.5, 9.0), [2.5, 3.5, 4.5, 5.5])
    red_sel, red_prob = best_line(clamp(hr + ar, 0.03, 1.4), [0.5, 1.5])

    quality = 20  # próximo partido identificado
    if len(home_results) >= 5 and len(away_results) >= 5:
        quality += 30
    elif home_results or away_results:
        quality += 15
    if home_stand.get("rank") is not None and away_stand.get("rank") is not None:
        quality += 15
    if home_disc.get("yellow_pg") is not None and away_disc.get("yellow_pg") is not None:
        quality += 10
    if home_inj.get("count") is not None and away_inj.get("count") is not None:
        quality += 10
    if context.get("referee"):
        quality += 5
    if odds.get("prob"):
        quality += 10
    quality = int(clamp(quality, 0, 100))

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
    if not odds.get("prob"):
        warnings.append("No se encontraron cuotas 1X2 públicas y estructuradas; no se usaron en el cálculo.")
    if not context.get("referee"):
        warnings.append("Árbitro/VAR todavía no confirmados en una fuente web legible.")
    if home_inj.get("count") is None or away_inj.get("count") is None:
        warnings.append("Alguna fuente de bajas no respondió o no entregó una tabla legible.")

    result = {
        "modo": "web-directa",
        "local": home_name,
        "visita": away_name,
        "fecha_chile": fecha,
        "hora_chile": hora,
        "fecha_hora_chile": iso,
        "competicion": fixture.get("competition") or "Competición no identificada",
        "ganador": metric(winner[0], winner[1], quality),
        "doble_oportunidad": metric(double[0], double[1], quality),
        "ambos_marcan": metric(btts_sel, btts_prob, quality),
        "tarjetas_amarillas": metric(yellow_sel, yellow_prob, quality),
        "tarjetas_rojas": metric(red_sel, red_prob, quality),
        "goles_totales": metric(goals_sel, goals_prob, quality),
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
        "cuotas_1x2": odds.get("raw"),
        "calidad_datos": quality,
        "actualizado_chile": now_chile().strftime("%d/%m/%Y %H:%M"),
        "fuentes": clean_sources,
        "advertencias": warnings,
        "cache": False,
        "prestigio_historico_peso": 0,
    }
    cache_set(key, result)
    return result


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
