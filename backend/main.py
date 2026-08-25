from __future__ import annotations

import asyncio
import json
import math
import os
import re
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_NAME = "Radar Fútbol"
API_BASE = "https://v3.football.api-sports.io"
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6").strip()

TZ = ZoneInfo("America/Santiago")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT, "frontend")

app = FastAPI(title=APP_NAME, version="3.0-optimizada")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

# CACHÉ EN MEMORIA
# - búsqueda de clubes: 15 minutos
# - análisis de un equipo/partido: 10 minutos
CACHE: dict[str, tuple[float, Any]] = {}
SEARCH_TTL = 15 * 60
ANALYSIS_TTL = 10 * 60

def cache_read(key: str, ttl: int):
    item = CACHE.get(key)
    if not item:
        return None
    ts, value = item
    if time.time() - ts > ttl:
        CACHE.pop(key, None)
        return None
    return value

def cache_write(key: str, value: Any):
    CACHE[key] = (time.time(), value)

def now_chile() -> datetime:
    return datetime.now(tz=TZ)

def to_chile(dt_str: str) -> tuple[str, str]:
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00")).astimezone(TZ)
    return dt.isoformat(), dt.strftime("%H:%M")

async def api_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    if not API_SPORTS_KEY:
        raise HTTPException(status_code=503, detail="Falta API_SPORTS_KEY.")

    headers = {"x-apisports-key": API_SPORTS_KEY}
    async with httpx.AsyncClient(timeout=20.0) as client:
        response = await client.get(f"{API_BASE}/{path}", params=params, headers=headers)

    try:
        data = response.json()
    except Exception:
        raise HTTPException(status_code=502, detail="La API deportiva devolvió una respuesta inválida.")

    if response.status_code >= 400 or data.get("errors"):
        raise HTTPException(
            status_code=502,
            detail=f"API deportiva: {data.get('errors') or response.status_code}"
        )
    return data

def pct(value: Any, default: float) -> float:
    try:
        return float(str(value).replace("%", "").replace(",", "."))
    except Exception:
        return default

def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def normalize3(a: float, b: float, c: float) -> tuple[float, float, float]:
    total = a + b + c
    if total <= 0:
        return (33.34, 33.33, 33.33)
    return (a / total * 100, b / total * 100, c / total * 100)

def poisson_cdf(k: int, lam: float) -> float:
    lam = max(0.01, lam)
    return sum(math.exp(-lam) * (lam ** i) / math.factorial(i) for i in range(k + 1))

def best_line(lam: float, lines: list[float]) -> tuple[str, int]:
    options = []
    for line in lines:
        options.append(("+", line, 1 - poisson_cdf(math.floor(line), lam)))
        options.append(("-", line, poisson_cdf(math.floor(line), lam)))
    sign, line, probability = max(options, key=lambda item: item[2])
    return f"{sign}{str(line).replace('.', ',')}", round(probability * 100)

def confidence(probability: float, quality: float) -> str:
    score = 0.55 * quality + 0.45 * abs(probability - 50) * 2
    if score >= 75:
        return "Alta"
    if score >= 55:
        return "Media"
    return "Baja"

def form_spanish(raw: str) -> str:
    return (raw or "").upper().replace("W", "G").replace("D", "E").replace("L", "P")

def team_stats(raw: dict[str, Any]) -> dict[str, float | str]:
    response = (raw or {}).get("response") or {}
    fixtures = response.get("fixtures") or {}

    played = float((fixtures.get("played") or {}).get("total") or 0)
    wins = float((fixtures.get("wins") or {}).get("total") or 0)
    draws = float((fixtures.get("draws") or {}).get("total") or 0)

    ppg = (3 * wins + draws) / played if played else 1.0

    def number(value, default):
        try:
            return float(value)
        except Exception:
            return default

    gf = number((((response.get("goals") or {}).get("for") or {}).get("average") or {}).get("total"), 1.2)
    ga = number((((response.get("goals") or {}).get("against") or {}).get("average") or {}).get("total"), 1.2)

    form = form_spanish(response.get("form") or "")
    n = max(1, len(form))
    form_score = (3 * form.count("G") + form.count("E")) / n if form else ppg

    cards = response.get("cards") or {}
    yellow_total = sum(
        float(v.get("total") or 0)
        for v in (cards.get("yellow") or {}).values()
        if isinstance(v, dict)
    )
    red_total = sum(
        float(v.get("total") or 0)
        for v in (cards.get("red") or {}).values()
        if isinstance(v, dict)
    )

    return {
        "ppg": ppg,
        "gf": gf,
        "ga": ga,
        "form_score": form_score,
        "form": form,
        "yellow": yellow_total / played if played else 1.8,
        "red": red_total / played if played else 0.08,
    }

def standing_rank(data: dict[str, Any], team_id: int):
    try:
        groups = data["response"][0]["league"]["standings"]
        rows = [row for group in groups for row in group]
        for row in rows:
            if int(row["team"]["id"]) == int(team_id):
                return int(row["rank"]), len(rows)
    except Exception:
        pass
    return None, None

def table_adjust(rank, total):
    if not rank or not total or total <= 1:
        return 0.0
    return 1 - 2 * ((rank - 1) / (total - 1))

def extract_1x2_market(odds_data: dict[str, Any]):
    home, draw, away = [], [], []
    try:
        for item in odds_data.get("response", []):
            for bookmaker in item.get("bookmakers", []):
                for bet in bookmaker.get("bets", []):
                    name = (bet.get("name") or "").lower()
                    if name not in {"match winner", "1x2"}:
                        continue
                    values = {
                        str(v.get("value", "")).lower(): v.get("odd")
                        for v in bet.get("values", [])
                    }
                    oh = values.get("home") or values.get("1")
                    od = values.get("draw") or values.get("x")
                    oa = values.get("away") or values.get("2")
                    if oh and od and oa:
                        ph, pd, pa = 1 / float(oh), 1 / float(od), 1 / float(oa)
                        s = ph + pd + pa
                        home.append(ph / s * 100)
                        draw.append(pd / s * 100)
                        away.append(pa / s * 100)
        if home:
            return (
                sum(home) / len(home),
                sum(draw) / len(draw),
                sum(away) / len(away),
            )
    except Exception:
        pass
    return None

async def recent_context(home: str, away: str, referee: str, competition: str, date_text: str):
    """
    OPCIONAL. No consume cuota API-Sports.
    Si OPENAI_API_KEY no está configurada, el análisis sigue funcionando.
    """
    if not OPENAI_API_KEY or OpenAI is None:
        return {}

    prompt = f"""
Analiza información PREPARTIDO muy reciente y verificable para:
{home} vs {away}
Competición: {competition}
Fecha: {date_text}
Árbitro: {referee or "no informado"}

Prioriza fuentes oficiales del torneo, clubes y medios reputados.
No uses prestigio histórico del club.
Busca sólo información que pueda modificar una predicción actual:
lesiones, suspensiones, cambios de técnico, rotaciones anunciadas, árbitro y VAR.

Devuelve SOLO JSON válido:
{{
  "bajas_local": 0,
  "bajas_visita": 0,
  "sancionados_local": 0,
  "sancionados_visita": 0,
  "cambio_tecnico_local": false,
  "cambio_tecnico_visita": false,
  "arbitro_amarillas_promedio": null,
  "arbitro_rojas_promedio": null,
  "var": null,
  "calidad_contexto": 0,
  "resumen": ""
}}
"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = await asyncio.to_thread(
            client.responses.create,
            model=OPENAI_MODEL,
            tools=[{"type": "web_search"}],
            input=prompt,
            store=False,
        )
        match = re.search(r"\{.*\}", response.output_text, re.S)
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}

@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "version": "3.0-optimizada",
        "hora_chile": now_chile().strftime("%Y-%m-%d %H:%M"),
        "datos_deportivos": bool(API_SPORTS_KEY),
        "openai_contexto": bool(OPENAI_API_KEY),
        "modo": "consulta por equipo",
        "consultas_al_abrir": 0,
    }

@app.get("/api/search")
async def search(q: str = Query(min_length=2, max_length=80)):
    key = f"search:{q.lower().strip()}"
    cached = cache_read(key, SEARCH_TTL)
    if cached is not None:
        return {"response": cached, "cache": True, "consultas_api_sports": 0}

    data = await api_get("teams", {"search": q})
    result = []
    for row in data.get("response", []):
        team = row.get("team") or {}
        venue = row.get("venue") or {}
        result.append({
            "id": team.get("id"),
            "nombre": team.get("name"),
            "pais": team.get("country"),
            "logo": team.get("logo"),
            "ciudad": venue.get("city"),
        })

    result = result[:15]
    cache_write(key, result)
    return {"response": result, "cache": False, "consultas_api_sports": 1}

@app.get("/api/team/{team_id}/next")
async def team_next(team_id: int):
    team_cache_key = f"team-analysis:{team_id}"
    cached = cache_read(team_cache_key, ANALYSIS_TTL)
    if cached is not None:
        cached = dict(cached)
        cached["cache"] = True
        cached["consultas_api_sports"] = 0
        return cached

    # Consulta 1 del análisis: próximo partido.
    fixture_data = await api_get(
        "fixtures",
        {"team": team_id, "next": 1, "timezone": "America/Santiago"},
    )
    if not fixture_data.get("response"):
        raise HTTPException(status_code=404, detail="No se encontró próximo partido.")

    fx = fixture_data["response"][0]
    fixture_id = int(fx["fixture"]["id"])
    league = fx["league"]
    teams = fx["teams"]
    season = league["season"]
    league_id = league["id"]
    home_id = teams["home"]["id"]
    away_id = teams["away"]["id"]
    chile_iso, chile_time = to_chile(fx["fixture"]["date"])

    async def safe(coro):
        try:
            return await coro
        except Exception:
            return {}

    # Consultas 2 a 8 del análisis profundo.
    # Total al seleccionar equipo: 8 respuestas API-Sports.
    pred, standings, home_raw, away_raw, odds, injuries, lineups = await asyncio.gather(
        safe(api_get("predictions", {"fixture": fixture_id})),
        safe(api_get("standings", {"league": league_id, "season": season})),
        safe(api_get("teams/statistics", {"league": league_id, "season": season, "team": home_id})),
        safe(api_get("teams/statistics", {"league": league_id, "season": season, "team": away_id})),
        safe(api_get("odds", {"fixture": fixture_id})),
        safe(api_get("injuries", {"fixture": fixture_id})),
        safe(api_get("fixtures/lineups", {"fixture": fixture_id})),
    )

    home_stats = team_stats(home_raw)
    away_stats = team_stats(away_raw)

    home_rank, table_size = standing_rank(standings, home_id)
    away_rank, table_size2 = standing_rank(standings, away_id)
    table_size = table_size or table_size2

    provider_percent = {}
    try:
        provider_percent = pred["response"][0]["predictions"]["percent"]
    except Exception:
        pass

    ph = pct(provider_percent.get("home"), 33.5)
    pd = pct(provider_percent.get("draw"), 31.0)
    pa = pct(provider_percent.get("away"), 35.5)

    home_strength = (
        0.40 * home_stats["form_score"]
        + 0.25 * home_stats["ppg"]
        + 0.20 * (home_stats["gf"] - home_stats["ga"] + 1.5)
        + 0.15 * (table_adjust(home_rank, table_size) + 1)
        + 0.18  # localía
    )
    away_strength = (
        0.40 * away_stats["form_score"]
        + 0.25 * away_stats["ppg"]
        + 0.20 * (away_stats["gf"] - away_stats["ga"] + 1.5)
        + 0.15 * (table_adjust(away_rank, table_size) + 1)
    )

    z = home_strength - away_strength
    mh = 100 / (1 + math.exp(-1.15 * z))
    md = clamp(29 - abs(z) * 5, 18, 31)
    ma = 100 - mh
    mh, md, ma = normalize3(mh * (100 - md) / 100, md, ma * (100 - md) / 100)

    market = extract_1x2_market(odds)
    if market:
        oh, od, oa = market
        fh = 0.45 * ph + 0.40 * mh + 0.15 * oh
        fd = 0.45 * pd + 0.40 * md + 0.15 * od
        fa = 0.45 * pa + 0.40 * ma + 0.15 * oa
    else:
        fh = 0.55 * ph + 0.45 * mh
        fd = 0.55 * pd + 0.45 * md
        fa = 0.55 * pa + 0.45 * ma

    fh, fd, fa = normalize3(fh, fd, fa)

    winner_options = {"1": fh, "X": fd, "2": fa}
    winner_sel, winner_prob = max(winner_options.items(), key=lambda item: item[1])

    double_options = {"1X": fh + fd, "X2": fd + fa, "12": fh + fa}
    double_sel, double_prob = max(double_options.items(), key=lambda item: item[1])

    lambda_home = clamp((home_stats["gf"] + away_stats["ga"]) / 2 * 1.06, 0.25, 3.5)
    lambda_away = clamp((away_stats["gf"] + home_stats["ga"]) / 2 * 0.96, 0.20, 3.5)
    lambda_total = lambda_home + lambda_away

    btts_yes = (1 - math.exp(-lambda_home)) * (1 - math.exp(-lambda_away))
    if btts_yes >= 0.5:
        btts_sel, btts_prob = "Sí", btts_yes * 100
    else:
        btts_sel, btts_prob = "No", (1 - btts_yes) * 100

    goals_sel, goals_prob = best_line(lambda_total, [1.5, 2.5, 3.5])

    context = await recent_context(
        teams["home"]["name"],
        teams["away"]["name"],
        fx["fixture"].get("referee") or "",
        league["name"],
        chile_iso[:10],
    )

    referee_yellow = context.get("arbitro_amarillas_promedio")
    referee_red = context.get("arbitro_rojas_promedio")
    try:
        referee_yellow = float(referee_yellow) if referee_yellow is not None else None
    except Exception:
        referee_yellow = None
    try:
        referee_red = float(referee_red) if referee_red is not None else None
    except Exception:
        referee_red = None

    team_yellow = home_stats["yellow"] + away_stats["yellow"]
    team_red = home_stats["red"] + away_stats["red"]

    lambda_yellow = clamp(
        0.65 * team_yellow + 0.35 * (referee_yellow if referee_yellow is not None else team_yellow),
        1.0,
        9.0,
    )
    lambda_red = clamp(
        0.70 * team_red + 0.30 * (referee_red if referee_red is not None else team_red),
        0.02,
        1.2,
    )

    yellow_sel, yellow_prob = best_line(lambda_yellow, [2.5, 3.5, 4.5, 5.5])
    red_sel, red_prob = best_line(lambda_red, [0.5, 1.5])

    signals = sum(
        bool(item.get("response"))
        for item in [pred, standings, home_raw, away_raw, odds, injuries, lineups]
    )
    quality = round(signals / 7 * 85)
    if context:
        quality = round(0.85 * quality + 0.15 * float(context.get("calidad_contexto") or 0))
    quality = int(clamp(quality, 0, 100))

    def market_obj(selection, probability):
        return {
            "seleccion": selection,
            "probabilidad": round(probability),
            "confianza": confidence(probability, quality),
        }

    warnings = []
    if not lineups.get("response"):
        warnings.append("Alineaciones confirmadas aún no disponibles.")
    if not odds.get("response"):
        warnings.append("Cuotas no disponibles para este partido.")
    if not context:
        warnings.append("Contexto OpenAI no configurado; se usa sólo la base deportiva.")

    result = {
        "fixture_id": fixture_id,
        "hora_chile": chile_time,
        "fecha_hora_chile": chile_iso,
        "competicion": league["name"],
        "pais": league.get("country") or "",
        "local": teams["home"]["name"],
        "visita": teams["away"]["name"],
        "arbitro": fx["fixture"].get("referee"),
        "ganador": market_obj(winner_sel, winner_prob),
        "doble_oportunidad": market_obj(double_sel, double_prob),
        "ambos_marcan": market_obj(btts_sel, btts_prob),
        "tarjetas_amarillas": market_obj(yellow_sel, yellow_prob),
        "tarjetas_rojas": market_obj(red_sel, red_prob),
        "goles_totales": market_obj(goals_sel, goals_prob),
        "forma_local": home_stats["form"],
        "forma_visita": away_stats["form"],
        "tabla_local": home_rank,
        "tabla_visita": away_rank,
        "calidad_datos": quality,
        "actualizado_chile": now_chile().strftime("%Y-%m-%d %H:%M"),
        "contexto": context.get("resumen", "") if context else "",
        "advertencias": warnings,
        "cache": False,
        "consultas_api_sports": 8,
    }

    cache_write(team_cache_key, result)
    return result

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
