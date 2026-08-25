from __future__ import annotations

import asyncio
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from zoneinfo import ZoneInfo

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

APP_NAME = "Radar Fútbol"
API_BASE = "https://v3.football.api-sports.io"
CHILE_TZ = ZoneInfo("America/Santiago")
API_SPORTS_KEY = os.getenv("API_SPORTS_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6").strip()
MAX_FIXTURES_ANALYZE = int(os.getenv("MAX_FIXTURES_ANALYZE", "30"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(ROOT, "frontend")

app = FastAPI(title=APP_NAME, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)

EUROPE = {
    "Albania","Andorra","Armenia","Austria","Azerbaijan","Belarus","Belgium","Bosnia","Bulgaria",
    "Croatia","Cyprus","Czech-Republic","Czech Republic","Denmark","England","Estonia","Faroe-Islands",
    "Finland","France","Georgia","Germany","Gibraltar","Greece","Hungary","Iceland","Ireland","Israel",
    "Italy","Kazakhstan","Kosovo","Latvia","Liechtenstein","Lithuania","Luxembourg","Malta","Moldova",
    "Montenegro","Netherlands","North-Macedonia","Northern-Ireland","Norway","Poland","Portugal",
    "Romania","Russia","San-Marino","Scotland","Serbia","Slovakia","Slovenia","Spain","Sweden",
    "Switzerland","Turkey","Türkiye","Ukraine","Wales"
}
AMERICA = {
    "Argentina","Bolivia","Brazil","Brasil","Canada","Chile","Colombia","Costa-Rica","Costa Rica",
    "Cuba","Dominican-Republic","Ecuador","El-Salvador","Guatemala","Haiti","Honduras","Jamaica",
    "Mexico","Nicaragua","Panama","Paraguay","Peru","Puerto-Rico","Trinidad-And-Tobago","USA",
    "United-States","United States","Uruguay","Venezuela"
}
ASIA = {
    "Australia","Bahrain","Bangladesh","Bhutan","Cambodia","China","Chinese-Taipei","Hong-Kong",
    "India","Indonesia","Iran","Iraq","Japan","Jordan","Kuwait","Kyrgyzstan","Lebanon","Malaysia",
    "Maldives","Mongolia","Myanmar","Nepal","North-Korea","Oman","Pakistan","Palestine","Philippines",
    "Qatar","Saudi-Arabia","Saudi Arabia","Singapore","South-Korea","South Korea","Sri-Lanka","Syria",
    "Tajikistan","Thailand","Turkmenistan","UAE","United-Arab-Emirates","Uzbekistan","Vietnam"
}

SECOND_TERMS = (
    "2. bundesliga","bundesliga 2","serie b","serie b ","liga 2","ligue 2","segunda","primera b",
    "championship","league one","j2 league","j league 2","j2","k league 2","1st division",
    "first division b","segunda división","segunda division","2nd division","second division",
    "serie b","national league"
)
CUP_TERMS = (
    "cup","copa","coppa","pokal","coupe","taça","taca","fa cup","carabao","libertadores",
    "sudamericana","champions league","europa league","conference league","super cup","supercopa",
    "knockout","trophy"
)

def now_chile() -> datetime:
    return datetime.now(tz=CHILE_TZ)

def to_chile_iso(dt_str: str) -> tuple[str, str]:
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        local = dt.astimezone(CHILE_TZ)
        return local.isoformat(), local.strftime("%H:%M")
    except Exception:
        return dt_str, "--:--"

def region_for(country: str) -> str:
    c = (country or "").strip()
    if c in EUROPE: return "Europa"
    if c in AMERICA: return "América"
    if c in ASIA: return "Asia"
    return "Otro"

def tier_for(league_name: str, league_type: str) -> str:
    name = (league_name or "").lower()
    typ = (league_type or "").lower()
    if typ == "cup" or any(t in name for t in CUP_TERMS):
        return "Copa"
    if any(t in name for t in SECOND_TERMS):
        return "Segunda"
    return "Primera"

async def api_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    if not API_SPORTS_KEY:
        raise HTTPException(
            status_code=503,
            detail="Falta API_SPORTS_KEY. Agrega una clave de API-FOOTBALL en el backend."
        )
    headers = {"x-apisports-key": API_SPORTS_KEY}
    timeout = httpx.Timeout(20.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(f"{API_BASE}/{path}", params=params, headers=headers)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"API deportiva respondió {r.status_code}.")
    data = r.json()
    if data.get("errors"):
        raise HTTPException(status_code=502, detail=f"API deportiva: {data['errors']}")
    return data

def pct(x: Any) -> Optional[float]:
    if x is None: return None
    if isinstance(x, (int, float)): return float(x)
    s = str(x).replace("%","").replace(",",".").strip()
    try: return float(s)
    except Exception: return None

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def normalize3(a: float, b: float, c: float) -> tuple[float, float, float]:
    total = a+b+c
    if total <= 0:
        return (33.34, 33.33, 33.33)
    return tuple(round(v/total*100, 2) for v in (a,b,c))

def poisson_cdf(k: int, lam: float) -> float:
    lam = max(0.01, float(lam))
    s = 0.0
    for i in range(k+1):
        s += math.exp(-lam) * (lam ** i) / math.factorial(i)
    return min(1.0, max(0.0, s))

def over_prob(line: float, lam: float) -> float:
    # +1.5 => 2 or más. +3.5 => 4 o más.
    threshold = math.floor(line)
    return 1.0 - poisson_cdf(threshold, lam)

def under_prob(line: float, lam: float) -> float:
    # -3.5 => 3 o menos.
    threshold = math.floor(line)
    return poisson_cdf(threshold, lam)

def best_line(lam: float, lines: list[float], prefer_over: bool | None = None) -> dict[str, Any]:
    options = []
    for line in lines:
        po = over_prob(line, lam)
        pu = under_prob(line, lam)
        options.append(("+", line, po))
        options.append(("-", line, pu))
    if prefer_over is True:
        over_opts = [x for x in options if x[0] == "+"]
        best = max(over_opts, key=lambda x: x[2])
    elif prefer_over is False:
        under_opts = [x for x in options if x[0] == "-"]
        best = max(under_opts, key=lambda x: x[2])
    else:
        best = max(options, key=lambda x: x[2])
    sign, line, p = best
    return {"seleccion": f"{sign}{str(line).replace('.',',')}", "probabilidad": round(p*100)}

def confidence(score: float) -> str:
    if score >= 75: return "Alta"
    if score >= 55: return "Media"
    return "Baja"

def parse_form(form: str) -> dict[str, int]:
    # API-Football suele usar W/D/L. Lo convertimos internamente.
    f = (form or "").upper()
    return {"G": f.count("W"), "E": f.count("D"), "P": f.count("L"), "n": len(f)}

def team_strength_from_stats(stats: dict[str, Any]) -> dict[str, float]:
    response = (stats or {}).get("response") or {}
    fixtures = response.get("fixtures") or {}
    wins = fixtures.get("wins") or {}
    draws = fixtures.get("draws") or {}
    loses = fixtures.get("loses") or {}
    played = fixtures.get("played") or {}
    n = float(played.get("total") or 0)
    w = float(wins.get("total") or 0)
    d = float(draws.get("total") or 0)
    l = float(loses.get("total") or 0)
    ppg = (3*w+d)/n if n else 1.0

    gf_avg = (((response.get("goals") or {}).get("for") or {}).get("average") or {}).get("total")
    ga_avg = (((response.get("goals") or {}).get("against") or {}).get("average") or {}).get("total")
    try: gf_avg = float(gf_avg)
    except Exception: gf_avg = 1.2
    try: ga_avg = float(ga_avg)
    except Exception: ga_avg = 1.2

    form = parse_form(response.get("form") or "")
    form_score = ((3*form["G"] + form["E"]) / max(1, form["n"])) if form["n"] else ppg

    cards = response.get("cards") or {}
    yellow = cards.get("yellow") or {}
    red = cards.get("red") or {}
    yellow_total = sum(float(v.get("total") or 0) for v in yellow.values() if isinstance(v, dict))
    red_total = sum(float(v.get("total") or 0) for v in red.values() if isinstance(v, dict))

    return {
        "ppg": ppg,
        "gf": gf_avg,
        "ga": ga_avg,
        "form_score": form_score,
        "yellow_per_match": yellow_total/n if n else 1.8,
        "red_per_match": red_total/n if n else 0.08,
        "played": n
    }

def standing_rank(data: dict[str, Any], team_id: int) -> tuple[Optional[int], Optional[int]]:
    try:
        groups = data["response"][0]["league"]["standings"]
        rows = [r for g in groups for r in g]
        total = len(rows)
        for r in rows:
            if int(r["team"]["id"]) == int(team_id):
                return int(r["rank"]), total
    except Exception:
        pass
    return None, None

def standings_adjust(rank: Optional[int], total: Optional[int]) -> float:
    if not rank or not total or total <= 1:
        return 0.0
    # +1 arriba, -1 abajo
    return 1.0 - 2.0*((rank-1)/(total-1))

def extract_market_odds(odds_data: dict[str, Any]) -> Optional[tuple[float,float,float]]:
    # Busca mercado Match Winner / 1X2 y promedia probabilidades implícitas
    home, draw, away = [], [], []
    try:
        for item in odds_data.get("response", []):
            for book in item.get("bookmakers", []):
                for bet in book.get("bets", []):
                    name = (bet.get("name") or "").lower()
                    if "match winner" not in name and "1x2" not in name:
                        continue
                    vals = {str(v.get("value","")).lower(): v.get("odd") for v in bet.get("values", [])}
                    h = vals.get("home") or vals.get("1")
                    d = vals.get("draw") or vals.get("x")
                    a = vals.get("away") or vals.get("2")
                    if h and d and a:
                        hp, dp, ap = 1/float(h), 1/float(d), 1/float(a)
                        s = hp+dp+ap
                        home.append(hp/s*100); draw.append(dp/s*100); away.append(ap/s*100)
        if home:
            return (sum(home)/len(home), sum(draw)/len(draw), sum(away)/len(away))
    except Exception:
        return None
    return None

async def openai_context(home: str, away: str, referee: str, competition: str, date_text: str) -> dict[str, Any]:
    if not OPENAI_API_KEY or OpenAI is None:
        return {}
    prompt = f"""
Busca información prepartido ACTUALIZADA para {home} vs {away}, competición {competition}, fecha {date_text}.
Árbitro: {referee or 'no informado'}.

Prioridad de fuentes: organismo/competición y clubes oficiales; después medios reputados; para estadísticas arbitrales se permiten
fuentes especializadas y de mercado. No uses prestigio histórico del club como factor.

Devuelve SOLO JSON válido, sin markdown, con este esquema:
{{
  "bajas_local": 0,
  "bajas_visita": 0,
  "sancionados_local": 0,
  "sancionados_visita": 0,
  "cambio_tecnico_local_reciente": false,
  "cambio_tecnico_visita_reciente": false,
  "arbitro_amarillas_promedio": null,
  "arbitro_rojas_promedio": null,
  "var": null,
  "calidad_contexto": 0,
  "resumen": ""
}}
"calidad_contexto" debe ser 0-100 según qué tan actuales y verificables sean los datos.
"""
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = await asyncio.to_thread(
            client.responses.create,
            model=OPENAI_MODEL,
            tools=[{"type":"web_search"}],
            input=prompt,
            store=False,
        )
        txt = response.output_text.strip()
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            return {}
        return json.loads(m.group(0))
    except Exception:
        return {}

class Market(BaseModel):
    seleccion: str
    probabilidad: int
    confianza: str

class PredictionOut(BaseModel):
    fixture_id: int
    hora_chile: str
    fecha_hora_chile: str
    competicion: str
    pais: str
    region: str
    categoria: str
    local: str
    visita: str
    arbitro: Optional[str] = None
    ganador: Market
    doble_oportunidad: Market
    ambos_marcan: Market
    tarjetas_amarillas: Market
    tarjetas_rojas: Market
    goles_totales: Market
    forma_local: str = ""
    forma_visita: str = ""
    tabla_local: Optional[int] = None
    tabla_visita: Optional[int] = None
    calidad_datos: int
    actualizado_chile: str
    advertencias: list[str] = []

async def full_prediction(fixture_id: int, enrich_context: bool = True) -> PredictionOut:
    fixture_data = await api_get("fixtures", {"id": fixture_id})
    if not fixture_data.get("response"):
        raise HTTPException(404, "Partido no encontrado.")
    fx = fixture_data["response"][0]
    league = fx["league"]
    teams = fx["teams"]
    fixture = fx["fixture"]
    season = league.get("season")
    league_id = league.get("id")
    home_id, away_id = teams["home"]["id"], teams["away"]["id"]
    chile_iso, chile_time = to_chile_iso(fixture.get("date"))

    # Se solicitan en paralelo para máxima frescura.
    pred_task = api_get("predictions", {"fixture": fixture_id})
    stand_task = api_get("standings", {"league": league_id, "season": season})
    hstat_task = api_get("teams/statistics", {"league": league_id, "season": season, "team": home_id})
    astat_task = api_get("teams/statistics", {"league": league_id, "season": season, "team": away_id})
    odds_task = api_get("odds", {"fixture": fixture_id})
    injuries_task = api_get("injuries", {"fixture": fixture_id})
    lineups_task = api_get("fixtures/lineups", {"fixture": fixture_id})

    async def safe(coro):
        try:
            return await coro
        except Exception:
            return {}

    pred, standings, hs_raw, as_raw, odds, injuries, lineups = await asyncio.gather(
        safe(pred_task), safe(stand_task), safe(hstat_task), safe(astat_task),
        safe(odds_task), safe(injuries_task), safe(lineups_task)
    )

    hs = team_strength_from_stats(hs_raw)
    aws = team_strength_from_stats(as_raw)
    hrank, table_size = standing_rank(standings, home_id)
    arank, table_size2 = standing_rank(standings, away_id)
    table_n = table_size or table_size2

    # Base proveedor
    pp = {}
    try:
        pp = pred["response"][0]["predictions"]["percent"]
    except Exception:
        pp = {}
    ph = pct(pp.get("home")) or 33.5
    pd = pct(pp.get("draw")) or 31.0
    pa = pct(pp.get("away")) or 35.5

    # Modelo propio actual: forma + tabla + localía + ataque/defensa
    htable = standings_adjust(hrank, table_n)
    atable = standings_adjust(arank, table_n)
    h_strength = 0.40*hs["form_score"] + 0.25*hs["ppg"] + 0.20*(hs["gf"]-hs["ga"]+1.5) + 0.15*(htable+1)
    a_strength = 0.40*aws["form_score"] + 0.25*aws["ppg"] + 0.20*(aws["gf"]-aws["ga"]+1.5) + 0.15*(atable+1)
    # localía moderada, nunca prestigio histórico
    h_strength += 0.18
    z = h_strength - a_strength
    mh = 100/(1+math.exp(-1.15*z))
    md = clamp(29 - abs(z)*5, 18, 31)
    ma = 100 - mh
    mh, md, ma = normalize3(mh*(100-md)/100, md, ma*(100-md)/100)

    # Mercado
    market = extract_market_odds(odds)
    if market:
        oh, od, oa = market
        # 45% predicción proveedor, 40% modelo propio, 15% mercado
        fh = 0.45*ph + 0.40*mh + 0.15*oh
        fd = 0.45*pd + 0.40*md + 0.15*od
        fa = 0.45*pa + 0.40*ma + 0.15*oa
    else:
        fh = 0.55*ph + 0.45*mh
        fd = 0.55*pd + 0.45*md
        fa = 0.55*pa + 0.45*ma
    fh, fd, fa = normalize3(fh, fd, fa)

    winner_map = {"1": fh, "X": fd, "2": fa}
    winner_sel, winner_prob = max(winner_map.items(), key=lambda kv: kv[1])

    doubles = {"1X": fh+fd, "X2": fd+fa, "12": fh+fa}
    do_sel, do_prob = max(doubles.items(), key=lambda kv: kv[1])

    # Goles esperados: ataque actual vs defensa rival.
    lam_home = clamp((hs["gf"] + aws["ga"])/2 * 1.06, 0.25, 3.5)
    lam_away = clamp((aws["gf"] + hs["ga"])/2 * 0.96, 0.20, 3.5)
    lam_total = lam_home + lam_away
    p_home_scores = 1-math.exp(-lam_home)
    p_away_scores = 1-math.exp(-lam_away)
    btts_yes = p_home_scores*p_away_scores
    if btts_yes >= 0.5:
        am_sel, am_prob = "Sí", btts_yes*100
    else:
        am_sel, am_prob = "No", (1-btts_yes)*100

    goals_market = best_line(lam_total, [1.5, 2.5, 3.5])

    # Contexto de árbitro/noticias sólo al abrir el partido individual.
    context = {}
    if enrich_context:
        context = await openai_context(
            teams["home"]["name"], teams["away"]["name"],
            fixture.get("referee") or "", league["name"],
            chile_iso[:10]
        )

    ref_y = context.get("arbitro_amarillas_promedio")
    ref_r = context.get("arbitro_rojas_promedio")
    try: ref_y = float(ref_y) if ref_y is not None else None
    except Exception: ref_y = None
    try: ref_r = float(ref_r) if ref_r is not None else None
    except Exception: ref_r = None

    team_y = hs["yellow_per_match"] + aws["yellow_per_match"]
    team_r = hs["red_per_match"] + aws["red_per_match"]
    lam_y = clamp(0.65*team_y + 0.35*(ref_y if ref_y is not None else team_y), 1.0, 9.0)
    lam_r = clamp(0.70*team_r + 0.30*(ref_r if ref_r is not None else team_r), 0.02, 1.2)

    yell_market = best_line(lam_y, [2.5, 3.5, 4.5, 5.5])
    red_market = best_line(lam_r, [0.5, 1.5])

    # Calidad de datos/confianza
    signals = 0
    signals += 1 if pred.get("response") else 0
    signals += 1 if standings.get("response") else 0
    signals += 1 if hs_raw.get("response") else 0
    signals += 1 if as_raw.get("response") else 0
    signals += 1 if odds.get("response") else 0
    signals += 1 if injuries.get("response") else 0
    signals += 1 if lineups.get("response") else 0
    quality = round(signals/7*85)
    if context:
        quality = round(0.85*quality + 0.15*float(context.get("calidad_contexto") or 0))
    quality = int(clamp(quality, 0, 100))

    warnings = []
    if not lineups.get("response"):
        warnings.append("Alineaciones confirmadas aún no disponibles.")
    if not odds.get("response"):
        warnings.append("Sin consenso de cuotas disponible.")
    if not context:
        warnings.append("Sin enriquecimiento de noticias/árbitro en esta consulta.")

    def mk(sel: str, prob: float, bias: float = 0) -> Market:
        cscore = clamp(0.55*quality + 0.45*abs(prob-50)*2 + bias, 0, 100)
        return Market(seleccion=sel, probabilidad=int(round(prob)), confianza=confidence(cscore))

    return PredictionOut(
        fixture_id=fixture_id,
        hora_chile=chile_time,
        fecha_hora_chile=chile_iso,
        competicion=league["name"],
        pais=league.get("country") or "",
        region=region_for(league.get("country") or ""),
        categoria=tier_for(league.get("name") or "", league.get("type") or ""),
        local=teams["home"]["name"],
        visita=teams["away"]["name"],
        arbitro=fixture.get("referee"),
        ganador=mk(winner_sel, winner_prob),
        doble_oportunidad=mk(do_sel, do_prob, 5),
        ambos_marcan=mk(am_sel, am_prob),
        tarjetas_amarillas=mk(yell_market["seleccion"], yell_market["probabilidad"]),
        tarjetas_rojas=mk(red_market["seleccion"], red_market["probabilidad"], 3),
        goles_totales=mk(goals_market["seleccion"], goals_market["probabilidad"], 5),
        forma_local=(hs_raw.get("response") or {}).get("form","").replace("W","G").replace("D","E").replace("L","P"),
        forma_visita=(as_raw.get("response") or {}).get("form","").replace("W","G").replace("D","E").replace("L","P"),
        tabla_local=hrank,
        tabla_visita=arank,
        calidad_datos=quality,
        actualizado_chile=now_chile().strftime("%Y-%m-%d %H:%M"),
        advertencias=warnings
    )

@app.get("/api/health")
async def health():
    return {
        "ok": True,
        "app": APP_NAME,
        "hora_chile": now_chile().strftime("%Y-%m-%d %H:%M"),
        "datos_deportivos": bool(API_SPORTS_KEY),
        "openai_contexto": bool(OPENAI_API_KEY),
    }

@app.get("/api/fixtures")
async def fixtures(
    date: Optional[str] = None,
    region: Optional[str] = None,
    categoria: Optional[str] = None,
):
    date = date or now_chile().strftime("%Y-%m-%d")
    data = await api_get("fixtures", {"date": date, "timezone": "America/Santiago"})
    out = []
    for fx in data.get("response", []):
        league = fx["league"]
        reg = region_for(league.get("country") or "")
        cat = tier_for(league.get("name") or "", league.get("type") or "")
        if reg not in {"América","Europa","Asia"}:
            continue
        if region and region != "Todos" and reg != region:
            continue
        if categoria and categoria != "Todas" and cat != categoria:
            continue
        chile_iso, chile_time = to_chile_iso(fx["fixture"]["date"])
        out.append({
            "fixture_id": fx["fixture"]["id"],
            "hora_chile": chile_time,
            "fecha_hora_chile": chile_iso,
            "competicion": league["name"],
            "pais": league.get("country"),
            "region": reg,
            "categoria": cat,
            "local": fx["teams"]["home"]["name"],
            "visita": fx["teams"]["away"]["name"],
            "arbitro": fx["fixture"].get("referee"),
            "estado": fx["fixture"]["status"].get("short"),
        })
    out.sort(key=lambda x: x["fecha_hora_chile"])
    return {"date": date, "count": len(out), "response": out, "actualizado_chile": now_chile().strftime("%Y-%m-%d %H:%M")}

@app.get("/api/search")
async def search(q: str = Query(min_length=2, max_length=80)):
    data = await api_get("teams", {"search": q})
    out = []
    for row in data.get("response", []):
        team = row.get("team") or {}
        venue = row.get("venue") or {}
        out.append({
            "id": team.get("id"),
            "nombre": team.get("name"),
            "pais": team.get("country"),
            "logo": team.get("logo"),
            "ciudad": venue.get("city"),
        })
    return {"response": out[:20]}

@app.get("/api/team/{team_id}/next")
async def team_next(team_id: int):
    data = await api_get("fixtures", {"team": team_id, "next": 1, "timezone": "America/Santiago"})
    if not data.get("response"):
        raise HTTPException(404, "No se encontró próximo partido.")
    fx = data["response"][0]
    return await full_prediction(int(fx["fixture"]["id"]), enrich_context=True)

@app.get("/api/prediction/{fixture_id}", response_model=PredictionOut)
async def prediction(fixture_id: int):
    return await full_prediction(fixture_id, enrich_context=True)

@app.get("/api/top")
async def top(
    date: Optional[str] = None,
    region: Optional[str] = None,
    categoria: Optional[str] = None,
):
    date = date or now_chile().strftime("%Y-%m-%d")
    fxdata = await fixtures(date=date, region=region, categoria=categoria)
    candidates = [f for f in fxdata["response"] if f["estado"] in {"NS","TBD"}][:MAX_FIXTURES_ANALYZE]

    sem = asyncio.Semaphore(6)
    async def one(f):
        async with sem:
            try:
                p = await full_prediction(int(f["fixture_id"]), enrich_context=False)
                markets = {
                    "Ganador": p.ganador,
                    "Doble Oportunidad": p.doble_oportunidad,
                    "Ambos Marcan": p.ambos_marcan,
                    "Tarjetas Amarillas": p.tarjetas_amarillas,
                    "Tarjetas Rojas": p.tarjetas_rojas,
                    "Goles Totales": p.goles_totales,
                }
                best_name, best_market = max(markets.items(), key=lambda kv: kv[1].probabilidad)
                return {
                    **p.model_dump(),
                    "mejor_prediccion": best_name,
                    "mejor_seleccion": best_market.seleccion,
                    "mejor_probabilidad": best_market.probabilidad,
                    "mejor_confianza": best_market.confianza,
                }
            except Exception:
                return None

    results = await asyncio.gather(*(one(f) for f in candidates))
    results = [r for r in results if r]
    results.sort(key=lambda r: (r["mejor_probabilidad"], r["calidad_datos"]), reverse=True)
    return {
        "date": date,
        "analizados": len(results),
        "limite_analisis": MAX_FIXTURES_ANALYZE,
        "actualizado_chile": now_chile().strftime("%Y-%m-%d %H:%M"),
        "response": results,
    }

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
