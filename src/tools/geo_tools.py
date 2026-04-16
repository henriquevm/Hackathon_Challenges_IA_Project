"""
Ferramentas de validação geográfica.
Cruza a localização GPS do cidadão com o local da transação.
Usadas pelo GeoValidatorAgent.
"""

import json
from datetime import timedelta
from langchain.tools import tool
from geopy.distance import geodesic

_BASELINES: dict = {}
_LOCATIONS = None


def init_tools(baselines: dict, locations):
    global _BASELINES, _LOCATIONS
    _BASELINES = baselines
    _LOCATIONS = locations


@tool
def check_geolocation(geo_check_json: str) -> str:
    """
    Verifica se o cidadão estava fisicamente presente no local da transação.
    Compara a localização GPS registrada próxima ao timestamp com o local da transação.

    Recebe JSON com:
    - citizen_id: ID do cidadão (biotag)
    - transaction_id: ID da transação
    - timestamp: timestamp da transação (ISO format)
    - transaction_lat: latitude do local da transação (opcional)
    - transaction_lng: longitude do local da transação (opcional)
    - transaction_city: cidade da transação (opcional)

    Retorna relatório com distância estimada e nível de suspeita.
    """
    try:
        params = json.loads(geo_check_json)
    except Exception:
        return json.dumps({"error": "JSON inválido", "alerts": [], "suspicion_score": 0.0})

    citizen_id = params.get("citizen_id", "")
    timestamp_str = params.get("timestamp", "")
    tx_lat = params.get("transaction_lat")
    tx_lng = params.get("transaction_lng")

    if not citizen_id or _LOCATIONS is None:
        return json.dumps({"alerts": ["Dados insuficientes para validação geográfica"], "suspicion_score": 0.0})

    # Filtrar registros GPS do cidadão
    citizen_locs = _LOCATIONS[_LOCATIONS["biotag"] == citizen_id].copy()
    if citizen_locs.empty:
        return json.dumps({"alerts": [f"Sem dados GPS para cidadão {citizen_id}"], "suspicion_score": 0.0})

    alerts = []
    score = 0.0

    # Se temos coordenadas da transação, comparar com GPS próximo ao timestamp
    if tx_lat is not None and tx_lng is not None:
        try:
            from datetime import datetime
            tx_time = datetime.fromisoformat(timestamp_str)
            window = timedelta(hours=12)

            nearby = citizen_locs[
                (citizen_locs["timestamp"] >= tx_time - window) &
                (citizen_locs["timestamp"] <= tx_time + window)
            ]

            if not nearby.empty:
                nearest = nearby.iloc[(nearby["timestamp"] - tx_time).abs().argsort()[:1]]
                gps_lat = float(nearest["lat"].iloc[0])
                gps_lng = float(nearest["lng"].iloc[0])
                gps_city = nearest["city"].iloc[0] if "city" in nearest.columns else "?"
                gps_time = nearest["timestamp"].iloc[0]

                distance_km = geodesic((gps_lat, gps_lng), (tx_lat, tx_lng)).km
                time_diff_h = abs((tx_time - gps_time).total_seconds()) / 3600

                result = {
                    "citizen_id": citizen_id,
                    "gps_at_time": {
                        "lat": gps_lat,
                        "lng": gps_lng,
                        "city": gps_city,
                        "timestamp": str(gps_time),
                    },
                    "transaction_location": {"lat": tx_lat, "lng": tx_lng},
                    "distance_km": round(distance_km, 2),
                    "time_diff_hours": round(time_diff_h, 1),
                }

                if distance_km > 500:
                    alerts.append(f"Impossible travel: cidadão estava a {distance_km:.0f}km do local da transação")
                    score += 0.7
                elif distance_km > 100:
                    alerts.append(f"Cidadão estava longe do local: {distance_km:.0f}km de distância")
                    score += 0.35
                elif distance_km > 30:
                    alerts.append(f"Cidadão estava a {distance_km:.0f}km da transação")
                    score += 0.15

                result["alerts"] = alerts
                result["suspicion_score"] = round(min(1.0, score), 3)
                return json.dumps(result, ensure_ascii=False, indent=2)
            else:
                return json.dumps({
                    "alerts": ["Sem registros GPS próximos ao horário da transação"],
                    "suspicion_score": 0.0,
                })
        except Exception as e:
            return json.dumps({"error": str(e), "alerts": [], "suspicion_score": 0.0})

    # Sem coordenadas: verificar se a cidade bate com a residência
    profile = _BASELINES.get(citizen_id, {})
    user_info = profile.get("user_info", {}) or {}
    residence = user_info.get("residence", {}) or {}
    home_city = residence.get("city", "").lower()
    tx_city = str(params.get("transaction_city", "")).lower()

    if home_city and tx_city and tx_city not in home_city and home_city not in tx_city:
        alerts.append(f"Transação em cidade diferente da residência: '{tx_city}' vs '{home_city}'")
        score += 0.2

    return json.dumps({
        "citizen_id": citizen_id,
        "alerts": alerts,
        "suspicion_score": round(min(1.0, score), 3),
    }, ensure_ascii=False, indent=2)
