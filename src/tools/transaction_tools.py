"""
Ferramentas para análise estatística de anomalias em transações.
Usadas pelo TransactionAnalyzerAgent.
"""

import json
import math
from langchain.tools import tool

_BASELINES: dict = {}


def init_tools(baselines: dict):
    global _BASELINES
    _BASELINES = baselines


@tool
def analyze_transaction_anomalies(transaction_json: str) -> str:
    """
    Analisa uma transação em busca de anomalias em relação ao baseline do cidadão.
    Recebe um JSON com os campos da transação e retorna um relatório com sinais de alerta.

    Campos esperados no JSON:
    - transaction_id, sender_id, recipient_id, transaction_type,
      amount, location, payment_method, sender_iban, recipient_iban,
      balance_after, description, timestamp

    Retorna um JSON com lista de alertas e um score de suspeita (0.0 a 1.0).
    """
    try:
        tx = json.loads(transaction_json)
    except Exception:
        return json.dumps({"error": "JSON inválido", "alerts": [], "suspicion_score": 0.0})

    citizen_id = tx.get("sender_id", "")
    profile = _BASELINES.get(citizen_id)

    if not profile:
        return json.dumps({
            "alerts": [f"Cidadão '{citizen_id}' não tem baseline — desconhecido"],
            "suspicion_score": 0.5,
        })

    alerts = []
    score = 0.0

    tx_type = tx.get("transaction_type", "")
    amount = float(tx.get("amount") or 0)
    recipient = tx.get("recipient_id", "")
    recipient_iban = tx.get("recipient_iban", "")
    payment_method = tx.get("payment_method", "")
    location = tx.get("location", "")

    # Parse hora
    try:
        from datetime import datetime
        ts = datetime.fromisoformat(str(tx.get("timestamp", "")))
        hour = ts.hour
    except Exception:
        hour = None

    # 1. Tipo de transação nunca visto antes
    if tx_type and tx_type not in profile["transaction_types"]:
        alerts.append(f"Tipo de transação inédito: '{tx_type}'")
        score += 0.35

    # 2. Valor anomalo para este tipo
    type_stats = profile["type_stats"].get(tx_type, {})
    if type_stats and amount > 0:
        mean = type_stats["amount_mean"]
        std = type_stats["amount_std"] or 1.0
        z_score = abs(amount - mean) / std
        if z_score > 3:
            alerts.append(f"Valor muito fora do padrão: €{amount:.2f} (z={z_score:.1f}, média=€{mean:.2f}±€{std:.2f})")
            score += min(0.45, z_score * 0.06)
        elif z_score > 2:
            alerts.append(f"Valor acima do normal: €{amount:.2f} (z={z_score:.1f})")
            score += 0.20

    # 3. Destinatário novo — sinal forte: cidadão nunca enviou para este destinatário
    if recipient and recipient not in profile["known_recipients"]:
        alerts.append(f"Destinatário desconhecido: '{recipient}'")
        score += 0.40

    # 4. IBAN de destino novo
    if recipient_iban and recipient_iban not in profile["known_recipient_ibans"]:
        alerts.append(f"IBAN destino desconhecido: '{recipient_iban}'")
        score += 0.30

    # 5. Método de pagamento novo
    if payment_method and payment_method not in profile["known_payment_methods"]:
        alerts.append(f"Método de pagamento inédito: '{payment_method}'")
        score += 0.25

    # 6. Local de pagamento novo (in-person)
    if tx_type == "in-person payment" and location and location not in profile["known_locations"]:
        alerts.append(f"Local in-person desconhecido: '{location}'")
        score += 0.20

    # 7. Horário de madrugada (0h–5h) — sinal forte, especialmente em débitos automáticos
    if hour is not None and hour < 5:
        alerts.append(f"Transação em horário de madrugada: {hour:02d}h")
        score += 0.45

    # 8. Horário fora do padrão histórico para este tipo
    if hour is not None and type_stats:
        typical_hours = type_stats.get("typical_hours", [])
        if typical_hours and hour not in typical_hours:
            closest = min(typical_hours, key=lambda h: abs(h - hour))
            if abs(hour - closest) > 4:
                alerts.append(f"Horário muito diferente do padrão ({hour:02d}h, típico: {sorted(set(typical_hours))})")
                score += 0.15

    score = min(1.0, score)

    return json.dumps({
        "transaction_id": tx.get("transaction_id"),
        "citizen_id": citizen_id,
        "alerts": alerts,
        "suspicion_score": round(score, 3),
        "alert_count": len(alerts),
    }, ensure_ascii=False, indent=2)
