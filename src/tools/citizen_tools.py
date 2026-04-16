"""
Ferramentas para consultar perfil e histórico de cidadãos.
Usadas pelo ProfileBuilderAgent e OrchestratorAgent.
"""

import json
from langchain.tools import tool

# Baseline é injetado globalmente antes da execução dos agentes
_BASELINES: dict = {}
_TRANSACTIONS = None


def init_tools(baselines: dict, transactions):
    global _BASELINES, _TRANSACTIONS
    _BASELINES = baselines
    _TRANSACTIONS = transactions


@tool
def get_citizen_profile(citizen_id: str) -> str:
    """
    Retorna o perfil completo de um cidadão: nome, profissão, salário,
    cidade de residência, coordenadas GPS médias e resumo comportamental.
    Use quando precisar entender quem é o cidadão antes de analisar transações.
    """
    profile = _BASELINES.get(citizen_id)
    if not profile:
        return f"Cidadão '{citizen_id}' não encontrado no baseline."

    user = profile.get("user_info") or {}
    result = {
        "citizen_id": citizen_id,
        "name": f"{user.get('first_name', '?')} {user.get('last_name', '?')}",
        "job": user.get("job"),
        "salary": user.get("salary"),
        "birth_year": user.get("birth_year"),
        "residence": user.get("residence"),
        "home_lat": profile.get("home_lat"),
        "home_lng": profile.get("home_lng"),
        "description": user.get("description", "")[:300],
    }
    return json.dumps(result, ensure_ascii=False, indent=2)


@tool
def get_transaction_history(citizen_id: str) -> str:
    """
    Retorna o histórico de padrões de transação de um cidadão: tipos usados,
    valores típicos por tipo, métodos de pagamento, destinatários e IBANs conhecidos.
    Use para comparar uma transação nova com o comportamento histórico do cidadão.
    """
    profile = _BASELINES.get(citizen_id)
    if not profile:
        return f"Cidadão '{citizen_id}' não encontrado no baseline."

    result = {
        "citizen_id": citizen_id,
        "transaction_types": profile["transaction_types"],
        "type_stats": profile["type_stats"],
        "known_recipients": profile["known_recipients"],
        "known_recipient_ibans": profile["known_recipient_ibans"],
        "known_payment_methods": profile["known_payment_methods"],
        "known_locations": profile["known_locations"],
        "total_transactions": profile["total_tx_count"],
        "balance_mean": round(profile["balance_mean"], 2),
        "balance_std": round(profile["balance_std"], 2),
    }
    return json.dumps(result, ensure_ascii=False, indent=2)
