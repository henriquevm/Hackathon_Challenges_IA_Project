"""
Ferramentas de análise de comunicações (SMS e e-mails).
Detecta indícios de engenharia social, phishing ou manipulação antes de transações suspeitas.
Usadas pelo CommunicationAnalyzerAgent.
"""

import json
import re
from datetime import datetime, timedelta
from langchain.tools import tool

_SMS: list = []
_MAILS: list = []
_BASELINES: dict = {}

# Mapa pré-computado: citizen_id → lista de textos de comunicações
_SMS_BY_CITIZEN: dict = {}
_MAILS_BY_CITIZEN: dict = {}

# Padrões de phishing / engenharia social
PHISHING_PATTERNS = [
    r"urgente|urgent",
    r"sua conta.*bloqueada|account.*blocked|conta.*suspensa",
    r"clique aqui|click here|acesse agora",
    r"verificar.*dados|verify.*account|confirmar.*identidade",
    r"ganhou|you.*won|prêmio|prize",
    r"transferência.*solicitada|transfer.*requested",
    r"senha|password.*reset|redefinir.*senha",
    r"banco.*informa|bank.*alert|alerta.*segurança",
    r"pagamento.*pendente|payment.*pending",
    r"bit\.ly|tinyurl|t\.co|rb\.gy",  # URL shorteners suspeitos
]

SUSPICIOUS_SENDERS = [
    r"mirrorhack|mirror.hack|hackteam",
    r"noreply@(?!amazon|edf|google|apple)",
    r"support@(?!amazon|edf|google|apple)",
]


def init_tools(baselines: dict, sms: list, mails: list):
    global _BASELINES, _SMS, _MAILS, _SMS_BY_CITIZEN, _MAILS_BY_CITIZEN
    _BASELINES = baselines
    _SMS = sms
    _MAILS = mails

    # Pré-filtra comunicações por cidadão usando o primeiro nome do perfil
    _SMS_BY_CITIZEN.clear()
    _MAILS_BY_CITIZEN.clear()

    for citizen_id, profile in baselines.items():
        user = profile.get("user_info") or {}
        first_name = user.get("first_name", "").lower()
        last_name = user.get("last_name", "").lower()

        citizen_sms = [
            r for r in sms
            if first_name in r.get("sms", "").lower() or last_name in r.get("sms", "").lower()
        ]
        citizen_mails = [
            r for r in mails
            if first_name in r.get("mail", "").lower() or last_name in r.get("mail", "").lower()
        ]

        _SMS_BY_CITIZEN[citizen_id] = citizen_sms
        _MAILS_BY_CITIZEN[citizen_id] = citizen_mails


def _extract_timestamp(text: str) -> datetime | None:
    """Extrai timestamp do texto e retorna sempre sem timezone (naive)."""
    patterns = [
        r"Date:\s*(.+?)(?:\n|$)",
        r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})",
        r"(\w+ \d{1,2}, \d{4} \d{2}:\d{2}:\d{2})",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                dt = datetime.fromisoformat(match.group(1).strip())
                # Remove timezone para comparação uniforme
                return dt.replace(tzinfo=None)
            except Exception:
                pass
    return None


def _score_text_for_phishing(text: str) -> tuple[float, list[str]]:
    alerts = []
    score = 0.0
    text_lower = text.lower()

    for pattern in PHISHING_PATTERNS:
        if re.search(pattern, text_lower):
            alerts.append(f"Padrão suspeito detectado: '{pattern}'")
            score += 0.15

    for pattern in SUSPICIOUS_SENDERS:
        if re.search(pattern, text_lower):
            alerts.append(f"Remetente suspeito: '{pattern}'")
            score += 0.25

    return min(1.0, score), alerts


@tool
def analyze_communications(comm_check_json: str) -> str:
    """
    Analisa SMS e e-mails do cidadão em busca de phishing ou engenharia social
    próximos ao timestamp de uma transação suspeita.

    Recebe JSON com:
    - citizen_id: ID do cidadão
    - transaction_id: ID da transação
    - timestamp: timestamp da transação (ISO format)
    - window_hours: janela de tempo em horas antes da transação para analisar (padrão: 72)

    Retorna lista de alertas e score de suspeita.
    """
    try:
        params = json.loads(comm_check_json)
    except Exception:
        return json.dumps({"error": "JSON inválido", "alerts": [], "suspicion_score": 0.0})

    citizen_id = params.get("citizen_id", "")
    timestamp_str = params.get("timestamp", "")
    window_hours = int(params.get("window_hours", 72))

    try:
        tx_time = datetime.fromisoformat(timestamp_str).replace(tzinfo=None)
    except Exception:
        tx_time = None

    all_alerts = []
    total_score = 0.0

    # Usa apenas comunicações filtradas para este cidadão
    citizen_sms   = _SMS_BY_CITIZEN.get(citizen_id, [])
    citizen_mails = _MAILS_BY_CITIZEN.get(citizen_id, [])

    # Analisa SMS do cidadão
    for sms_record in citizen_sms:
        text = sms_record.get("sms", "")
        sms_score, sms_alerts = _score_text_for_phishing(text)
        if sms_alerts:
            # Verifica se o SMS é próximo ao timestamp da transação
            sms_time = _extract_timestamp(text)
            if tx_time and sms_time:
                diff_h = (tx_time - sms_time).total_seconds() / 3600
                if 0 <= diff_h <= window_hours:
                    for a in sms_alerts:
                        all_alerts.append(f"[SMS {diff_h:.0f}h antes da transação] {a}")
                    total_score += sms_score
            else:
                for a in sms_alerts:
                    all_alerts.append(f"[SMS] {a}")
                total_score += sms_score * 0.5

    # Analisa e-mails do cidadão
    for mail_record in citizen_mails:
        text = mail_record.get("mail", "")
        mail_score, mail_alerts = _score_text_for_phishing(text)
        if mail_alerts:
            mail_time = _extract_timestamp(text)
            if tx_time and mail_time:
                diff_h = (tx_time - mail_time).total_seconds() / 3600
                if 0 <= diff_h <= window_hours:
                    for a in mail_alerts:
                        all_alerts.append(f"[EMAIL {diff_h:.0f}h antes da transação] {a}")
                    total_score += mail_score
            else:
                for a in mail_alerts:
                    all_alerts.append(f"[EMAIL] {a}")
                total_score += mail_score * 0.5

    # Vulnerabilidade do cidadão a phishing (do perfil)
    profile = _BASELINES.get(citizen_id, {})
    user_info = profile.get("user_info", {}) or {}
    description = user_info.get("description", "").lower()
    phishing_vulnerability = 0.0
    if "phishing" in description or "suspicious" in description:
        if "47" in description or "50" in description:  # 47% mencionado no perfil
            phishing_vulnerability = 0.47
        elif "confian" in description or "trust" in description:
            phishing_vulnerability = 0.35

    return json.dumps({
        "citizen_id": citizen_id,
        "transaction_id": params.get("transaction_id"),
        "alerts": all_alerts,
        "phishing_vulnerability": phishing_vulnerability,
        "suspicion_score": round(min(1.0, total_score), 3),
        "sms_analyzed": len(citizen_sms),
        "mails_analyzed": len(citizen_mails),
    }, ensure_ascii=False, indent=2)
