"""
Constrói contexto rico para os agentes:
- Few-shot examples: transações legítimas e suspeitas do mesmo cidadão
- Histórico recente: últimas N transações antes da atual
- Perfil narrativo: descrição do cidadão + resumo financeiro
- Comunicações próximas: SMS/emails na janela de 72h antes da transação
"""

import json
from datetime import datetime, timedelta
import pandas as pd


class ContextBuilder:
    def __init__(self, baselines: dict, transactions: pd.DataFrame,
                 sms: list, mails: list,
                 sms_by_citizen: dict, mails_by_citizen: dict):
        self.baselines = baselines
        self.transactions = transactions
        self.sms_by_citizen = sms_by_citizen
        self.mails_by_citizen = mails_by_citizen

    def build(self, tx: dict) -> dict:
        """Retorna dict com todo o contexto necessário para análise de uma transação."""
        citizen_id = tx.get("sender_id", "")
        tx_type = tx.get("transaction_type", "")
        timestamp_str = str(tx.get("timestamp", ""))

        try:
            tx_time = datetime.fromisoformat(timestamp_str).replace(tzinfo=None)
        except Exception:
            tx_time = None

        return {
            "citizen_profile": self._get_profile_narrative(citizen_id),
            "recent_history": self._get_recent_history(citizen_id, tx_time, n=5),
            "few_shot_legit": self._get_legit_examples(citizen_id, tx_type),
            "few_shot_suspicious": self._get_suspicious_examples(citizen_id),
            "nearby_communications": self._get_nearby_comms(citizen_id, tx_time),
        }

    # ------------------------------------------------------------------

    def _get_profile_narrative(self, citizen_id: str) -> str:
        profile = self.baselines.get(citizen_id, {})
        user = profile.get("user_info") or {}

        name = f"{user.get('first_name', '?')} {user.get('last_name', '?')}"
        job = user.get("job", "?")
        salary = user.get("salary", "?")
        city = (user.get("residence") or {}).get("city", "?")
        description = user.get("description", "")[:400]

        type_stats = profile.get("type_stats", {})
        stats_lines = []
        for tx_type, stats in type_stats.items():
            stats_lines.append(
                f"  [{tx_type}] valor típico: €{stats['amount_mean']:.0f}±€{stats['amount_std']:.0f}, "
                f"horários: {sorted(set(stats['typical_hours']))}"
            )

        known_r = profile.get("known_recipients", [])
        outliers = profile.get("outliers_excluded", 0)

        return (
            f"Nome: {name} | Profissão: {job} | Salário: €{salary}/ano | Cidade: {city}\n"
            f"Perfil: {description}\n"
            f"Padrões financeiros:\n" + "\n".join(stats_lines) + "\n"
            f"Destinatários recorrentes conhecidos: {', '.join(known_r)}\n"
            f"Transações excluídas como outliers no treino: {outliers}"
        )

    def _get_recent_history(self, citizen_id: str, tx_time: datetime | None, n: int = 5) -> str:
        txs = self.transactions[self.transactions["sender_id"] == citizen_id].copy()
        if tx_time is not None:
            txs = txs[txs["timestamp"] < tx_time]
        txs = txs.sort_values("timestamp", ascending=False).head(n)

        if txs.empty:
            return "Sem histórico anterior disponível."

        lines = []
        for _, r in txs.iterrows():
            lines.append(
                f"  {str(r['timestamp'])[:16]}  {r['transaction_type']:<16}  "
                f"€{r['amount']:<10.2f}  → {r['recipient_id']}  "
                f"{str(r.get('description',''))[:50]}"
            )
        return "\n".join(lines)

    def _get_legit_examples(self, citizen_id: str, tx_type: str, n: int = 2) -> str:
        """Retorna exemplos de transações claramente legítimas do mesmo tipo e cidadão."""
        txs = self.transactions[
            (self.transactions["sender_id"] == citizen_id) &
            (self.transactions["transaction_type"] == tx_type)
        ].copy()

        # "Claramente legítimas": horário diurno (8h–20h), destinatário recorrente
        profile = self.baselines.get(citizen_id, {})
        known_recipients = set(profile.get("known_recipients", []))

        legit = txs[
            (txs["timestamp"].dt.hour >= 8) &
            (txs["timestamp"].dt.hour <= 20) &
            (txs["recipient_id"].isin(known_recipients))
        ].head(n)

        if legit.empty:
            legit = txs.head(n)

        if legit.empty:
            return "Sem exemplos legítimos disponíveis para este tipo."

        lines = []
        for _, r in legit.iterrows():
            lines.append(
                f"  ✅ LEGÍTIMA: {r['transaction_type']}, €{r['amount']:.2f}, "
                f"{str(r['timestamp'])[:16]} ({r['timestamp'].hour:02d}h), "
                f"→ {r['recipient_id']}, descrição: {str(r.get('description',''))[:40]}"
            )
        return "\n".join(lines)

    def _get_suspicious_examples(self, citizen_id: str) -> str:
        """Retorna exemplos de transações suspeitas já conhecidas do cidadão."""
        txs = self.transactions[self.transactions["sender_id"] == citizen_id].copy()
        profile = self.baselines.get(citizen_id, {})
        known_recipients = set(profile.get("known_recipients", []))

        suspicious = txs[
            (txs["timestamp"].dt.hour < 5) |
            (~txs["recipient_id"].isin(known_recipients) & txs["recipient_id"].notna())
        ].head(2)

        if suspicious.empty:
            return "Nenhum exemplo suspeito identificado no histórico de treino."

        lines = []
        for _, r in suspicious.iterrows():
            hour = r["timestamp"].hour
            flags = []
            if hour < 5:
                flags.append(f"horário {hour:02d}h")
            if r["recipient_id"] not in known_recipients and pd.notna(r["recipient_id"]):
                flags.append("destinatário novo")
            lines.append(
                f"  🚨 SUSPEITA: {r['transaction_type']}, €{r['amount']:.2f}, "
                f"{str(r['timestamp'])[:16]}, → {r['recipient_id']} "
                f"[sinais: {', '.join(flags)}]"
            )
        return "\n".join(lines)

    def _get_nearby_comms(self, citizen_id: str, tx_time: datetime | None, hours: int = 72) -> str:
        """Resume comunicações suspeitas próximas ao timestamp da transação."""
        import re

        SUSPICIOUS_PATTERNS = [
            r"urgente|urgent|bloqueada|blocked|suspensa",
            r"clique aqui|click here|acesse agora|verify.*account",
            r"ganhou|you.*won|prêmio|prize",
            r"senha|password.*reset|confirm.*identity",
            r"bit\.ly|tinyurl|t\.co",
        ]

        def _extract_dt(text: str) -> datetime | None:
            m = re.search(r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})", text)
            if m:
                try:
                    return datetime.fromisoformat(m.group(1)).replace(tzinfo=None)
                except Exception:
                    pass
            return None

        def _is_suspicious(text: str) -> bool:
            t = text.lower()
            return any(re.search(p, t) for p in SUSPICIOUS_PATTERNS)

        alerts = []
        all_comms = (
            [(r.get("sms", ""), "SMS") for r in self.sms_by_citizen.get(citizen_id, [])] +
            [(r.get("mail", ""), "EMAIL") for r in self.mails_by_citizen.get(citizen_id, [])]
        )

        for text, kind in all_comms:
            if not _is_suspicious(text):
                continue
            dt = _extract_dt(text)
            if tx_time and dt:
                diff_h = (tx_time - dt).total_seconds() / 3600
                if 0 <= diff_h <= hours:
                    preview = text[:120].replace("\n", " ")
                    alerts.append(f"  [{kind} {diff_h:.0f}h antes] {preview}...")
            elif not tx_time:
                preview = text[:80].replace("\n", " ")
                alerts.append(f"  [{kind}] {preview}...")

        if not alerts:
            return "Nenhuma comunicação suspeita identificada na janela de 72h."

        return "\n".join(alerts[:5])  # máximo 5 alertas para não inflar o prompt
