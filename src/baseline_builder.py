"""
Constrói o perfil comportamental (baseline) de cada cidadão a partir do
dataset de treino. O baseline é usado pelos agentes para detectar desvios.

IMPORTANTE: usa dois passos para evitar contaminação do baseline por fraudes:
  1. Remove outliers estatísticos (z-score > 2 por tipo) antes de computar o perfil
  2. Constrói o perfil apenas com transações consideradas "normais"

Uso:
    from src.baseline_builder import BaselineBuilder
    builder = BaselineBuilder(data)
    baselines = builder.build()
    profile = baselines["GRSC-KRLH-807-DIE-1"]
"""

import numpy as np
import pandas as pd


class BaselineBuilder:
    def __init__(self, data: dict):
        self.transactions: pd.DataFrame = data["transactions"]
        self.users: list[dict] = data["users"]
        self.locations: pd.DataFrame = data["locations"]

    def _citizen_id_from_iban(self, iban: str) -> str | None:
        mask = self.transactions["sender_iban"] == iban
        ids = self.transactions.loc[mask, "sender_id"].dropna().unique()
        return ids[0] if len(ids) > 0 else None

    def build(self) -> dict:
        """Retorna um dict {citizen_id: profile_dict} com o baseline de cada cidadão."""
        baselines = {}
        citizen_ids = self.transactions["sender_id"].dropna().unique()

        for citizen_id in citizen_ids:
            if citizen_id.startswith("EMP"):
                continue
            profile = self._build_profile(citizen_id)
            if profile:
                baselines[citizen_id] = profile

        return baselines

    def _remove_outliers(self, txs: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers estatísticos por tipo de transação usando z-score.
        Transações com z-score > 2 no valor são excluídas do baseline.
        Também remove transações em horário de madrugada (0h–5h) que sejam únicas
        — padrão típico de fraude automatizada.
        """
        clean_rows = []

        for tx_type, group in txs.groupby("transaction_type"):
            amounts = group["amount"].dropna()

            if len(amounts) < 3:
                # Poucos dados: mantém tudo, não há base estatística suficiente
                clean_rows.append(group)
                continue

            mean = amounts.mean()
            std = amounts.std()

            if std == 0:
                clean_rows.append(group)
                continue

            z_scores = ((group["amount"] - mean) / std).abs()
            normal = group[z_scores <= 2.0]

            # Remove também transações de madrugada (0h–5h) isoladas
            # — se for o único horário nesse range, é suspeito
            hours = normal["timestamp"].dt.hour
            madrugada = normal[hours < 5]
            if len(madrugada) == 1:
                # Único registro na madrugada — exclui do baseline
                normal = normal[hours >= 5]

            clean_rows.append(normal)

        if not clean_rows:
            return txs

        return pd.concat(clean_rows).sort_values("timestamp")

    def _build_profile(self, citizen_id: str) -> dict | None:
        txs_all = self.transactions[self.transactions["sender_id"] == citizen_id].copy()
        if txs_all.empty:
            return None

        # Usa apenas transações "limpas" para construir o baseline
        txs = self._remove_outliers(txs_all)
        user_info = self._find_user(citizen_id)

        type_stats = {}
        for tx_type, group in txs.groupby("transaction_type"):
            amounts = group["amount"].dropna()
            hours = group["timestamp"].dt.hour
            type_stats[tx_type] = {
                "count": len(group),
                "amount_mean": float(amounts.mean()),
                "amount_std": float(amounts.std()) if len(amounts) > 1 else 0.0,
                "amount_min": float(amounts.min()),
                "amount_max": float(amounts.max()),
                "typical_hours": hours.tolist(),
            }

        # Destinatários, IBANs e métodos baseados APENAS em transações limpas
        known_recipients = set(txs["recipient_id"].dropna().unique().tolist())
        known_recipient_ibans = set(txs["recipient_iban"].dropna().unique().tolist())
        known_payment_methods = set(txs["payment_method"].dropna().unique().tolist())

        in_person = txs[txs["transaction_type"] == "in-person payment"]
        known_locations = set(in_person["location"].dropna().unique().tolist())

        salary_days, rent_days = [], []
        for _, row in txs.iterrows():
            desc = str(row.get("description", "")).lower()
            if "salary" in desc:
                salary_days.append(row["timestamp"].day)
            elif "rent" in desc:
                rent_days.append(row["timestamp"].day)

        balances = txs.sort_values("timestamp")["balance_after"].dropna()

        loc_data = self.locations[self.locations["biotag"] == citizen_id]
        home_lat = float(loc_data["lat"].mean()) if not loc_data.empty else None
        home_lng = float(loc_data["lng"].mean()) if not loc_data.empty else None

        # Registra quantas transações foram excluídas como outliers
        excluded_count = len(txs_all) - len(txs)

        return {
            "citizen_id": citizen_id,
            "user_info": user_info,
            "transaction_types": list(txs["transaction_type"].unique()),
            "type_stats": type_stats,
            "known_recipients": list(known_recipients),
            "known_recipient_ibans": list(known_recipient_ibans),
            "known_payment_methods": list(known_payment_methods),
            "known_locations": list(known_locations),
            "typical_salary_days": salary_days,
            "typical_rent_days": rent_days,
            "balance_mean": float(balances.mean()) if not balances.empty else 0.0,
            "balance_std": float(balances.std()) if len(balances) > 1 else 0.0,
            "home_lat": home_lat,
            "home_lng": home_lng,
            "total_tx_count": len(txs),
            "outliers_excluded": excluded_count,
        }

    def _find_user(self, citizen_id: str) -> dict | None:
        iban_row = self.transactions[
            self.transactions["sender_id"] == citizen_id
        ]["sender_iban"].dropna()

        if iban_row.empty:
            return None

        iban = iban_row.iloc[0]
        for user in self.users:
            if user.get("iban") == iban:
                return user
        return None
