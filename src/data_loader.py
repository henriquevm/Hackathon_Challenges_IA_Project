"""
Carrega e normaliza todos os datasets do desafio Reply Mirror.

Uso:
    from src.data_loader import DataLoader
    loader = DataLoader("caminho/para/pasta/do/nivel")
    data = loader.load_all()
"""

import json
import os
import pandas as pd
from pathlib import Path


class DataLoader:
    def __init__(self, dataset_path: str):
        self.path = Path(dataset_path)

    def load_transactions(self) -> pd.DataFrame:
        filepath = self.path / "transactions.csv"
        df = pd.read_csv(filepath)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
        df["balance_after"] = pd.to_numeric(df["balance_after"], errors="coerce")
        return df

    def load_users(self) -> list[dict]:
        filepath = self.path / "users.json"
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def load_locations(self) -> pd.DataFrame:
        filepath = self.path / "locations.json"
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def load_sms(self) -> list[dict]:
        filepath = self.path / "sms.json"
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def load_mails(self) -> list[dict]:
        filepath = self.path / "mails.json"
        with open(filepath, encoding="utf-8") as f:
            return json.load(f)

    def load_all(self) -> dict:
        return {
            "transactions": self.load_transactions(),
            "users": self.load_users(),
            "locations": self.load_locations(),
            "sms": self.load_sms(),
            "mails": self.load_mails(),
        }
