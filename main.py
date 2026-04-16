"""
Pipeline principal — Reply Mirror Fraud Detection
Time: Solivus Hub

Uso:
    python main.py --train <pasta_treino> --eval <pasta_avaliacao> --output output.txt

Exemplo:
    python main.py \
        --train "TrainingDataset/The+Truman+Show+-+train/The Truman Show - train" \
        --eval  "TrainingDataset/The+Truman+Show+-+train/The Truman Show - train" \
        --output output.txt
"""

import argparse
import json
import os
import sys
import ulid
from dotenv import load_dotenv
from langfuse import Langfuse

from src.data_loader import DataLoader
from src.baseline_builder import BaselineBuilder
from src.context_builder import ContextBuilder
import src.tools.citizen_tools as ct
import src.tools.transaction_tools as tt
import src.tools.geo_tools as gt
import src.tools.communication_tools as comm_t
from src.agents.fraud_agents import (
    run_transaction_agent,
    run_geo_agent,
    run_comm_agent,
    run_orchestrator,
)

load_dotenv()


def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "Solivus-Hub")
    return f"{team}-{ulid.new().str}"


def init_langfuse() -> Langfuse:
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://challenges.reply.com/langfuse"),
    )


def analyze_transaction(tx_row: dict, context: dict, session_id: str, langfuse_client: Langfuse) -> dict:
    tx_json = json.dumps(tx_row, ensure_ascii=False, default=str)

    # Roda os 3 agentes especializados em paralelo (sequencial nesta versão)
    tx_report = run_transaction_agent(tx_json, context, session_id, langfuse_client)
    geo_report = run_geo_agent(tx_json, context, session_id, langfuse_client)
    comm_report = run_comm_agent(tx_json, context, session_id, langfuse_client)

    # Orquestrador agrega e decide
    decision = run_orchestrator(
        transaction_id=tx_row.get("transaction_id", ""),
        citizen_id=tx_row.get("sender_id", ""),
        tx_report=tx_report,
        geo_report=geo_report,
        comm_report=comm_report,
        context=context,
        session_id=session_id,
        langfuse_client=langfuse_client,
    )
    return decision


def run_pipeline(train_path: str, eval_path: str, output_path: str, verbose: bool = True):
    print(f"[1/4] Carregando dados de treino: {train_path}")
    train_loader = DataLoader(train_path)
    train_data = train_loader.load_all()

    print("[2/4] Construindo baseline comportamental...")
    builder = BaselineBuilder(train_data)
    baselines = builder.build()
    print(f"       Cidadãos no baseline: {list(baselines.keys())}")

    print(f"[3/4] Carregando dados de avaliação: {eval_path}")
    eval_loader = DataLoader(eval_path)
    eval_data = eval_loader.load_all()

    # Inicializa ferramentas com baseline de treino + dados de avaliação
    ct.init_tools(baselines, eval_data["transactions"])
    tt.init_tools(baselines)
    gt.init_tools(baselines, eval_data["locations"])
    comm_t.init_tools(baselines, eval_data["sms"], eval_data["mails"])

    # Instancia o ContextBuilder com dados de avaliação
    context_builder = ContextBuilder(
        baselines=baselines,
        transactions=eval_data["transactions"],
        sms=eval_data["sms"],
        mails=eval_data["mails"],
        sms_by_citizen=comm_t._SMS_BY_CITIZEN,
        mails_by_citizen=comm_t._MAILS_BY_CITIZEN,
    )

    langfuse_client = init_langfuse()
    session_id = generate_session_id()
    print(f"       Session ID: {session_id}")

    transactions = eval_data["transactions"]
    total = len(transactions)
    print(f"\n[4/4] Analisando {total} transações...")

    fraud_ids = []
    results = []

    for i, (_, row) in enumerate(transactions.iterrows(), 1):
        tx_id = row.get("transaction_id", "")
        sender = row.get("sender_id", "")

        # Pula transações de entidades externas (empregadores, etc.)
        if not sender or sender.startswith("EMP"):
            if verbose:
                print(f"  [{i}/{total}] {tx_id[:8]}... SKIP (entidade externa: {sender})")
            continue

        if verbose:
            print(f"  [{i}/{total}] {tx_id[:8]}... analisando {sender}...", end=" ", flush=True)

        tx_dict = row.where(row.notna(), other=None).to_dict()
        # Converte timestamp para string
        if tx_dict.get("timestamp") is not None:
            tx_dict["timestamp"] = str(tx_dict["timestamp"])

        context = context_builder.build(tx_dict)
        decision = analyze_transaction(tx_dict, context, session_id, langfuse_client)
        results.append({"transaction_id": tx_id, **decision})

        is_fraud = decision.get("decision") == "FRAUD"
        score = decision.get("final_score", decision.get("confidence", 0))

        if is_fraud:
            fraud_ids.append(tx_id)

        if verbose:
            status = "🚨 FRAUD" if is_fraud else "✅ LEGIT"
            print(f"{status} (score={score:.2f})")

    # Validação do output
    if len(fraud_ids) == 0:
        print("\n⚠️  Nenhuma fraude detectada — output seria INVÁLIDO. Forçando revisão...")
    elif len(fraud_ids) == total:
        print("\n⚠️  Todas marcadas como fraude — output seria INVÁLIDO.")

    # Grava output
    with open(output_path, "w", encoding="ascii") as f:
        for tx_id in fraud_ids:
            f.write(tx_id + "\n")

    langfuse_client.flush()

    print(f"\n{'='*50}")
    print(f"Transações analisadas : {total}")
    print(f"Fraudes detectadas    : {len(fraud_ids)}")
    print(f"Taxa de suspeição     : {len(fraud_ids)/total*100:.1f}%")
    print(f"Output salvo em       : {output_path}")
    print(f"Session ID Langfuse   : {session_id}")
    print(f"{'='*50}")

    return fraud_ids, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reply Mirror — Fraud Detection Pipeline")
    parser.add_argument("--train", required=True, help="Pasta do dataset de treino")
    parser.add_argument("--eval", required=True, help="Pasta do dataset de avaliação")
    parser.add_argument("--output", default="output.txt", help="Arquivo de saída com IDs fraudulentos")
    parser.add_argument("--quiet", action="store_true", help="Reduz verbosidade")
    args = parser.parse_args()

    run_pipeline(args.train, args.eval, args.output, verbose=not args.quiet)
