"""
Agentes especializados para detecção de fraude — Reply Mirror
Usa create_agent do LangChain v1.2.15 com Few-shot Prompting + Chain-of-Thought.
"""

import json
import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langfuse import Langfuse, observe, propagate_attributes
from langfuse.langchain import CallbackHandler

from src.tools.citizen_tools import get_citizen_profile, get_transaction_history
from src.tools.transaction_tools import analyze_transaction_anomalies
from src.tools.geo_tools import check_geolocation
from src.tools.communication_tools import analyze_communications


def _build_model():
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=1500,
    )


def _extract_json(text: str) -> dict:
    """Extrai o primeiro bloco JSON de uma string de resposta."""
    try:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0:
            return json.loads(text[start:end])
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# TransactionAnalyzerAgent — Few-shot + análise estatística
# ---------------------------------------------------------------------------

TX_SYSTEM_PROMPT = """Você é um analista sênior de fraudes financeiras do sistema The Eye.

Sua tarefa é classificar a transação fornecida como FRAUDE ou LEGÍTIMA com base no histórico do cidadão.

METODOLOGIA — siga exatamente estes passos:
1. Leia o perfil e histórico do cidadão fornecido no contexto
2. Compare a transação atual com os exemplos legítimos e suspeitos fornecidos
3. Use a ferramenta analyze_transaction_anomalies para obter sinais estatísticos
4. Identifique especificamente o que desvia do padrão normal

Responda APENAS com um JSON válido:
{
  "transaction_id": "...",
  "alerts": ["desvio 1", "desvio 2"],
  "suspicion_score": 0.0,
  "reasoning": "explicação em 1-2 frases baseada nos exemplos e no histórico"
}"""


def run_transaction_agent(transaction_json: str, context: dict,
                          session_id: str, langfuse_client: Langfuse) -> dict:
    model = _build_model()
    tools = [get_transaction_history, analyze_transaction_anomalies]
    agent = create_agent(model=model, tools=tools, system_prompt=TX_SYSTEM_PROMPT)

    # Monta prompt com few-shot examples + histórico rico
    prompt = f"""PERFIL DO CIDADÃO:
{context.get('citizen_profile', 'Não disponível')}

HISTÓRICO RECENTE (5 transações antes desta):
{context.get('recent_history', 'Não disponível')}

EXEMPLOS DE REFERÊNCIA — TRANSAÇÕES LEGÍTIMAS DESTE CIDADÃO:
{context.get('few_shot_legit', 'Não disponível')}

EXEMPLOS DE REFERÊNCIA — PADRÕES SUSPEITOS IDENTIFICADOS:
{context.get('few_shot_suspicious', 'Nenhum identificado')}

COMUNICAÇÕES PRÓXIMAS (72h antes):
{context.get('nearby_communications', 'Nenhuma suspeita')}

TRANSAÇÃO A ANALISAR:
{transaction_json}

Com base nos exemplos e no histórico acima, analise se esta transação é FRAUDE ou LEGÍTIMA."""

    @observe()
    def _run(session_id, prompt_text):
        with propagate_attributes(session_id=session_id):
            handler = CallbackHandler()
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt_text)]},
                config={"callbacks": [handler]},
            )
            return result["messages"][-1].content

    raw = _run(session_id, prompt)
    return _extract_json(raw) or {"alerts": [], "suspicion_score": 0.0, "raw": raw}


# ---------------------------------------------------------------------------
# GeoValidatorAgent — validação geográfica com contexto
# ---------------------------------------------------------------------------

GEO_SYSTEM_PROMPT = """Você é o GeoValidatorAgent do sistema antifraude The Eye.

Sua função: validar se a localização geográfica da transação é consistente com o histórico do cidadão.

Use a ferramenta check_geolocation com os dados da transação e responda APENAS com JSON:
{
  "transaction_id": "...",
  "alerts": ["alerta geo"],
  "suspicion_score": 0.0,
  "reasoning": "1-2 frases sobre consistência geográfica"
}"""


def run_geo_agent(transaction_json: str, context: dict,
                  session_id: str, langfuse_client: Langfuse) -> dict:
    model = _build_model()
    tools = [get_citizen_profile, check_geolocation]
    agent = create_agent(model=model, tools=tools, system_prompt=GEO_SYSTEM_PROMPT)

    prompt = f"""PERFIL DO CIDADÃO (inclui cidade de residência e coordenadas GPS médias):
{context.get('citizen_profile', 'Não disponível')}

TRANSAÇÃO A VALIDAR GEOGRAFICAMENTE:
{transaction_json}"""

    @observe()
    def _run(session_id, prompt_text):
        with propagate_attributes(session_id=session_id):
            handler = CallbackHandler()
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt_text)]},
                config={"callbacks": [handler]},
            )
            return result["messages"][-1].content

    raw = _run(session_id, prompt)
    return _extract_json(raw) or {"alerts": [], "suspicion_score": 0.0, "raw": raw}


# ---------------------------------------------------------------------------
# CommunicationAnalyzerAgent — análise de engenharia social com contexto
# ---------------------------------------------------------------------------

COMM_SYSTEM_PROMPT = """Você é o CommunicationAnalyzerAgent do sistema antifraude The Eye.

Sua função: avaliar se o cidadão foi alvo de engenharia social ou phishing antes desta transação.

Use a ferramenta analyze_communications e responda APENAS com JSON:
{
  "transaction_id": "...",
  "alerts": ["alerta comm"],
  "suspicion_score": 0.0,
  "reasoning": "1-2 frases sobre risco de manipulação"
}"""


def run_comm_agent(transaction_json: str, context: dict,
                   session_id: str, langfuse_client: Langfuse) -> dict:
    model = _build_model()
    tools = [get_citizen_profile, analyze_communications]
    agent = create_agent(model=model, tools=tools, system_prompt=COMM_SYSTEM_PROMPT)

    prompt = f"""PERFIL DO CIDADÃO (inclui vulnerabilidade a phishing):
{context.get('citizen_profile', 'Não disponível')}

COMUNICAÇÕES SUSPEITAS PRÉ-IDENTIFICADAS NA JANELA DE 72H:
{context.get('nearby_communications', 'Nenhuma identificada')}

TRANSAÇÃO A ANALISAR:
{transaction_json}"""

    @observe()
    def _run(session_id, prompt_text):
        with propagate_attributes(session_id=session_id):
            handler = CallbackHandler()
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt_text)]},
                config={"callbacks": [handler]},
            )
            return result["messages"][-1].content

    raw = _run(session_id, prompt)
    return _extract_json(raw) or {"alerts": [], "suspicion_score": 0.0, "raw": raw}


# ---------------------------------------------------------------------------
# OrchestratorAgent — Chain-of-Thought para decisão final
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """Você é o OrchestratorAgent do sistema antifraude The Eye.

Você recebe relatórios de três agentes especializados e deve tomar a decisão final.

RACIOCINE EM VOZ ALTA seguindo exatamente estes 5 passos antes do veredito:

Passo 1 — PERFIL DO CIDADÃO: o que o histórico diz sobre os hábitos normais desta pessoa?
Passo 2 — ANOMALIAS DA TRANSAÇÃO: o que desvia do perfil? (valor, horário, destinatário, método)
Passo 3 — CONTEXTO EXTERNO: houve comunicações suspeitas? O GPS é consistente?
Passo 4 — CONVERGÊNCIA: quantos agentes sinalizam risco? Os sinais se reforçam mutuamente?
Passo 5 — VEREDITO FINAL: FRAUD ou LEGIT com justificativa baseada nos passos anteriores

REGRAS DE DECISÃO:
- Classifique como FRAUD se score ponderado >= 0.19 (TX=40%, GEO=30%, COMM=30%)
- FRAUD exige sinais concretos — não classifique por dúvida sem evidência
- Custo equilibrado: falso positivo e falso negativo ambos têm custo real
- Na dúvida sem evidências convergentes: prefira LEGIT

Responda em JSON:
{
  "transaction_id": "...",
  "decision": "FRAUD|LEGIT",
  "final_score": 0.0,
  "confidence": 0.0,
  "step1_profile": "resumo do perfil",
  "step2_anomalies": "anomalias encontradas",
  "step3_context": "comunicações e geo",
  "step4_convergence": "convergência dos agentes",
  "reasoning": "justificativa final concisa"
}"""


def run_orchestrator(
    transaction_id: str,
    citizen_id: str,
    tx_report: dict,
    geo_report: dict,
    comm_report: dict,
    context: dict,
    session_id: str,
    langfuse_client: Langfuse,
) -> dict:
    model = _build_model()
    agent = create_agent(model=model, tools=[], system_prompt=ORCHESTRATOR_SYSTEM_PROMPT)

    prompt = f"""Transaction ID: {transaction_id}
Citizen ID: {citizen_id}

=== PERFIL E HISTÓRICO DO CIDADÃO ===
{context.get('citizen_profile', 'Não disponível')}

=== HISTÓRICO RECENTE ===
{context.get('recent_history', 'Não disponível')}

=== EXEMPLOS LEGÍTIMOS (few-shot) ===
{context.get('few_shot_legit', 'Não disponível')}

=== EXEMPLOS SUSPEITOS (few-shot) ===
{context.get('few_shot_suspicious', 'Nenhum')}

=== COMUNICAÇÕES PRÓXIMAS ===
{context.get('nearby_communications', 'Nenhuma suspeita')}

=== RELATÓRIO — TransactionAnalyzerAgent ===
Score: {tx_report.get('suspicion_score', 0.0)}
Alertas: {tx_report.get('alerts', [])}
Reasoning: {tx_report.get('reasoning', '')}

=== RELATÓRIO — GeoValidatorAgent ===
Score: {geo_report.get('suspicion_score', 0.0)}
Alertas: {geo_report.get('alerts', [])}
Reasoning: {geo_report.get('reasoning', '')}

=== RELATÓRIO — CommunicationAnalyzerAgent ===
Score: {comm_report.get('suspicion_score', 0.0)}
Alertas: {comm_report.get('alerts', [])}
Reasoning: {comm_report.get('reasoning', '')}

Siga os 5 passos do Chain-of-Thought e dê o veredito final."""

    @observe()
    def _run(session_id, prompt_text):
        with propagate_attributes(session_id=session_id):
            handler = CallbackHandler()
            result = agent.invoke(
                {"messages": [HumanMessage(content=prompt_text)]},
                config={"callbacks": [handler]},
            )
            return result["messages"][-1].content

    raw = _run(session_id, prompt)
    result = _extract_json(raw) or {"decision": "LEGIT", "final_score": 0.0, "raw": raw}

    # Threshold determinístico — safety net independente do LLM
    tx_score   = float(tx_report.get("suspicion_score", 0.0))
    geo_score  = float(geo_report.get("suspicion_score", 0.0))
    comm_score = float(comm_report.get("suspicion_score", 0.0))
    weighted   = tx_score * 0.40 + geo_score * 0.30 + comm_score * 0.30

    result["final_score"] = round(weighted, 3)
    result["decision"]    = "FRAUD" if weighted >= 0.19 else "LEGIT"

    return result
