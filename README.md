# Hackathon Challenges IA — Solivus Hub

Solução do time **Solivus Hub** para o Hackathon Reply (Challenges IA).

## Stack Técnica

| Componente | Tecnologia |
|---|---|
| Framework de agentes | LangChain + LangGraph |
| Modelo LLM | `gpt-4o-mini` via OpenRouter |
| Observabilidade | Langfuse v4 |
| Linguagem | Python 3.13 |

## Configuração do Ambiente

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd Hackathon_Challenges_IA_Project
```

### 2. Crie o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

```bash
cp .env.example .env
```

Edite o `.env` com suas credenciais:
- `OPENROUTER_API_KEY` — chave da API OpenRouter
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — credenciais Langfuse
- `LANGFUSE_HOST` — host do Langfuse (fornecido pelo Hackathon)
- `TEAM_NAME` — nome do time no formato `Nome-Do-Time`

### 5. Inicie o Jupyter

```bash
jupyter notebook
```

## Estrutura do Projeto

```
Hackathon_Challenges_IA_Project/
├── .env.example              # Template de variáveis de ambiente
├── requirements.txt          # Dependências Python
├── 04-resource_management.ipynb  # Rastreamento com Langfuse v4
└── CLAUDE.md                 # Contexto do projeto para Claude Code
```

## Rastreamento de Custos

Todas as chamadas ao LLM são rastreadas via Langfuse com session IDs no formato:

```
{TEAM_NAME}-{ULID}
```

Isso permite que a organização do Hackathon contabilize o uso de tokens por time.


## Derscrição do Readme do Hackathon

How to correctly trace your submission with Langfuse

This project shows how to set up Langfuse tracing for your LangChain agent so that
token usage, costs, and latency are automatically tracked for the challenge.

What this script does:
- Configures a LangChain model via OpenRouter
- Initializes Langfuse with @observe() and CallbackHandler
- Generates a unique session ID in the format {TEAM_NAME}-{ULID}
- Sends 3 traced LLM calls grouped under a single session ID
- Passes session_id via metadata (config={"metadata": {"langfuse_session_id": ...}})
- Flushes traces to Langfuse after the run

How to view your traces:
You can see your tracing details on the Langfuse dashboard in the platform page.
The dashboard is associated with your team. It is not updated in real time,
so there may be a few minutes of delay before the latest traces appear.

Recommended setup:
- Python 3.13 (suggested)
- Langfuse SDK v3 (v4 is not fully supported and may cause unexpected issues)

For the full tutorial and additional examples, refer to
"Resource Management & Toolkit for the Challenge" in the Learn & Train section.
