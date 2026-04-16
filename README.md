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
