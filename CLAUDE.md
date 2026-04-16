# Hackathon Challenges IA — Projeto

## Contexto do Projeto
Este é o projeto de desenvolvimento da solução para o Hackathon da Reply (Challenges IA).
O time é **Solivus Hub**.

## Stack Técnica
- **Framework de agentes:** LangChain + LangGraph (`create_agent` de `langchain.agents`)
- **Modelo LLM:** `gpt-4o-mini` via OpenRouter (`https://openrouter.ai/api/v1`)
- **Observabilidade:** Langfuse **v3** — `langfuse>=3,<4` (v4 NÃO suportado pelo Hackathon)
- **Linguagem:** Python 3.13.3
- **Ambiente:** venv isolado

## Configuração do Ambiente

### Criar e ativar venv
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Variáveis de ambiente (.env)
```
OPENROUTER_API_KEY=...
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=...
TEAM_NAME=Solivus-Hub
```

## Padrões do Projeto

### Criar agentes (LangChain v1.2+)
```python
from langchain.agents import create_agent

agent = create_agent(
    model=model,
    tools=[...],
    system_prompt="...",
)
response = agent.invoke({"messages": [HumanMessage("...")]})
print(response["messages"][-1].content)
```

### Criar ferramentas
```python
from langchain.tools import tool

@tool
def minha_ferramenta(param: str) -> str:
    """Descrição clara — o agente lê isso para decidir quando usar."""
    return resultado
```

### Rastreamento Langfuse v3
```python
from langfuse import Langfuse, observe, propagate_attributes
from langfuse.langchain import CallbackHandler
import ulid

langfuse_client = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

def generate_session_id():
    return f"{os.getenv('TEAM_NAME')}-{ulid.new().str}"

@observe()
def run_llm_call(session_id, model, prompt):
    with propagate_attributes(session_id=session_id):
        handler = CallbackHandler()
        response = model.invoke(
            [HumanMessage(content=prompt)],
            config={"callbacks": [handler]}
        )
        return response.content

# Sempre fazer flush após as chamadas
langfuse_client.flush()
```

## Armadilhas Conhecidas

- `create_agent` usa `system_prompt=` (não `prompt=`)
- `propagate_attributes` é **função importada** de `langfuse`, NÃO método do cliente
- Sempre fazer `langfuse_client.flush()` após chamadas
- `response["messages"][-1].content` para pegar a última resposta do agente
- O `temperature` deve ser baixo (0.1) quando o agente usa ferramentas
- Reiniciar o kernel do Jupyter após alterar o `.env`
- **NÃO usar Langfuse v4** — o Hackathon só suporta v3 (`langfuse>=3,<4`)

## Tutoriais de Referência
Os tutoriais completos estão em:
`/home/henrique/Documentos/Hackathon_Challenges_IA/`
- `01-agent_creation.ipynb` — criação básica de agente
- `02-adding_tools.ipynb` — ferramentas com @tool
- `03-multi_agent.ipynb` — sistema multi-agente (orquestrador + especialistas)
- `04-resource_management.ipynb` — rastreamento com Langfuse v4

## Desafio — Reply Mirror (Hackathon Reply, 16 Abril 2026)

### Problema
Detectar transações financeiras fraudulentas num sistema onde os padrões de fraude **evoluem ao longo do tempo**. Os fraudadores (Mirror Hackers) mudam táticas constantemente: novos comerciantes, horários, regiões geográficas, valores e sequências comportamentais. Modelos estáticos falham — o sistema precisa aprender e se adaptar.

### Estrutura do Desafio
- **5 níveis progressivos**, cada um com dataset de treino + dataset de avaliação
- A dificuldade e a complexidade dos padrões de fraude aumentam a cada nível
- **Regra crítica: apenas a primeira submissão por nível é aceita e considerada final**
- Top teams podem ser reavaliados em datasets inéditos após o desafio

### Dados de Entrada (por nível)

| Dataset | Campos relevantes |
|---|---|
| `Transactions.csv` | Transaction ID, Sender ID, Recipient ID, Transaction Type (bank transfer / in-person payment / e-commerce / direct debit / withdrawal), Amount, Location (só in-person), Payment Method (debit card / mobile device / smartwatch / GooglePay / PayPal), Sender IBAN, Recipient IBAN (só bank transfers), Balance, Timestamp |
| `Locations` | BioTag (citizen ID), Datetime, Lat, Lng |
| `Users` | Dados pessoais dos cidadãos |
| `Conversations` | User ID, SMS (thread de mensagens) |
| `Messages` | mail (thread de e-mails) |

### Formato de Output
Arquivo ASCII com um `Transaction ID` por linha — apenas os IDs suspeitos de fraude.

**Submissão inválida se:**
- Nenhuma transação reportada
- Todas as transações reportadas
- Menos de 15% das fraudes reais identificadas

### Scoring
Score composto de duas dimensões:
1. **Acurácia** — equilíbrio entre detectar fraudes (reduzir falsos negativos) e não bloquear transações legítimas (reduzir falsos positivos). Custo assimétrico: FP = dano econômico/reputacional; FN = dano financeiro direto
2. **Custo, velocidade e eficiência** — recompensa arquiteturas otimizadas com baixa latência e baixo custo operacional (tokens, infraestrutura)

### Requisitos Obrigatórios
- **Somente soluções baseadas em agentes são aceitas**
- Abordagens puramente determinísticas são penalizadas
- Entregar código-fonte + instruções de execução + lista de dependências
- O sistema deve ser adaptável entre níveis — não pode ser retreinado do zero a cada submissão

### Objetivo do Sistema de Agentes
Construir um sistema de **agentes cooperativos** capaz de:
- Detectar comportamentos fraudulentos que evoluem e se misturam com transações legítimas
- Antecipar novos padrões usando memória de interações passadas
- Responder em tempo real a mudanças sem degradar performance
- Manter baixa taxa de falsos positivos

### Custo Assimétrico — Diretriz de Design
Ambos os erros têm custo real e devem ser minimizados:
- **Falso positivo** (bloquear legítima) → dano econômico e reputacional ao cliente
- **Falso negativo** (deixar fraude passar) → dano financeiro direto

O sistema deve classificar como FRAUD **somente** quando houver sinais concretos e convergentes de múltiplos agentes. Na dúvida sem evidências sólidas, prefere LEGIT. O score final usa pesos: TX=40%, GEO=30%, COMM=30% com threshold >= 0.19.
