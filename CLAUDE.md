# Hackathon Challenges IA — Projeto

## Contexto do Projeto
Este é o projeto de desenvolvimento da solução para o Hackathon da Reply (Challenges IA).
O time é **Solivus Hub**.

## Stack Técnica
- **Framework de agentes:** LangChain + LangGraph (`create_agent` de `langchain.agents`)
- **Modelo LLM:** `gpt-4o-mini` via OpenRouter (`https://openrouter.ai/api/v1`)
- **Observabilidade:** Langfuse v4 (rastreamento de tokens, custos e sessões)
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

### Rastreamento Langfuse v4
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
- `propagate_attributes` é função importada de `langfuse`, NÃO método do cliente
- Sempre fazer `langfuse_client.flush()` após chamadas
- `response["messages"][-1].content` para pegar a última resposta do agente
- O `temperature` deve ser baixo (0.1) quando o agente usa ferramentas matemáticas
- Reiniciar o kernel do Jupyter após alterar o `.env`

## Tutoriais de Referência
Os tutoriais completos estão em:
`/home/henrique/Documentos/Hackathon_Challenges_IA/`
- `01-agent_creation.ipynb` — criação básica de agente
- `02-adding_tools.ipynb` — ferramentas com @tool
- `03-multi_agent.ipynb` — sistema multi-agente (orquestrador + especialistas)
- `04-resource_management.ipynb` — rastreamento com Langfuse v4

## Desafio
<!-- PREENCHER: descreva aqui o problema que o Hackathon pede para resolver -->
