# 🛡️ InsuranceAI — Chatbot de Seguros

Protótipo de chatbot RAG para responder perguntas sobre seguros,
baseado no dataset InsuranceQA com 27.987 pares pergunta-resposta.

## Estrutura de pastas

```
insurance-chatbot/
├── app.py                        # Aplicação principal Streamlit
├── requirements.txt              # Dependências Python
├── .gitignore                    # Arquivos a ignorar no Git
├── .streamlit/
│   ├── config.toml               # Tema e configurações do Streamlit
│   └── secrets.toml              # 🔑 Chave da API (NÃO subir no Git)
└── data/
    └── insuranceqa_completo.csv  # Base de conhecimento (27.987 linhas)
```

## Configuração local

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/insurance-chatbot.git
cd insurance-chatbot
```

### 2. Crie um ambiente virtual
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Configure a chave da API
Edite `.streamlit/secrets.toml`:
```toml
OPENROUTER_API_KEY = "sk-or-v1-sua-chave-aqui"
```

Obtenha sua chave **gratuitamente** em: https://openrouter.ai/keys
(sem cartão de crédito — modelos com `:free` são 100% gratuitos)

### Modelos gratuitos disponíveis
No `app.py`, a variável `OPENROUTER_MODEL` controla o modelo usado.
O projeto usa `openrouter/free` por padrão para desenvolvimento e testes sem custo.
Para fixar um modelo específico, descomente uma das linhas no código.

## Como funciona o RAG

- A pergunta do usuário é traduzida para inglês antes da recuperação de contexto.
- A recuperação é baseada em palavras-chave, sem embeddings ou banco vetorial nesta versão.
- Quando nenhuma fonte confiável é encontrada, o app responde com uma mensagem controlada, reduzindo alucinações e evitando chamadas desnecessárias ao modelo de geração.
- Uma melhoria futura é usar embeddings com FAISS ou outra busca vetorial para recuperar contexto por similaridade semântica.

### 5. Adicione o CSV na pasta data/
Coloque o arquivo `insuranceqa_completo.csv` em `data/`.

### 6. Rode localmente
```bash
streamlit run app.py
```
Acesse: http://localhost:8501

## Deploy no Streamlit Cloud (gratuito)

1. Suba o projeto para o GitHub (sem o secrets.toml)
2. Acesse https://share.streamlit.io
3. Conecte seu repositório GitHub
4. Em **Advanced settings > Secrets**, adicione:
   ```
   OPENROUTER_API_KEY = "sk-or-v1-sua-chave-aqui"
   ```
5. Clique em **Deploy**

> ⚠️ O arquivo CSV precisa estar no repositório GitHub ou
> ser carregado via LFS para o deploy funcionar.
