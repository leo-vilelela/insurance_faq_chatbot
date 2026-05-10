import streamlit as st
import pandas as pd
from openai import OpenAI
import os
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsuranceAI · Assistente de Seguros",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --surface2: #222536;
    --border: #2e3150;
    --accent: #4f8ef7;
    --accent2: #7c6af5;
    --text: #e8eaf0;
    --muted: #7b7f96;
    --success: #3ecf8e;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }

.stChatMessage {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}

.stChatMessage[data-testid*="user"] {
    background-color: var(--surface2) !important;
    border-color: var(--accent2) !important;
}

.stChatInputContainer {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

.stChatInputContainer textarea {
    color: var(--text) !important;
    background: transparent !important;
}

div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'DM Serif Display', serif;
    font-size: 2rem !important;
}

div[data-testid="stMetricLabel"] { color: var(--muted) !important; }

.stSelectbox > div > div {
    background-color: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white;
    margin-bottom: 8px;
}

.source-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.85rem;
}

.source-card .src-category {
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.source-card .src-question {
    color: var(--muted);
    font-size: 0.82rem;
    margin-top: 4px;
    font-style: italic;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #4f8ef7, #7c6af5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin-bottom: 1.5rem;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

hr { border-color: var(--border) !important; }

.stSpinner > div { color: var(--accent) !important; }

/* hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando base de conhecimento...")
def load_knowledge_base():
    import csv
    data_dir = Path(__file__).parent / "data"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        st.error("❌ Nenhum CSV encontrado em `data/`. Adicione ao menos um arquivo.")
        st.stop()

    frames = []
    for csv_path in csv_files:
        print(f"Carregando {csv_path}...")
        try:
            # Usa engine='python' para melhor tolerância com CSVs mal-formatados
            df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip', encoding='utf-8')
            
            # aceita CSVs que tenham ao menos as colunas obrigatórias
            if {"pergunta", "resposta"}.issubset(df.columns):
                if "categoria" not in df.columns:
                    df["categoria"] = csv_path.stem  # usa o nome do arquivo como categoria
                frames.append(df)
            else:
                st.warning(f"⚠️ '{csv_path.name}' ignorado — precisa ter colunas 'pergunta' e 'resposta'.")
        except Exception as e:
            st.warning(f"⚠️ Erro ao ler '{csv_path.name}': {e}")

    if not frames:
        st.error("❌ Nenhum CSV válido encontrado. Verifique o formato dos arquivos.")
        st.stop()

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["pergunta", "resposta"])
    df = df.drop_duplicates(subset=["pergunta", "resposta"])
    return df


@st.cache_data(show_spinner=False)
def get_categories(df):
    return sorted(df["categoria"].unique().tolist())


def retrieve_context(df, query: str, category_filter: str, top_k: int = 5) -> list[dict]:
    """Simple keyword-based retrieval (works without embeddings)."""
    if category_filter != "Todas":
        subset = df[df["categoria"] == category_filter]
    else:
        subset = df

    query_lower = query.lower()
    keywords = [w for w in query_lower.split() if len(w) > 3]

    if not keywords:
        return subset.sample(min(top_k, len(subset))).to_dict("records")

    def score(row):
        text = (row["pergunta"] + " " + row["resposta"]).lower()
        return sum(text.count(kw) for kw in keywords)

    subset = subset.copy()
    subset["_score"] = subset.apply(score, axis=1)
    top = subset[subset["_score"] > 0].nlargest(top_k, "_score")

    if top.empty:
        top = subset.sample(min(top_k, len(subset)))

    return top.drop(columns=["_score"]).to_dict("records")


def build_prompt(query: str, context: list[dict]) -> str:
    ctx_text = ""
    for i, item in enumerate(context, 1):
        ctx_text += f"\n---\nCategoria: {item['categoria']}\nPergunta similar: {item['pergunta']}\nResposta: {item['resposta']}\n"

    return f"""Você é um assistente especializado em seguros. Use as informações da base de conhecimento abaixo para responder à pergunta do usuário de forma clara, precisa e em português do Brasil.

BASE DE CONHECIMENTO:
{ctx_text}

INSTRUÇÕES:
- Responda sempre em português do Brasil
- Seja claro e objetivo
- NÃO cite números de referência como [1], [2], [3] na sua resposta
- NÃO mencione as fontes diretamente no texto
- Se a base não cobrir completamente a pergunta, diga o que sabe e recomende consultar um corretor
- Não invente informações
- Use linguagem acessível, sem jargões desnecessários

PERGUNTA DO USUÁRIO: {query}"""


def clean_response(response: str) -> str:
    """Remove referências numeradas como [1], [2], [3], etc. da resposta."""
    import re
    # Remove padrões como [1], [2], [123], etc.
    cleaned = re.sub(r'\[\d+\]', '', response)
    # Remove espaços extras deixados pela remoção
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def translate_to_english(query: str, client: OpenAI) -> str:
    """Traduz a query para inglês para melhorar a busca no CSV."""
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            max_tokens=100,
            messages=[{
                "role": "user",
                "content": (
                    f"Translate the following question to English. "
                    f"Return ONLY the translated question, nothing else.\n\n{query}"
                )
            }]
        )
        msg = response.choices[0].message

        # Tenta content normal primeiro
        if msg.content:
            print(f"Content: '{msg.content.strip()}'")
            return msg.content.strip()

        # Modelos de raciocínio (DeepSeek-R1, etc.) às vezes só preenchem reasoning
        # Extrai a segunda frase entre aspas do reasoning como fallback
        if hasattr(msg, "reasoning") and msg.reasoning:
            import re
            matches = re.findall(r'"([^"]+\?)"', msg.reasoning)
            if len(matches) > 1:
                print(f"Matches: '{matches[1].strip()}'")
                return matches[1].strip()
            elif matches:
                print(f"Matches: '{matches[0].strip()}'")
                return matches[0].strip()
    except Exception as e:
        print(f"Translation error: {e}")
        return query

# ── OpenRouter client (OpenAI-compatible) ─────────────────────────────────────
OPENROUTER_MODEL = "openrouter/auto"   # router automático — escolhe o melhor modelo gratuito
# Alternativas gratuitas específicas (descomente para fixar um modelo):
# OPENROUTER_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
# OPENROUTER_MODEL = "deepseek/deepseek-r1:free"
# OPENROUTER_MODEL = "qwen/qwen3-235b-a22b:free"
# OPENROUTER_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"

@st.cache_resource
def get_client():
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        st.error("❌ OPENROUTER_API_KEY não configurada. Veja o README para instruções.")
        st.stop()
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


# ── Main UI ───────────────────────────────────────────────────────────────────
df = load_knowledge_base()
categories = get_categories(df)
client = get_client()

# Sidebar
with st.sidebar:
    st.markdown('<p class="hero-title" style="font-size:1.5rem">🛡️ InsuranceAI</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Filtrar por categoria**")
    category_filter = st.selectbox(
        "Categoria",
        ["Todas"] + categories,
        label_visibility="collapsed"
    )

    st.markdown("**Contexto recuperado**")
    top_k = st.slider("Nº de referências", 3, 10, 5, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**📊 Base de dados**")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Pares Q&A", f"{len(df):,}".replace(",", "."))
    with col2:
        st.metric("Categorias", len(categories))

    st.markdown("---")
    if st.button("🗑️ Limpar conversa", use_container_width=True):
        st.session_state.messages = []
        st.session_state.contexts = []
        st.rerun()

    st.markdown('<p style="color:var(--muted);font-size:0.75rem;margin-top:1rem">Protótipo · InsuranceQA Dataset</p>', unsafe_allow_html=True)

# Main area
st.markdown('<p class="hero-title">Assistente de Seguros</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Tire suas dúvidas sobre seguros com base em milhares de respostas especializadas.</p>', unsafe_allow_html=True)

# Initialize state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "contexts" not in st.session_state:
    st.session_state.contexts = []

def process_user_question(prompt: str):
    """Processa uma pergunta do usuário: recupera contexto e gera resposta."""
    query_for_search = translate_to_english(prompt, client)
    # Retrieve
    context = retrieve_context(df, query_for_search, category_filter, top_k)
    st.session_state.contexts.append(context)

    # Generate
    with st.spinner("Consultando base de conhecimento..."):
        full_prompt = build_prompt(prompt, context)
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            max_tokens=1000,
            messages=[{"role": "user", "content": full_prompt}]
        )
        answer = response.choices[0].message.content
        answer = clean_response(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()


# Suggested questions
if not st.session_state.messages:
    st.markdown("**💡 Sugestões para começar:**")
    suggestions = [
        "O que o seguro de vida cobre?",
        "Como funciona a franquia do seguro auto?",
        "Qual a diferença entre seguro saúde HMO e PPO?",
        "Quando devo contratar seguro de cuidados de longo prazo?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        if cols[i % 2].button(suggestion, use_container_width=True, key=f"sug_{i}"):
            st.session_state.messages.append({"role": "user", "content": suggestion})
            print(f"Sugestão selecionada: {suggestion}")
            with st.spinner("⏳ Processando sua pergunta..."):
                query_for_search = translate_to_english(suggestion, client)
                context = retrieve_context(df, query_for_search, category_filter, top_k)
                st.session_state.contexts.append(context)
                full_prompt = build_prompt(suggestion, context)
                response = client.chat.completions.create(
                    model=OPENROUTER_MODEL,
                    max_tokens=1000,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                answer = response.choices[0].message.content
                answer = clean_response(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()


# Chat history
chat_col, ctx_col = st.columns([2, 1])

with chat_col:
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🛡️"):
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("Digite sua pergunta sobre seguros..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        process_user_question(prompt)

with ctx_col:
    if st.session_state.contexts:
        st.markdown("**📚 Fontes consultadas**")
        latest_ctx = st.session_state.contexts[-1]
        for item in latest_ctx:
            st.markdown(f"""
            <div class="source-card">
                <div class="src-category">🏷️ {item['categoria']}</div>
                <div class="src-question">"{item['pergunta'][:80]}..."</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:var(--muted);font-size:0.85rem">As fontes da base de conhecimento aparecerão aqui após sua primeira pergunta.</p>', unsafe_allow_html=True)
