import os
import streamlit as st
import wikipedia
import gdown
from llama_cpp import Llama

# ------------------ CONFIG ------------------
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "phi-2.Q4_K_M.gguf")

# 🔹 Replace this with your Google Drive file URL
MODEL_DRIVE_URL = "https://drive.google.com/uc?id=1bquBi_ccK4XDsatiHZsucysPUBXzmga6"


# --------------------------------------------

def download_model():
    """Download model from Google Drive if not already present."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading Phi-2 model... This may take a few minutes ⏳")
        gdown.download(MODEL_DRIVE_URL, MODEL_PATH, quiet=False)
        st.success("Model downloaded successfully ✅")
    else:
        st.info("Model already available ✅")

@st.cache_resource
def load_model():
    """Load the Phi-2 model."""
    llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8)
    return llm

def get_wiki_content(topic):
    """Fetch Wikipedia content for a given topic."""
    try:
        page = wikipedia.page(topic)
        content = page.content
        return content[:6000]  # truncate to avoid context overflow
    except Exception as e:
        st.error(f"❌ Error fetching content: {e}")
        return None

def summarize_content(llm, topic, content, question=None):
    """Generate summary or answer using Phi-2."""
    if question:
        prompt = f"""
You are an AI assistant. Based on the Wikipedia content below,
answer the question clearly and concisely.

Wikipedia content:
\"\"\"{content}\"\"\"

Question: {question}
Answer:
"""
    else:
        prompt = f"""
Summarize the following Wikipedia content about '{topic}' in a clear,
structured, and easy-to-read manner.

Wikipedia content:
\"\"\"{content}\"\"\"

Summary:
"""
    output = llm(prompt, max_tokens=512, temperature=0.6)
    return output["choices"][0]["text"].strip()

# ------------------ STREAMLIT UI ------------------

st.set_page_config(page_title="Wikipedia Q&A Assistant", layout="wide")
st.title("📚 Wikipedia Q&A Assistant (Phi-2)")
st.markdown("Ask questions or get summaries from any Wikipedia topic using the Phi-2 model!")

# Sidebar: Model Setup
st.sidebar.header("⚙️ Setup")
if st.sidebar.button("🔽 Download/Check Model"):
    download_model()

# Topic Input
topic = st.text_input("🔍 Enter a Wikipedia topic (e.g., Quantum Computing):")

if topic:
    with st.spinner("Fetching Wikipedia content..."):
        wiki_text = get_wiki_content(topic)

    if wiki_text:
        with st.expander("📄 View Retrieved Wikipedia Text"):
            st.write(wiki_text[:2000] + "..." if len(wiki_text) > 2000 else wiki_text)

        # Load Model
        with st.spinner("Loading Phi-2 model..."):
            llm = load_model()

        # Choose Action
        st.subheader("🧩 Choose Action")
        action = st.radio("Select what you want to do:", ["Summarize Topic", "Ask a Question"])

        if action == "Summarize Topic":
            if st.button("📝 Generate Summary"):
                with st.spinner("Generating summary..."):
                    summary = summarize_content(llm, topic, wiki_text)
                st.success("✅ Summary generated!")
                st.write(summary)

        elif action == "Ask a Question":
            question = st.text_input("❓ Enter your question about this topic:")
            if st.button("💬 Get Answer") and question:
                with st.spinner("Generating answer..."):
                    answer = summarize_content(llm, topic, wiki_text, question)
                st.success("✅ Answer generated!")
                st.write(answer)

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit + Phi-2")
