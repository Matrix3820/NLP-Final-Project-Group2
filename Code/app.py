import streamlit as st
from RAG.llm3 import generate_medical_recipe_response, generate_llm_only_response

st.set_page_config(page_title="Medically Aware Recipe Assistant", layout="wide")

# ---------- Sidebar ----------
st.sidebar.title("ℹ️ About")
st.sidebar.markdown("""
Welcome to RAGnRecipe
A LLM powered with **Retrieval-Augmented Generation (RAG)** to suggest medically-aware recipes tailored to your query.

- 🤖 **LLM:** [Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1)  
- 📚 **Recipe Data:** Indexed from a curated Recipe Dataset  
- 🧠 **Two generation modes**:
    - 🔗 *RAG + LLM*: Retrieval-informed, grounded in indexed ingredient facts  
    - 🧠 *LLM Only*: Based purely on general language model knowledge
""")

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ Generation Parameters")
st.sidebar.markdown("""
- `truncation = True`  
- `max_length = 7400`  
- `max_new_tokens = 700`  
- `top_k (retrieval) = 3`  
- `top_p` = *Disabled*  
- `temperature` = *Disabled*  
- `do_sample = False`  
""")

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚠️ Disclaimer")
st.sidebar.markdown("""
This tool is **not a substitute for medical advice** and its use is considered **AT WILL**.

- The recipe suggestions are AI-generated.
- Ingredient facts are sourced from public nutrition data through manual curation and through GPT prompting. They may not be 100% accurate

📊 Dataset used for Recipe Data:  
[Kaggle - Recipe Dataset](https://www.kaggle.com/datasets/wilmerarltstrmberg/recipe-dataset-over-2m)

Please consult a healthcare provider before making dietary decisions.
""")



# ---------- Main ----------
st.title("🥗 RAGnRecipe")
st.markdown("Get recipes tailored to medical needs with ingredient-level facts.")

query = st.text_area(
    "🔍 Enter your query:",
    placeholder="e.g., Thyroid and Gout friendly recipe for weight loss"
)

if st.button("Generate"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Generating both responses..."):
            rag_response = generate_medical_recipe_response(query)
            llm_response = generate_llm_only_response(query)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔗 RAG + LLM Response")
            st.markdown(rag_response)

        with col2:
            st.subheader("🧠 LLM-Only Response")
            st.markdown(llm_response)
