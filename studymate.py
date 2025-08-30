import streamlit as st
from pypdf import PdfReader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch

st.set_page_config(layout="wide", page_title="Reliable PDF Q&A")

# --- Model Loading ---
@st.cache_resource
def load_models():
    st.info("⏳ Loading models...")
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    st.success("✅ Models loaded!")
    return tokenizer, model, embed_model

tokenizer, model, embed_model = load_models()

# --- PDF Processing ---
def get_pdf_text_chunks(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        if not text.strip():
            st.error("❌ No text found in PDF.")
            return []

        # Split into sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 500:
                current_chunk += (sentence + ". ")
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    except Exception as e:
        st.error(f"❌ Error processing PDF: {e}")
        return []

# --- Streamlit UI ---
st.title("📘 Reliable PDF Q&A")
st.markdown("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    with st.spinner("⏳ Processing PDF..."):
        chunks = get_pdf_text_chunks(uploaded_file)

        if chunks:
            st.success(f"PDF processed into {len(chunks)} chunks.")
            
            # Encode all chunks
            chunk_embeddings = embed_model.encode(chunks, convert_to_tensor=True, batch_size=16)

            query = st.text_input("❓ Ask a question about the PDF:")

            if query:
                with st.spinner("⏳ Generating answer..."):
                    try:
                        query_emb = embed_model.encode(query, convert_to_tensor=True)
                        similarities = util.cos_sim(query_emb, chunk_embeddings)[0]
                        top_k = min(3, len(chunks))
                        top_indices = torch.topk(similarities, k=top_k).indices
                        top_indices = top_indices[torch.argsort(similarities[top_indices], descending=True)]
                        relevant_text = "\n\n".join([chunks[i] for i in top_indices])

                        # Prepare T5 prompt
                        prompt = f"Answer the question based on the context below.\n\nContext: {relevant_text}\n\nQuestion: {query}\nAnswer:"
                        inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
                        outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4, early_stopping=True)
                        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

                        if answer.strip():
                            st.subheader("✅ Answer:")
                            st.write(answer)
                        else:
                            st.warning("⚠️ Could not generate an answer. Try rephrasing the question.")

                    except Exception as e:
                        st.error(f"❌ Error generating answer: {e}")