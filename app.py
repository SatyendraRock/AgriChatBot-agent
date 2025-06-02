import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json

# Load data
agri_data = [
    {
        "question": "What fertilizer is best for tomatoes?",
        "answer": "Tomatoes thrive with a balanced NPK fertilizer, preferably 10-10-10 or 5-10-10 depending on growth stage."
    },
    {
        "question": "How do you control pests in rice?",
        "answer": "Use integrated pest management, monitor fields regularly, and apply biopesticides or neem oil."
    },
    {
        "question": "How often should I water wheat crops?",
        "answer": "Water wheat crops every 7–10 days depending on soil type and weather conditions."
    }
]

model = SentenceTransformer('all-MiniLM-L6-v2')

questions = [item["question"] for item in agri_data]
answers = [item["answer"] for item in agri_data]
embeddings = model.encode(questions)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

def get_response(query):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), 1)
    return agri_data[I[0][0]]["answer"]

# Streamlit App
st.title("🌾 Agri AI Assistant")
st.write("Ask agriculture-related questions:")

user_input = st.text_input("Your question")
if user_input:
    with st.spinner("Thinking..."):
        response = get_response(user_input)
        st.success(response)
