# Import necessary libraries
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import torch

# Initialize SentenceTransformer model
model_name = 'msmarco-distilbert-base-v4'
model = SentenceTransformer(model_name)

# Define your corpus
corpus = [
    "Introduction to machine learning techniques.",
    "Deep learning models for image recognition.",
    "Natural language processing with transformers.",
    "Advanced machine learning algorithms.",
    "Applications of AI in healthcare."
]

# Initialize BM25
bm25 = BM25Okapi([doc.split() for doc in corpus])

# Initialize QA model and tokenizer
qa_model_name = 'distilbert-base-cased-distilled-squad'
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)

# Function to retrieve sections from a document
def retrieve_sections(doc):
    return doc.split('. ')

# Function for hybrid retrieval
def hybrid_retrieve(query, corpus, top_k=5):
    bm25_scores = bm25.get_scores(query.split())
    query_embedding = model.encode(query, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(query_embedding, model.encode(corpus, convert_to_tensor=True))[0]
    combined_scores = bm25_scores + semantic_scores.cpu().numpy()
    top_k_indices = combined_scores.argsort()[-top_k:][::-1]
    return [corpus[i] for i in top_k_indices]

# Function to perform re-ranking based on query keywords overlap
def re_rank(results, query):
    return sorted(results, key=lambda x: len(set(query.split()) & set(x.split())), reverse=True)

# Function to perform RAG system
def rag_system(question):
    top_k_docs = hybrid_retrieve(question, corpus)
    re_ranked_docs = re_rank(top_k_docs, question)
    context = ' '.join(re_ranked_docs)
    return context

# Function to answer questions
def answer_question(question, context):
    inputs = qa_tokenizer(question, context, truncation=True, padding=True, return_tensors='pt')
    qa_outputs = qa_model(**inputs)
    answer_start_scores = qa_outputs.start_logits
    answer_end_scores = qa_outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
    )
    return answer

# Define Streamlit app
def main():
    st.title("RAG Model for Question Answering")
    user_query = st.text_input("Enter your question:")
    
    if st.button("Answer"):
        context = rag_system(user_query)
        answer = answer_question(user_query, context)
        st.subheader("Question:")
        st.write(user_query)
        st.subheader("Answer:")
        st.write(answer)

if __name__ == "__main__":
    main()
