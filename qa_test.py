from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from rank_bm25 import BM25Okapi

# Assuming the corpus is already generated from the notebook and saved
corpus = [
    "Introduction to machine learning techniques.",
    "Deep learning models for image recognition.",
    "Natural language processing with transformers.",
    "Advanced machine learning algorithms.",
    "Applications of AI in healthcare."
]


# Initialize BM25
bm25 = BM25Okapi([doc.split() for doc in corpus])
model = SentenceTransformer('msmarco-distilbert-base-v4')

def hybrid_retrieve(query, corpus, top_k=5):
    bm25_scores = bm25.get_scores(query.split())
    query_embedding = model.encode(query, convert_to_tensor=True)
    semantic_scores = util.pytorch_cos_sim(query_embedding, model.encode(corpus, convert_to_tensor=True))[0]
    combined_scores = bm25_scores + semantic_scores.cpu().numpy()
    top_k_indices = combined_scores.argsort()[-top_k:][::-1]
    return [corpus[i] for i in top_k_indices]



# Initialize the model for semantic search
model = SentenceTransformer('msmarco-distilbert-base-v4')

# Initialize the QA model and tokenizer
qa_tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
qa_model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

# Define your corpus
corpus = [
    "Introduction to machine learning techniques.",
    "Deep learning models for image recognition.",
    "Natural language processing with transformers.",
    "Advanced machine learning algorithms.",
    "Applications of AI in healthcare."
]

# Encode the corpus once
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)

# Define a function to retrieve sections from a document
def retrieve_sections(doc):
    return doc.split('. ')

# Define the RAG system function
def rag_system(question):
    # Encode the query
    query_embedding = model.encode(question, convert_to_tensor=True)
    
    # Perform semantic search
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=3)  # Reduced to top 3 for more relevant context
    
    # Retrieve sections from relevant documents
    relevant_sections = []
    for hit in hits[0]:
        doc = corpus[hit['corpus_id']]
        sections = retrieve_sections(doc)
        relevant_sections.extend(sections)
    
    # Combine the sections into a single context
    context = ' '.join(relevant_sections)
    return context

def re_rank(results, query):
    return sorted(results, key=lambda x: len(set(query.split()) & set(x.split())), reverse=True)

def rag_system(question):
    top_k_docs = hybrid_retrieve(question, corpus)
    re_ranked_docs = re_rank(top_k_docs, question)
    context = ' '.join(re_ranked_docs)
    return context


# Define the function to answer questions
def answer_question(question, context):
    inputs = qa_tokenizer(question, context, truncation=True, padding=True, return_tensors='pt')
    
    # Answer the question using the QA model
    qa_outputs = qa_model(**inputs)
    
    # Get the answer span
    answer_start_scores = qa_outputs.start_logits
    answer_end_scores = qa_outputs.end_logits
    
    # Find the tokens with the highest start and end scores
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    
    # Get the answer from the tokens
    answer = qa_tokenizer.convert_tokens_to_string(
        qa_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
    )
    
    return answer

# Example queries
queries = [
    "How do transformers work?",
    "What are applications of AI in healthcare?"
]

# Process each query
for query in queries:
    context = rag_system(query)
    answer = answer_question(query, context)
    print(f"Question: {query}")
    print(f"Answer: {answer}")
    print("-" * 50)

