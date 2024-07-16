# Question Answering System

This project involves the implementation of a Question Answering (QA) system that extracts text from textbooks, creates a hierarchical tree-based index, and retrieves relevant information to answer user queries using advanced natural language processing (NLP) techniques.

## Prerequisites:
PDF1- Introduction to Machine Learning (file:///C:/Users/Sanskruti/Desktop/Introduction%20to%20Machine%20Learning%20with%20Python%20(%20PDFDrive.com%20)-min.pdf)
PDF2- Speech and Language Processing (file:///C:/Users/Sanskruti/Desktop/Speech%20and%20language%20processing.pdf)
PDF3- Pattern Recognition and Machine Learning (file:///C:/Users/Sanskruti/Desktop/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

## Project Structure

- Python.ipynb: Jupyter notebook that handles text extraction from PDF textbooks and creates a hierarchical tree-based index.
- qa_test.py: Python script that implements retrieval and ranking methods to answer questions based on the indexed content.
- application.py: Pyton script that implements the streamlit for the front-end for the model.

## Requirements

To run the code, you need the following libraries:
- PyPDF2
- numpy
- rank_bm25
- sentence_transformers
- transformers
- streamlit
- re

## How to use
1. Download the Github repository in your local server.
2. Run all the python code of python.ipynb.
3. Install all the required libraries.
4. Run the qa_test.py. (You will get the answers to the queries in the terminal section).
5. Run the application.py in the terminal as "streamlit run application.py". A local server will appear on the desktop to streamlit for asking your queries.
Try asking "What is deep learning?"

Feedback will be highly appreciated!

You can install these libraries using pip:
```bash
pip install PyPDF2 numpy rank_bm25 sentence-transformers transformers streamlit
