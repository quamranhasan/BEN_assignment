from flask import Flask, request, jsonify
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import time
import sqlite3

app = Flask(__name__)


def setup_database():
    #ALTER THE DATABASE NAME IF NEEDED
    conn = sqlite3.connect('BEN_task_500_100.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                 (query TEXT, response TEXT, latency TEXT)''')
    conn.commit()
    conn.close()

setup_database()

import json

def store_query_response(query, response, latency):
    #ALTER THE DATABASE NAME IF NEEDED
    conn = sqlite3.connect('BEN_task_500_100.db')
    c = conn.cursor()
    
    response_json = json.dumps(response)
    
    c.execute("INSERT INTO chat_history (query, response, latency) VALUES (?, ?, ?)", (query, response_json, latency))
    conn.commit()
    conn.close()

#ALTER THE DATABASE NAME IF NEEDED
DB_FAISS_PATH = 'vectorstore_500_100/db_faiss'

custom_prompt_template = """Use the following information to answer the questions asked.
If you don't know the answer, just say that you don't know the answer.
Keep your answers short and within 50 words.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

#Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(llm=llm,
                                       chain_type='stuff',
                                       retriever=db.as_retriever(search_kwargs={'k': 2}),
                                       return_source_documents=True,
                                       chain_type_kwargs={'prompt': prompt}
                                       )
    return qa_chain

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

#QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

#output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({'query': query})
    return response

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    start_time = time.time()
    response = final_result(query)
    end_time = time.time()

    latency = str(end_time - start_time)
    query = response['query']
    result = response['result']
    response_source_documents = response['source_documents']
    page_contents = [doc.page_content for doc in response_source_documents]


    store_query_response(query, result, latency)

    return jsonify({
        'query': query,
        'response': result,
        'source_documents': page_contents,
        'latency': latency
    })

if __name__ == '__main__':
    app.run(debug=True)




