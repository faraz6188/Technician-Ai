import google.generativeai as genai
import PyPDF2
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss  # Import FAISS
import os
from dotenv import load_dotenv

def create_dotenv_file():
    """
    Creates a .env file with placeholder values for the Google API key.
    """
    with open(".env", "w") as f:
        f.write("GOOGLE_API_KEY=\"YOUR_GOOGLE_API_KEY\"\n")

def check_dotenv_file():
    """
    Checks if the .env file exists.  If it does not, it creates it.
    """
    if not os.path.exists(".env"):
        create_dotenv_file()
        print(".env file created with placeholder. Please edit it with your API key.")
    else:
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            print(".env file exists but GOOGLE_API_KEY is not set.  Please edit it with your API key.")

check_dotenv_file()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("GOOGLE_API_KEY is not set. Exiting.")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)
# model = genai.GenerativeModel("gemini-2.0-flash-lite-001")  # Removed hardcoded model name
embed_model = SentenceTransformer("all-minilm-l6-v2")

class VectorDB:
    def __init__(self, dimension: int, index_name: str = "default_index"):
        self.index = faiss.IndexFlatL2(dimension)
        self.index_name = index_name
        self.namespace_map = {}
        self.dimension = dimension

    def create_index(self):
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)

    def store(self, vectors, texts, namespace):
        if self.index is None:
            self.create_index()

        if namespace not in self.namespace_map:
            self.namespace_map[namespace] = 0
        offset = self.namespace_map[namespace]
        ids = [offset + i for i in range(len(vectors))]
        self.index.add(np.array(vectors))
        if not hasattr(self, 'id_to_text'):
            self.id_to_text = {}
        for i, text in enumerate(texts):
            self.id_to_text[ids[i]] = {"text": text, "namespace": namespace}
        self.namespace_map[namespace] = offset + len(vectors)

    def query(self, vector, top_k, namespace):
        if self.index is None:
            print("warning: index is None")
            return []
        distances, faiss_ids = self.index.search(np.array([vector]), top_k)
        results = []
        for i, id in enumerate(faiss_ids[0]):
            if id in self.id_to_text and self.id_to_text[id]['namespace'] == namespace:
                text = self.id_to_text[id]['text']
                results.append((distances[0][i], text))
        return results

    def delete(self, namespace):
        if namespace in self.namespace_map:
            offset = self.namespace_map[namespace]
            ids_to_delete = [offset + i for i in range(self.namespace_map[namespace])]
            self.index.remove_ids(np.array(ids_to_delete))
            del self.namespace_map[namespace]
        else:
            print(f"Namespace {namespace} not found.")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def get_embeddings(texts):
    embeddings = embed_model.encode(texts).tolist()
    return embeddings

def store_embeddings(vector_db, texts, embeddings, namespace):
    vector_db.store(vectors=embeddings, texts=texts, namespace=namespace)

def retrieve_relevant_chunks(vector_db, query, namespace, top_k=5):
    query_embedding = embed_model.encode([query]).tolist()[0]
    results = vector_db.query(vector=query_embedding, top_k=top_k, namespace=namespace)
    relevant_chunks = [text for _, text in results]
    return relevant_chunks

def generate_response(query, context_chunks, model_name="gemini-2.0-flash-lite-001"): #added model_name
    context = "\n\n".join(context_chunks)
    prompt = f"""
    You are a technical support assistant. Use the information below to answer the user's question.
    Context:
    {context}

    User Question:
    {query}
    """
    try:
        # Use the model_name parameter here.
        response = genai.GenerativeModel(model_name).generate_content(prompt)
        return response.text if response.text else "I couldn't find relevant information in the knowledge base."
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I encountered an error while processing your request."

def process_query(vector_db, pdf_path, query, model_name="gemini-2.0-flash-lite-001"): #added model_name
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        return "I could not extract information from the PDF."
    text_chunks = chunk_text(pdf_text)
    embeddings = get_embeddings(text_chunks)
    pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
    store_embeddings(vector_db, text_chunks, embeddings, namespace=pdf_filename)
    relevant_chunks = retrieve_relevant_chunks(vector_db, query, namespace=pdf_filename)
    response = generate_response(query, relevant_chunks, model_name) #added model_name
    return response

import numpy as np

def main():
    index_name = "tech-support-index"
    pdf_files = ["pdf/ABS0130.pdf", "pdf/ABS0131.pdf", "pdf/ACC0048.pdf", "pdf/ACT0021.pdf", "pdf/AOS0832.pdf", "pdf/TMT0158.pdf"]  # List of PDF file paths
    # Create dummy pdf files
    with open("example.pdf", "w") as f:
        f.write("This is a dummy pdf file.  Error code E05 means the printer is out of paper.  To fix it, add paper.  The printhead is located in the front of the printer.  To replace it, open the front cover, remove the old printhead, and insert the new one.")
    with open("example2.pdf", "w") as f:
        f.write("This is another dummy pdf file.  Step one is to turn off the device. Step two is to unplug the power cord.")


    vector_db = VectorDB(dimension=384, index_name=index_name)

    # Process all PDF files
    for pdf_file in pdf_files:
        pdf_text = extract_text_from_pdf(pdf_file)
        if pdf_text:
            text_chunks = chunk_text(pdf_text)
            embeddings = get_embeddings(text_chunks)
            pdf_filename = os.path.splitext(os.path.basename(pdf_file))[0]
            store_embeddings(vector_db, text_chunks, embeddings, namespace=pdf_filename)

    while True:
        user_query = input("Ask your question (or type 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break

        #query across all pdfs
        relevant_chunks = []
        for pdf_file in pdf_files:
            pdf_filename = os.path.splitext(os.path.basename(pdf_file))[0]
            relevant_chunks.extend(retrieve_relevant_chunks(vector_db, user_query, namespace=pdf_filename))
        answer = generate_response(user_query, relevant_chunks)
        print(f"Query: {user_query}")
        print(f"Answer: {answer}")
    
    for pdf_file in pdf_files:
        pdf_filename = os.path.splitext(os.path.basename(pdf_file))[0]
        vector_db.delete(namespace=pdf_filename) #cleanup

if __name__ == "__main__":
    main()
