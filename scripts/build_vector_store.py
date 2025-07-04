# Load policy documents → Chunk → Embed → Store in FAISS

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

folder_path=r'/Users/destiny_mac/Documents/Suraj/projects/policy-qa-llm'
model_name="all-MiniLM-L6-v2"

# Load the documents
def load_documents(folder_path) :
    docs=[]
    for filename in os.listdir(folder_path+ "/data/policy_docs/"):
        if filename.endswith(".txt") or filename.endswith(".md"):
            with open(os.path.join(folder_path+ "/data/policy_docs", filename), 'r', encoding='utf-8') as f:
                docs.append(f.read())
    return docs

# Split text into chunks
def chunk_documents(docs,chunk_size=300,chunk_overlap=50):
    splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.create_documents(docs)

# Embed and store in FAISS
def build_and_save_vector_store(chunks,save_path=folder_path+"/models/hr_faiss_index"):
    embedder=HuggingFaceEmbeddings(model_name=model_name)
    db=FAISS.from_documents(chunks,embedder)
    db.save_local(save_path)
    print(f"Saved FAISS index to: {save_path}")

if __name__ == "__main__":
    docs = load_documents(folder_path)
    chunks = chunk_documents(docs)
    build_and_save_vector_store(chunks)
