from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from transformers import pipeline
import os

folder_path=r'/Users/destiny_mac/Documents/Suraj/projects/policy-qa-llm'
model_name="all-MiniLM-L6-v2"

# load FAISS Vector DB
def load_vector_store(index_path=folder_path+'/models/hr_faiss_index'):
    embedder=HuggingFaceBgeEmbeddings(model_name=model_name)
    vectorstore=FAISS.load_local(index_path,embedder,allow_dangerous_deserialization=True)
    return vectorstore

# Get relevent context from FAISS
def retrive_context(query,vectorstore,k=3):
    docs=vectorstore.similarity_search(query=query,k=k)
    context="\n\n".join([doc.page_content for doc in docs])
    return context

# Generate Prompt for LLM
def construct_prompt(user_query,context):
    propmt=f""" You are an HR policy assistant. Answer the user's question based only on the context below.
If the answer is not found in the context, say "I could not find that in the current policy document."

Context: 
{context}

Question:
{user_query}

Answer:
"""
    return propmt

# query the language model 
# def query_llm(prompt,model_name="google/flan-t5-base"): # model_name="google/flan-t5-base" for initial modelling
#     llm=pipeline("text2text-generation", # text-generation if we want to use the mistralai/Mistral-7B-Instruct-v0.1 model
#                  model=model_name,
#                  max_new_tokens=300,
#                  do_sample=False)
#     output=(llm(prompt)[0]["generated_text"])
#     return output.strip()

# query the language model after finetuning

def query_llm(prompt):
    model_path=folder_path+'/models/lora_checkpoints'
    llm=pipeline(
        "text2text-generation",
        model=model_path,
        tokenizer=model_path,
        max_new_tokens=300,
        do_sample=False,
        )
    output=(llm(prompt)[0]["generated_text"])
    return output

# RAG Pipeline
def rag_pipeline(user_query):
    vectorstore=load_vector_store()
    context=retrive_context(user_query, vectorstore)
    prompt=construct_prompt(user_query,context)
    answer=query_llm(prompt)
    return answer

# CLI usage
if __name__ == "__main__":
    question = input("Enter your HR policy question: ")
    answer = rag_pipeline(question)
    print("\n💬 LLM Answer:\n", answer)