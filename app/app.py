import sys

new_path = "/Users/rutujadoble/Documents/Suraj/policy-qa-llm"
sys.path.append(new_path)

import gradio  as gr
from scripts.rag_query import rag_pipeline

def chat_fn(user_query,history):
    return rag_pipeline(user_query)

demo=gr.ChatInterface(
    fn=chat_fn,
    title="HR Policy Chat Bot",
    description="Ask any HR policy-related questions. Powered by RAG + fine-tuned LLM.",
    examples=[
        "Can I take sick leave during probation?",
        "How many casual leaves are allowed per year?",
        "Is there a policy on remote work?",
    ]
)
if __name__=="__main__":
    demo.launch(share=True)