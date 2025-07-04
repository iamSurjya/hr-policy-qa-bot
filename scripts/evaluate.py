import json
from transformers import pipeline
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from tqdm import tqdm
import os

folder_path=r'/Users/destiny_mac/Documents/Suraj/projects/policy-qa-llm'

# Load fine tuned model
model_path=folder_path+'/models/lora_checkpoints'
llm=pipeline(
        "text2text-generation",
        model=model_path,
        tokenizer=model_path,
        max_new_tokens=300,
        do_sample=False,
        )
# load evaluation set
def load_eval_data(filepath):
    with open(filepath,"r") as f:
        return [json.loads(line.strip()) for line in f]

# format input for model 
def format_prompt(question):
    return f"### Human: {question}\n### Assistant:"

# compute metrics
def compute_metrics(pred,ref):
    scorer=rouge_scorer.RougeScorer(['rougeL'],use_stemmer=True)
    rouge=scorer.score(ref,pred)['rougeL'].fmeasure
    bleu=sentence_bleu([ref.split()],pred.split())
    exact_match=int(pred.strip().lower()==ref.strip().lower())
    return {
        "rougeL":rouge,
        "bleu":bleu,
        "exact_match":exact_match,
    }

# Evaluation Loop
def evaluate(eval_data):
    results=[]
    for item in tqdm(eval_data,desc="Evaluating"):
        prompt=format_prompt(item["question"])
        output=llm(prompt)[0]["generated_text"].strip()
        metrics=compute_metrics(output,item['expected_answer'])
        results.append({
            "question":item["question"],
            "expected":item["expected_answer"],
            "predicted":output,
            **metrics
        })
    return results

# save results
def save_results(results,path):
    os.makedirs("results",exist_ok=True)
    with open(path,"w") as f:
        for r in results:
            f.write(json.dumps(r)+"\n")

#Evaluation Summary
def summarize(results):
    avg_bleu = sum(r["bleu"] for r in results) / len(results)
    avg_rouge = sum(r["rougeL"] for r in results) / len(results)
    exact = sum(r["exact_match"] for r in results)
    print(f"\n✅ Evaluation Summary:")
    print(f"Exact Match: {exact}/{len(results)} ({(100*exact/len(results)):.2f}%)")
    print(f"Avg BLEU: {avg_bleu:.4f}")
    print(f"Avg ROUGE-L: {avg_rouge:.4f}")

if __name__=="__main__":
    eval_data=load_eval_data(folder_path +"/data/eval_pairs.jsonl")
    results=evaluate(eval_data)
    save_results(results,folder_path+"/results/eval_results.jsonl")
    summarize(results)
