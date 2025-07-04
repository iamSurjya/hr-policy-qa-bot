import json
from datasets import Dataset
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,TrainingArguments,Trainer,DataCollatorForSeq2Seq
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
#from transformers import BitsAndBytesConfig
import torch

# configurations
folder_path=r'/Users/destiny_mac/Documents/Suraj/projects/policy-qa-llm'
MODEL_NAME = "google/flan-t5-base" 
OUTPUT_DIR = folder_path+ "/models/lora_checkpoints/"
QA_PATH = folder_path + "/data/qa_pairs.json"

# load qa dataset
def load_qa_dataset(path):
    with open(path,"r") as f:
        qa_data=json.load(f)
    return Dataset.from_list(qa_data)

# Foramt : "### Human: {question} \n### Assistant:{answer}"
def tokenize_example(example,tokenizer,max_length=512):
    prompt = f"### Human: {example['question']}\n### Assistant:"
    input_ids=tokenizer(prompt, 
                        truncation=True,
                        padding="max_length",
                        max_length=max_length,
                        return_tensors="pt")
    
    labels=tokenizer(example['answer'],
                     truncation=True,
                     padding="max_length",
                     max_length=max_length,
                     return_tensors="pt")["input_ids"]
    labels[labels==tokenizer.pad_token_id]=-100

    return {
        "input_ids":input_ids['input_ids'][0],
        "attention_mask":input_ids['attention_mask'][0],
        "labels":labels[0],
    }

def main():
    # Load tokenizer and model

    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME)

    # bnb_config=BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type='nf4',
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    #model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME,quantization_config=bnb_config,device_map="auto") # quantization only work on Linux and NVIDIA GPU only
    
    model=AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model=prepare_model_for_kbit_training(model)

    # PEFT LoRA Config

    lora_config=LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q","v"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )

    model=get_peft_model(model,lora_config)

    # Load and Tokenize dataset
    raw_dataset=load_qa_dataset(QA_PATH)
    tokenised_dataset=raw_dataset.map(lambda x:
                                      tokenize_example(x,tokenizer),
                                      remove_columns=['question','answer'])
    
    # training examples
    training_args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_dir=folder_path +"/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        fp16=True,
        report_to=None
    )

    data_collator=DataCollatorForSeq2Seq(tokenizer,model)

    # train
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_dataset,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()