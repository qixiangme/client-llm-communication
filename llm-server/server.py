from flask import Flask, request, jsonify
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from datasets import Dataset

app = Flask(__name__)

# 기본 설정
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

# LoRA 설정
base_model = prepare_model_for_kbit_training(base_model)
peft_config = LoraConfig(
    lora_alpha=32,
    lora_dropout=0.05,
    r=32,
    bias="none",
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(base_model, peft_config)
peft_model.config.use_cache = False

fine_tuned_model = None

@app.route("/finetune", methods=["POST"])
def finetune():
    global fine_tuned_model
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "No texts provided"}), 400

    # 학습용 데이터셋 준비
    formatted_data = [{"text": f"<|user|>\n{item['input']}\n<|assistant|>\n{item['output']}"} for item in texts]
    train_dataset = Dataset.from_list(formatted_data)

    training_arguments = TrainingArguments(
        output_dir="tinyllama-lora-checkpoint",
        per_device_train_batch_size=3,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        save_steps=10,
        logging_steps=10,
        learning_rate=2e-3,
        max_grad_norm=0.3,
        max_steps=50,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        push_to_hub=False
    )

    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        args=training_arguments
    )

    trainer.train()
    fine_tuned_model = trainer.model
    trainer.model.save_pretrained("tinyllama-lora-checkpoint")


    return jsonify({"message": "Fine-tuning completed!"})

@app.route("/generate", methods=["POST"])
def generate_text():
    global fine_tuned_model
    data = request.get_json()
    prompt = data.get("prompt", "")

        # Checkpoint에서 fine-tuned 모델 로드
    if fine_tuned_model is None and os.path.exists("tinyllama-lora-checkpoint"):
        # base_model은 이미 로드되어 있으므로 여기에 LoRA weights만 덮어씌움
        fine_tuned_model = PeftModel.from_pretrained(base_model, "tinyllama-lora-checkpoint")
        fine_tuned_model.to(device)


    model_to_use = fine_tuned_model if fine_tuned_model else base_model

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model_to_use.generate(input_ids, max_new_tokens=256, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
