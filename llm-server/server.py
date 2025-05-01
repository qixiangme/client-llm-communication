from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# 모델 및 토크나이저 로드 (TinyLlama 모델 경로 지정)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0").to(device)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

tokenizer.pad_token = tokenizer.eos_token

fine_tuned_model = None

@app.route("/finetune", methods=["POST"])
def finetune():
    global fine_tuned_model
    data = request.get_json()
    texts = data.get("texts", [])

    input_texts = [text['input'] for text in texts]
    target_texts = [text['output'] for text in texts]

    model.train()

    # 입력과 정답을 한꺼번에 배치로 토크나이징, max_length를 동일하게 설정
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True).to(device)
    
    # target_texts도 max_length에 맞춰 잘라서 패딩 적용
    labels = tokenizer(target_texts, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True).input_ids.to(device)

    # 라벨의 pad 토큰을 무시하도록 설정
    labels[labels == tokenizer.pad_token_id] = -100

    # 입력과 라벨의 배치 크기가 동일한지 확인
    batch_size = inputs.input_ids.size(0)
    if batch_size != labels.size(0):
        # 배치 크기 맞추기 (입력과 레이블의 길이가 동일하지 않으면 최소 길이에 맞추기)
        min_size = min(batch_size, labels.size(0))
        inputs.input_ids = inputs.input_ids[:min_size]
        labels = labels[:min_size]

    # 모델에 입력
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()

    # fine_tuned_model을 업데이트
    fine_tuned_model = model

    return jsonify({"message": f"Fine-tuning completed! Loss: {loss.item():.4f}"})

@app.route("/generate", methods=["POST"])
def generate_text():
    global fine_tuned_model
    data = request.get_json()
    prompt = data.get("prompt", "")

    # fine_tuned_model이 있으면 이를 사용, 없으면 원본 모델 사용
    model_to_use = fine_tuned_model if fine_tuned_model else model

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model_to_use.generate(inputs["input_ids"], max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
