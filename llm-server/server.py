from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

app = Flask(__name__)

# 모델과 토크나이저 로드
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
# Fine-tuning 데이터셋 클래스
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.input_ids = []
        self.labels = []
        for text in texts:
            input_text = text['input']
            label_text = text['output']
            encoding = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
            label_encoding = tokenizer(label_text, truncation=True, padding='max_length', max_length=max_length, return_tensors='pt')
            self.input_ids.append(encoding['input_ids'][0])  # 텐서로 저장
            self.labels.append(label_encoding['input_ids'][0])  # 텐서로 저장

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'labels': self.labels[idx]}

# TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=500,
    logging_dir='./logs',
)

# Fine-tuning Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=None,  # 최초에는 None으로 설정
)

# Fine-tuning을 위한 엔드포인트
@app.route("/finetune", methods=["POST"])
def finetune():
    data = request.get_json()
    texts = data.get("texts", [])
    
    # Fine-tuning용 데이터셋 생성
    train_dataset = TextDataset(texts, tokenizer)
    trainer.train_dataset = train_dataset  # 데이터셋을 설정
    trainer.train()  # 모델 훈련

    return jsonify({"message": "Fine-tuning started!"})

# 면접 질문 생성 엔드포인트
@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "")

    # 텍스트 생성
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)

    return jsonify({"generated_text": generated_text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
