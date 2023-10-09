import torch
import torch.nn as nn

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA device name:", torch.cuda.get_device_name(0))

# 定义模型
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 加载模型和词汇表
saved_data = torch.load("model_and_params.pth")
vocab = saved_data['vocab']
label_to_idx = saved_data['label_to_idx']

hyper_params = saved_data['hyper_params']
VOCAB_SIZE = hyper_params['VOCAB_SIZE']
EMBED_DIM = hyper_params['EMBED_DIM']
NUM_CLASSES = hyper_params['NUM_CLASSES']

model = TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)
# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 将模型移动到GPU上
model.to(device)

model.load_state_dict(saved_data['model_state_dict'])
model.eval()

# 定义预测函数
def predict(sentence, model, vocab, label_to_idx, confidence_threshold=0.5):
    model.eval()

    tokenized_sentence = list(sentence)
    vectorized_sentence = [vocab[char] for char in tokenized_sentence if char in vocab]

    tensor_sentence = torch.tensor(vectorized_sentence, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor_sentence)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        print(confidence.item())

        if torch.isnan(confidence).item():
            return "I'm sorry, I can't understand this input."

        if confidence.item() < confidence_threshold:
            return "I'm not sure about this input."

        idx_to_label = {v: k for k, v in label_to_idx.items()}
        predicted_label = idx_to_label[predicted_idx.item()]

    return predicted_label


# 使用模型进行预测
sentence = "你是那位"
predicted_label = predict(sentence, model, vocab, label_to_idx)

print(f"The sentence '{sentence}' is classified as: {predicted_label}")

# 循环，允许用户输入句子进行预测
while True:
    sentence = input("Please input a sentence (or 'exit' to stop): ")
    if sentence.lower() == 'exit':
        print("Exiting the program.")
        break

    predicted_label = predict(sentence, model, vocab, label_to_idx,0.7)
    print(f"The sentence '{sentence}' is classified as: {predicted_label}\n" )
