import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import csv


def load_data(filenames):
    data = []
    for filename in filenames:
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                # Join all the elements except the last one, and use the last element as the label
                sentence = ','.join(row[:-1])
                label = row[-1]
                data.append((sentence, label))
    return data


filenames = ["data.csv", "datas/toutiao_cat_data_mark.txt"]
# filenames = ["data.csv"]

data = load_data(filenames)


# [数据集、数据预处理、词汇表构建、数据集类定义等与上述代码相同]
# 数据集
# data = [
#     ("你是谁", "identity"),
#     ("你叫什么名字", "identity"),
#     ("你能做什么", "capabilities"),
#     ("你会做什么", "capabilities"),
#     ("你知道北京吗", "other"),
#     ("你吃饭了吗", "other")
# ]

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


# 训练模型
# [模型训练代码与上述相同]


# 分割数据集
train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)


# 建立词汇表
def build_vocab(data):
    char_counter = Counter()
    for sentence, _ in data:
        char_counter.update(list(sentence))
    vocab = {char: idx for idx, (char, _) in enumerate(char_counter.most_common())}
    return vocab


vocab = build_vocab(train_data)

# 获取数据集中所有的标签
all_labels = set([label for _, label in data])

# 构建标签到索引的映射
label_to_idx = {label: idx for idx, label in enumerate(all_labels)}



# 数据预处理
def preprocess_data(data, vocab, label_to_idx):
    processed_data = []
    for sentence, label in data:
        tokenized_sentence = list(sentence)  # 分词（字符级）
        vectorized_sentence = [vocab[char] for char in tokenized_sentence if char in vocab]  # 文本向量化
        label_idx = label_to_idx[label]  # 标签向量化
        processed_data.append((vectorized_sentence, label_idx))
    return processed_data


processed_train_data = preprocess_data(train_data, vocab, label_to_idx)
processed_test_data = preprocess_data(test_data, vocab, label_to_idx)


# 创建PyTorch数据集
class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx][0]), torch.tensor(self.data[idx][1])


train_dataset = TextDataset(processed_train_data)
test_dataset = TextDataset(processed_test_data)

# 超参数
VOCAB_SIZE = len(vocab)
EMBED_DIM = 50
NUM_CLASSES = len(label_to_idx)
LEARNING_RATE = 0.001
EPOCHS = 10

# 实例化模型、定义损失函数和优化器
model = TextClassifier(VOCAB_SIZE, EMBED_DIM, NUM_CLASSES)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 将模型移动到GPU上
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练模型
# 在训练循环中
for epoch in range(EPOCHS):
    total_loss = 0
    for inputs, labels in train_dataset:
        # 将数据移到GPU上
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(0))  # 确保输入数据在正确的设备上
        loss = criterion(outputs, labels.unsqueeze(0))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss}")

# 模型评估
# 模型评估
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in DataLoader(test_dataset):  # 注意这里使用DataLoader
        # 将数据移到GPU上
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")

# 保存模型和词汇表
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab': vocab,
    'label_to_idx': label_to_idx,
    'hyper_params': {
        'VOCAB_SIZE': len(vocab),
        'EMBED_DIM': EMBED_DIM,
        'NUM_CLASSES': len(label_to_idx)
    }
}, "model_and_params.pth")
