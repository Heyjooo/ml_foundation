import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataset.mnist import load_mnist

# 하이퍼파라미터
input_size = 784
hidden1_size = 50
hidden2_size = 30
hidden3_size = 20
output_size = 10
learning_rate = 0.01
epochs = 100
batch_size = 100

# 네트워크 정의
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, hidden3_size)
        self.fc4 = nn.Linear(hidden3_size, output_size)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.fc4(x)
        return x

# 데이터 로드 및 전처리
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(t_train.argmax(axis=1)))
test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(t_test.argmax(axis=1)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 및 최적화기 정의
model = NeuralNetwork(input_size, hidden1_size, hidden2_size, hidden3_size, output_size)
optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)

# 손실 함수 정의
criterion = nn.CrossEntropyLoss()

# 학습
costs = []
for epoch in range(epochs):
    epoch_cost = 0.0
    for x_batch, t_batch in train_loader:
        optimizer.zero_grad()
        y = model(x_batch)
        loss = criterion(y, t_batch)
        loss.backward()
        optimizer.step()
        epoch_cost += loss.item()

    costs.append(epoch_cost)
    print(f"Epoch {epoch + 1}/{epochs}, Cost: {epoch_cost}")

# 비용 그래프 시각화
import matplotlib.pyplot as plt

plt.plot(costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost Function J() over Epochs')
plt.show()

# 테스트 및 정확도 출력
correct_count = 0
with torch.no_grad():
    for x_batch, t_batch in test_loader:
        y_pred = model(x_batch)
        predicted_labels = torch.argmax(y_pred, dim=1)
        correct_count += torch.sum(predicted_labels == t_batch).item()

accuracy = correct_count / len(x_test)
print(f"Test Accuracy: {accuracy}")
