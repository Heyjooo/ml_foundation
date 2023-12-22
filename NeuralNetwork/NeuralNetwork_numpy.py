import numpy as np
from dataset.mnist import load_mnist
from Functions import sigmoid, softmax, cross_entropy_error

# 네트워크 초기화
def init_network(input_size, hidden1_size, hidden2_size, hidden3_size, output_size):
    network = {}
    network['W1'] = np.random.randn(input_size, hidden1_size)
    network['b1'] = np.zeros(hidden1_size)
    network['W2'] = np.random.randn(hidden1_size, hidden2_size)
    network['b2'] = np.zeros(hidden2_size)
    network['W3'] = np.random.randn(hidden2_size, hidden3_size)
    network['b3'] = np.zeros(hidden3_size)
    network['W4'] = np.random.randn(hidden3_size, output_size)
    network['b4'] = np.zeros(output_size)
    # Adagrad에 필요한 추가 변수 초기화
    network['h_W1'] = np.zeros_like(network['W1'])
    network['h_b1'] = np.zeros_like(network['b1'])
    network['h_W2'] = np.zeros_like(network['W2'])
    network['h_b2'] = np.zeros_like(network['b2'])
    network['h_W3'] = np.zeros_like(network['W3'])
    network['h_b3'] = np.zeros_like(network['b3'])
    network['h_W4'] = np.zeros_like(network['W4'])
    network['h_b4'] = np.zeros_like(network['b4'])
    return network

# 예측 수행
def predict(network, x):
    W1, b1, W2, b2, W3, b3, W4, b4 = network['W1'], network['b1'], network['W2'], network['b2'], network['W3'], network['b3'], network['W4'], network['b4']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    z3 = sigmoid(a3)
    a4 = np.dot(z3, W4) + b4
    y = softmax(a4)

    # Forward pass의 각 층의 출력을 저장 (Backward pass에 사용)
    network['z1'], network['z2'], network['z3'] = z1, z2, z3

    return y

def compute_cost(network, x, t):
    y = predict(network, x)
    return cross_entropy_error(y, t)

def train(network, x, t, learning_rate, epochs, batch_size):
    costs = []  # 각 에폭마다 비용 기록
    for epoch in range(epochs):
        for i in range(0, len(x), batch_size):
            x_batch = x[i:i + batch_size]
            t_batch = t[i:i + batch_size]

            # Forward
            y = predict(network, x_batch)

            # Backward (Backpropagation)
            # 출력층의 그래디언트 계산
            dy = (y - t_batch) / batch_size

            # 출력층의 가중치와 편향에 대한 그래디언트 계산
            grad_W4 = np.dot(network['z3'].T, dy)
            grad_b4 = np.sum(dy, axis=0)

            # 은닉층 3의 그래디언트 계산
            dz3 = np.dot(dy, network['W4'].T)
            da3 = dz3 * (1 - network['z3']) * network['z3']

            # 은닉층 3의 가중치와 편향에 대한 그래디언트 계산
            grad_W3 = np.dot(network['z2'].T, da3)
            grad_b3 = np.sum(da3, axis=0)

            # 은닉층 2의 그래디언트 계산
            dz2 = np.dot(da3, network['W3'].T)
            da2 = dz2 * (1 - network['z2']) * network['z2']

            # 은닉층 2의 가중치와 편향에 대한 그래디언트 계산
            grad_W2 = np.dot(network['z1'].T, da2)
            grad_b2 = np.sum(da2, axis=0)

            # 은닉층 1의 그래디언트 계산
            dz1 = np.dot(da2, network['W2'].T)
            da1 = dz1 * (1 - network['z1']) * network['z1']

            # 은닉층 1의 가중치와 편향에 대한 그래디언트 계산
            grad_W1 = np.dot(x_batch.T, da1)
            grad_b1 = np.sum(da1, axis=0)

            # Adagrad 업데이트
            network['h_W1'] += grad_W1 ** 2
            network['W1'] -= learning_rate * grad_W1 / (np.sqrt(network['h_W1']) + 1e-7)

            network['h_b1'] += grad_b1 ** 2
            network['b1'] -= learning_rate * grad_b1 / (np.sqrt(network['h_b1']) + 1e-7)

            network['h_W2'] += grad_W2 ** 2
            network['W2'] -= learning_rate * grad_W2 / (np.sqrt(network['h_W2']) + 1e-7)

            network['h_b2'] += grad_b2 ** 2
            network['b2'] -= learning_rate * grad_b2 / (np.sqrt(network['h_b2']) + 1e-7)

            network['h_W3'] += grad_W3 ** 2
            network['W3'] -= learning_rate * grad_W3 / (np.sqrt(network['h_W3']) + 1e-7)

            network['h_b3'] += grad_b3 ** 2
            network['b3'] -= learning_rate * grad_b3 / (np.sqrt(network['h_b3']) + 1e-7)

            network['h_W4'] += grad_W4 ** 2
            network['W4'] -= learning_rate * grad_W4 / (np.sqrt(network['h_W4']) + 1e-7)

            network['h_b4'] += grad_b4 ** 2
            network['b4'] -= learning_rate * grad_b4 / (np.sqrt(network['h_b4']) + 1e-7)

        # 각 에폭의 끝에서 비용 기록
        epoch_cost = compute_cost(network, x, t)
        costs.append(epoch_cost)
        print(f"Epoch {epoch + 1}/{epochs}, Cost: {epoch_cost}")
    return costs

# 데이터 로드
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=True)

# 하이퍼파라미터
input_size = 784
hidden1_size = 50
hidden2_size = 30
hidden3_size = 20
output_size = 10
learning_rate = 0.01
epochs = 100
batch_size = 100

# 네트워크 초기화
network = init_network(input_size, hidden1_size, hidden2_size, hidden3_size, output_size)

# 학습
costs = train(network, x_train, t_train, learning_rate, epochs, batch_size)

# 비용 그래프 시각화
import matplotlib.pyplot as plt

plt.plot(costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost Function J() over Epochs')
plt.show()

# 테스트 및 정확도 출력
correct_count = 0
for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i + batch_size]
    t_batch = t_test[i:i + batch_size]

    y_pred = predict(network, x_batch)
    predicted_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(t_batch, axis=1)

    correct_count += np.sum(predicted_labels == true_labels)

accuracy = correct_count / len(x_test)
print(f"Test Accuracy: {accuracy}")
