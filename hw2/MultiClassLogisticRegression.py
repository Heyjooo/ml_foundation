import numpy as np

class MultiClassLogisticRegression():
    def __init__(self, input_data, target_output, num_classes):
        # 입력 데이터와 목표 출력 초기화
        self.input_data = self.add_bias(input_data.reshape(-1, 28 * 28).astype('float32'))
        self.target_output = target_output  # 클래스 레이블 (0부터 num_classes - 1까지의 정수)
        self.num_classes = num_classes
        self.weights = np.random.randn(self.input_data.shape[1], num_classes)

    def add_bias(self, X):
        # 입력 데이터에 bias term을 추가
        return np.c_[np.ones(X.shape[0]), X]

    def softmax(self, z):
        # Softmax 함수를 사용하여 클래스별 확률을 계산
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def cost(self, predictions, labels):
        m = len(labels)
        # 다중 클래스 로지스틱 회귀 비용 함수 계산
        epsilon = 1e-15  # 아주 작은 상수 (0에 가까운 값)
        clipped_predictions = np.clip(predictions, epsilon, 1 - epsilon)  # 예측 확률을 클리핑
        individual_costs = -np.log(clipped_predictions[range(m), labels])
        return np.mean(individual_costs)

    def learn(self, lr, epoch):
        cost_history = []

        for e in range(epoch):
            predictions = self.softmax(np.dot(self.input_data, self.weights))
            errors = predictions - (np.arange(self.num_classes) == self.target_output[:, None])
            self.weights -= lr * np.dot(self.input_data.T, errors)
            cost_history.append(self.cost(predictions, self.target_output))

        return cost_history

    def predict(self, X):
        # 예측 메서드: 입력 데이터의 클래스 레이블을 예측
        X_bias = self.add_bias(X)
        predictions = self.softmax(np.dot(X_bias, self.weights))
        return np.argmax(predictions, axis=1)

