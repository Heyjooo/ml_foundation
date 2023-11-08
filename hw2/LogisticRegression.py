import numpy as np

class LogisticRegression():
    def __init__(self, input_data, target_output):
        # 입력 데이터와 목표 출력 초기화
        self.input_data = self.add_bias(input_data.reshape(-1, 28 * 28).astype('float32'))
        self.target_output = target_output
        # 가중치를 무작위로 초기화
        self.weights = np.random.randn(self.input_data.shape[1])

    def add_bias(self, X):
        # 입력 데이터에 bias term을 추가
        return np.c_[np.ones(X.shape[0]), X]

    def sigmoid(self, z):
        # overflow 주의하여 시그모이드 함수를 정의
        z = np.clip(z, -20, 20)
        return 1 / (1 + np.exp(-z))

    def cost(self, predictions, labels):
        # 크로스 엔트로피 비용 함수를 계산
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        m = len(labels)
        return -1 / m * np.sum(labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions))

    def learn(self, lr, epoch, label):
        # lr: 학습률
        # epoch: 학습 에포크 수
        # label: 현재 학습하려는 클래스 레이블

        y_label = (self.target_output == label).astype(int)  # 현재 클래스에 대한 레이블 생성
        cost_history = []

        for e in range(epoch):
            predictions = self.sigmoid(np.dot(self.input_data, self.weights))
            errors = y_label - predictions
            self.weights += lr * np.dot(self.input_data.T, errors)  # 가중치 업데이트
            cost_history.append(self.cost(predictions, y_label))

        return cost_history

    def predict(self, X):
        # 예측 메서드: 입력 데이터의 클래스 레이블을 예측
        X_bias = np.insert(X, 0, 1)
        return self.sigmoid(np.dot(X_bias, self.weights))