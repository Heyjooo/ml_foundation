import numpy as np

class KNN():
    def __init__(self, x_train, y_train, x_test, y_test):
        self._x_train = x_train  # 학습 데이터의 특성
        self._y_train = y_train  # 학습 데이터의 레이블
        self._x_test = x_test    # 테스트 데이터의 특성
        self._y_test = y_test    # 테스트 데이터의 실제 레이블

    def distance(self, x1, x2):
        # 두 데이터 포인트 간의 유클리디안 거리 계산
        return np.sqrt(np.sum((x1 - x2)**2))

    def predict(self, x, k=1, weighted=False):
        distance = [self.distance(x, x_train_i) for x_train_i in self._x_train]
        k_indices = np.argsort(distance)[:k]  # 거리가 가장 가까운 k개의 데이터 인덱스 선택
        k_nearest_labels = [self._y_train[i] for i in k_indices]  # k개의 가장 가까운 데이터의 레이블

        if weighted:
            # Weighted Majority Vote
            k_distances = np.array([distance[i] for i in k_indices])
            k_distances[k_distances == 0] = 1e-5  # 나누기 0을 피하기 위한 작은 값 설정
            weights = 1 / k_distances  # 가중치 계산
            weighted_votes = {}
            for label, weight in zip(k_nearest_labels, weights):
                weighted_votes[label] = weighted_votes.get(label, 0) + weight
            most_common = max(weighted_votes, key=weighted_votes.get)  # 가장 많은 가중치를 받은 레이블 선택
        else:
            # Simple Majority Vote
            most_common = np.bincount(k_nearest_labels).argmax()  # 가장 많은 투표를 받은 레이블 선택

        return most_common
