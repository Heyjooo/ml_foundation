import numpy as np
from collections import Counter, defaultdict

class KNN:
    # KNN 클래스 생성, k의 기본값 3으로 설정
    def __init__(self, k=3):
        self.k = k

    # 두 점 사이의 유클리디안 거리를 계산해주는 함수
    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    # 주어진 테스트 데이터에 대한 레이블을 예측하여 반환하는 함수
    def predict(self, X_train, y_train, test_data):
        # 각 훈련 데이터 포인트와의 거리를 계산
        distance = [(self.euclidean_distance(test_data, x), y) for x, y in zip(X_train, y_train)]
        # 거리를 기준으로 정렬
        sorted_distance = sorted(distance, key=lambda x: x[0])
        # 가장 가까운 k개의 레이블 찾기
        k_nearest_label = [item[1] for item in sorted_distance[:self.k]]
        # 이 중에서 가장 많은 레이블을 예측값으로 반환
        most = Counter(k_nearest_label).most_common(1)
        return most[0][0]

    # 가중치를 사용하여 테스트 데이터에 대한 레이블을 예측하여 반환하는 함수
    def weighted_predict(self, X_train, y_train, test_data):
        # 테스트 포인트와 모든 훈련 데이터 포인트 사이의 거리 계산
        distance = [(self.euclidean_distance(test_data, x), y) for x, y in zip(X_train, y_train)]
        # 거리를 기준으로 정렬
        sorted_distance = sorted(distance, key=lambda x: x[0])
        # 가장 가까운 k개의 데이터 포인트 선택
        k_nearest = sorted_distance[:self.k]
        vote = defaultdict(float)
        for distance, label in k_nearest:
            # 거리가 0인 경우, 해당 레이블을 직접 반환
            if distance == 0:
                return label
            # 가중치를 계산 (거리의 역수)
            weight = 1 / distance
            vote[label] += weight

        # 가장 높은 가중치를 가진 레이블 반환
        return max(vote, key=vote.get)