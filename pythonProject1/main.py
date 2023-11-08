from sklearn.datasets import load_iris
from KNN import KNN
import numpy as np

# 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 지정한 index들을 테스트 데이터로 사용 (전체의 15분의 1만 test data, 나머지는 모두 train data)
# test_index 에는 14, 29, 44, ... 등의 index가 포함되며, 나머지는 train_index에 포함
test_index = list(range(14, len(X), 15))
train_index = [i for i in range(len(X)) if i not in test_index]

X_train = X[train_index]
y_train = y[train_index]
X_test = X[test_index]
y_test = y[test_index]

# KNN 클래스를 사용하여 kNN 모델을 생성
knn_classification = KNN(k=7)
# 테스트 데이터에 대한 예측 수행
y_pred = [knn_classification.predict(X_train, y_train, test_data) for test_data in X_test]

# Weighted kNN 알고리즘 사용하여 테스트 데이터에 대한 에측 수행
y_weighted_pred = [knn_classification.weighted_predict(X_train, y_train, test_data) for test_data in X_test]

# kNN을 이용하여 얻은 예측값과 실제값 비교
print("<Compare KNN Computed class and True class>")
for pred, true in zip(y_pred, y_test):
    pred_name = iris.target_names[pred]
    true_name = iris.target_names[true]
    print(f"Computed class: {pred_name}, True class: {true_name}")

print("\n")  # 줄 바꿈

# Weighted kNN을 이용하여 얻은 예측값과 실제값 비교
print("<Compare Weighted KNN Computed class and True class>")
for pred, true in zip(y_weighted_pred, y_test):
    pred_name = iris.target_names[pred]
    true_name = iris.target_names[true]
    print(f"Computed class: {pred_name}, True class: {true_name}")

print("\n")  # 줄 바꿈

# kNN 정확도 계산
accuracy_kNN = np.mean(np.array(y_pred) == y_test)
print(f"kNN 정확도: {accuracy_kNN:.2f}")

# Weighted kNN 정확도 계산
accuracy_weighted_kNN = np.mean(np.array(y_weighted_pred) == y_test)
print(f"Weighted kNN 정확도: {accuracy_weighted_kNN:.2f}")