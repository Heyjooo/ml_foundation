import time
from dataset.mnist import load_mnist
from KNN import KNN

# MNIST 데이터셋을 로드하고 300부터 399까지의 데이터를 선택하여 테스트용 데이터와 레이블을 설정
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True)
x_test_subset = x_test[:100]
y_test_subset = y_test[:100]

# KNN 모델을 초기화
classification = KNN(x_train, y_train, x_test_subset, y_test_subset)

# 단순 KNN
start_time = time.time()
# 각 테스트 데이터에 대해 k=1,3,5,7,9로 단순 KNN을 사용하여 예측
predictions = [classification.predict(x, k=1) for x in x_test_subset]
end_time = time.time()
# 예측 정확도 계산
accuracy = sum(1 for p, y in zip(predictions, y_test_subset) if p == y) / len(y_test_subset)
print("Simple KNN")
print("Actual vs Predicted (100 samples):")
for actual, pred in zip(y_test_subset[:100], predictions[:100]):
    print(f"Actual: {actual}, Predicted: {pred}")
print(f"Fit Time: {end_time - start_time:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# 가중치를 적용한 KNN
start_time = time.time()
# 각 테스트 데이터에 대해 k=1,3,5,7,9로 가중치를 적용한 KNN을 사용하여 예측
predictions_weighted = [classification.predict(x, k=1, weighted=True) for x in x_test_subset]
end_time = time.time()
# 가중치를 적용한 예측 정확도 계산
accuracy_weighted = sum(1 for p, y in zip(predictions_weighted, y_test_subset) if p == y) / len(y_test_subset)
print("\nWeighted KNN:")
print("Actual vs Predicted (100 samples):")
for actual, pred in zip(y_test_subset[:100], predictions_weighted[:100]):
    print(f"Actual: {actual}, Predicted: {pred}")
print(f"Fit Time: {end_time - start_time:.2f}")
print(f"Accuracy: {accuracy_weighted:.2f}")