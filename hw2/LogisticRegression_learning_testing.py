from dataset.mnist import load_mnist
from LogisticRegression import LogisticRegression
from MultiClassLogisticRegression import MultiClassLogisticRegression
import matplotlib.pyplot as plt
import numpy as np

# 클래스 번호 설정 (0부터 9 중 원하는 클래스 넣기)
class_num = 9

def calculate_accuracy(clf, x_test, y_test, class_num):
    # 해당 클래스에 대한 예측
    predictions = [1 if clf.predict(x) >= 0.5 else 0 for x in x_test]
    # 실제 클래스가 해당 클래스인 데이터 중에서 예측이 맞는 경우 (True Positive)
    TP = sum([1 for pred, y in zip(predictions, y_test) if y == class_num and pred == 1])
    # 실제 클래스가 해당 클래스가 아닌데, 해당 클래스가 아니라고 맞게 예측한 경우 (True Negative)
    TN = sum([1 for pred, y in zip(predictions, y_test) if y != class_num and pred == 0])
    # 정확도 계산 (TP + TN) / 전체 데이터 수
    accuracy = (TP + TN) / len(y_test)

    return accuracy

# Load data
(x_train, y_train), (x_test, y_test) = load_mnist(flatten=True, normalize=True)

# 학습 관련 매개변수 설정
learning_rate = 0.0001
epochs = 200

# 모델 초기화 (단일 클래스에 대한 로지스틱 회귀 모델)

classification = LogisticRegression(x_train, y_train)

# 클래스에 대한 비용을 저장할 리스트 초기화
class_costs = []

# 모델 학습 및 에포크 비용 수집
for e in range(epochs):
    costs = classification.learn(lr=learning_rate, epoch=1, label=class_num)
    class_costs.append(costs[0])

    print(f"epoch: {e} cost: {costs[0]:.11f}")

# 해당 클래스에 대한 정확도 계산
accuracy = calculate_accuracy(classification, x_test, y_test, class_num)
print(f"Accuracy = {accuracy:.7f}")

plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title(f'Cost for Class {class_num} (Logistic Regression) During Training')
plt.plot(range(epochs), class_costs)
plt.show()
print("\n")

# 다중 클래스 로지스틱 회귀 모델 학습
num_classes = 10
classification_multi = MultiClassLogisticRegression(x_train, y_train, num_classes)

# 클래스에 대한 비용을 저장할 리스트 초기화
class_costs_multi = [[] for _ in range(num_classes)]
accuracies_multi = []

# 모델 학습 및 에포크 비용 수집
for e in range(epochs):
    for class_num in range(num_classes):
        costs_multi = classification_multi.learn(lr=learning_rate, epoch=1)
        class_costs_multi[class_num].append(costs_multi[0])

# 각 클래스에 대한 에포크 0부터 99까지의 비용 출력
for e in range(epochs):
    costs_str = " ".join([f"{class_costs_multi[class_num][e]:.5f}" for class_num in range(num_classes)])
    print(f"epoch: {e} cost: [{costs_str}]")

# 각 클래스에 대한 정확도 계산
for class_num in range(num_classes):
    accuracy_multi = calculate_accuracy(classification, x_test, y_test, class_num)
    accuracies_multi.append(accuracy_multi)

# 평균 정확도 계산
mean_accuracy_multi = sum(accuracies_multi) / num_classes
print(f"Accuracy = {mean_accuracy_multi:.7f}")

# 각 클래스에 대한 비용 그래프 그리기
plt.figure(figsize=(12, 8))
colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))  # 색상 설정
for class_num in range(num_classes):
    plt.plot(range(epochs), class_costs_multi[class_num][:epochs], label=f'Class {class_num}', color=colors[class_num])
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost for All Classes (Multi-Class Logistic Regression)')
plt.legend()
plt.tight_layout()
plt.show()