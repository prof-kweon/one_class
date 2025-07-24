# one_class
2025년 1기 원클래스

--------------------------------------------------------------------------------------------
1. TensorFlow 예제 (MNIST 손글씨 숫자 분류)
   MNIST 데이터로 손글씨 분류, 학습 정확도 그래프 확인 가능.
   
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# 데이터 불러오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()  # MNIST 데이터셋 로드
x_train, x_test = x_train / 255.0, x_test / 255.0         # 픽셀 값을 0~1로 정규화

# 모델 생성
model = tf.keras.models.Sequential([                        # 순차 모델 생성
    tf.keras.layers.Flatten(input_shape=(28, 28)),         # 28x28 이미지를 1D로 변환
    tf.keras.layers.Dense(128, activation='relu'),         # 은닉층: 128 뉴런, ReLU 활성화
    tf.keras.layers.Dense(10, activation='softmax')        # 출력층: 10 클래스, softmax
])

# 컴파일 & 학습
model.compile(optimizer='adam',                            # 최적화 알고리즘: Adam
              loss='sparse_categorical_crossentropy',      # 손실 함수: 다중 클래스
              metrics=['accuracy'])                        # 평가지표: 정확도
history = model.fit(x_train, y_train, epochs=3,            # 3번 반복 학습
                    validation_data=(x_test, y_test))      # 검증 데이터 설정

# 학습 결과 시각화
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Test Acc')
plt.title("TensorFlow: MNIST Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

--------------------------------------------------------------------------------------------
2. Keras 예제 (단순 선형회귀)
   설명: 단순한 회귀 모델로 X와 y의 관계를 학습하고 직선을 그림.
   
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성: y = 2x + 1
X = np.linspace(0, 1, 100)                              # 0~1 사이 100개 점
y = 2 * X + 1 + np.random.randn(100) * 0.1             # 노이즈 추가

# 모델 생성
model = Sequential([Dense(1, input_dim=1)])            # 입력 1, 출력 1
model.compile(optimizer='sgd', loss='mse')             # 확률적 경사하강법, 손실 MSE

# 학습
model.fit(X, y, epochs=100, verbose=0)                 # 100 에포크 학습

# 예측
y_pred = model.predict(X)

# 시각화
plt.scatter(X, y, label='Data')                        # 실제 데이터
plt.plot(X, y_pred, color='red', label='Prediction')   # 예측 선
plt.title("Keras: Linear Regression")
plt.legend()
plt.show()

--------------------------------------------------------------------------------------------
3. PyTorch 예제 (선형회귀)
   설명: PyTorch를 사용해 선형 회귀 학습, 예측 결과 시각화.

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 데이터: y = 2x + 1
X = torch.linspace(0, 1, 100).unsqueeze(1)            # 0~1 사이 100개, (100,1) 형태
y = 2 * X + 1 + torch.randn(X.size()) * 0.1           # 노이즈 추가

# 모델 정의
model = nn.Linear(1, 1)                               # 입력 1, 출력 1
criterion = nn.MSELoss()                               # 손실: 평균제곱오차
optimizer = optim.SGD(model.parameters(), lr=0.1)     # SGD 최적화, 학습률 0.1

# 학습
for epoch in range(100):
    optimizer.zero_grad()                              # 기울기 초기화
    output = model(X)                                  # 예측
    loss = criterion(output, y)                        # 손실 계산
    loss.backward()                                    # 역전파
    optimizer.step()                                   # 가중치 업데이트

# 예측
pred = model(X).detach()                               # 그래프 분리 후 예측

# 시각화
plt.scatter(X.numpy(), y.numpy(), label='Data')
plt.plot(X.numpy(), pred.numpy(), color='red', label='Prediction')
plt.title("PyTorch: Linear Regression")
plt.legend()
plt.show()

--------------------------------------------------------------------------------------------
4. Scikit-learn 예제 (Iris 분류)
   설명: Iris 데이터로 다중 클래스 로지스틱 회귀 모델 학습 후 평가.

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 데이터 로드
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
                                                    test_size=0.2, random_state=42)

# 모델 학습
model = LogisticRegression(max_iter=200)             # 반복 200회 설정
model.fit(X_train, y_train)                          # 학습
y_pred = model.predict(X_test)                       # 예측

# 혼동행렬 시각화
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot()
plt.title("Scikit-learn: Iris Confusion Matrix")
plt.show()


--------------------------------------------------------------------------------------------



--------------------------------------------------------------------------------------------
