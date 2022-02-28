# Nearest Neighbor 
- Train 단계에서는 아무일도 하지 않고, 단지 모든 데이터를 기억만한다.
- Test 단계에서는 기억한 데이터중에서 가장 유사한데이터를 찾는다.
- 비교하는 방법에는 L1 Distance & L2 Distance 가 있다. ( 하기 참조 )
- Nearest Neighobors 의 단점을 보완하고자 K-Nearest Neigobors 등장 ( 가까운 이웃을 k개만큼 찾고 투표를 통해 결정 )

# Image Classifier & Loss Functions

- Manhattan distance (L1) vs Euclidean distance (L2)
- L1 Distance는 좌표계에 많은 영향을 받는다. 그래서, 벡터간 요소들이 개별적 의미를 가지고 있다면, L1 Distance가 어울린다.
![IMG_CF046D7D79F0-1](https://user-images.githubusercontent.com/98244339/150664933-7639c166-8340-4ca6-92d4-1f8f8918510a.jpeg)


# Linear Classifier : y = wx + b
![IMG_B931D09E9F5B-1](https://user-images.githubusercontent.com/98244339/150664945-1222acc0-fba4-490b-a1ce-92eb764a7529.jpeg)

# Linear Classifier 의 loss는 SVM과 Softmax 크게 두종류
- SVM Loss : MAX ( 0 , 잘못예측한 클래스 점수 - 정답 클래스 점수 +1 )의 합계의 평균 / 1은 margin 으로 설정가능 / hinge loss
- Softmax : 스코어 자체애대한 해석은 고려하지 않음 / 클래스별 로그, 지수, 표준화 적용,  Cross Entropy /  확률 결과 return , 합은 항상 1
![스크린샷 2022-02-28 오전 10 04 15](https://user-images.githubusercontent.com/98244339/155908503-5c984ed1-8e24-4c4c-b5f2-7835d49e47d8.png)
![cs231n_2017_lecture3-1-7](https://user-images.githubusercontent.com/98244339/155905775-700c6a05-3712-47b0-911f-af05913bea83.jpg)


# Regularization : Loss에 특정 값을 붙여서 Overfitting 을 방지한다.( 모델이 Train data에 완벽히 FIT 하지 못하도록 모델의 복잡도에 패널티부여 )
보통 L2 Regularization을 사용한다. 
![IMG_1D655079234F-1](https://user-images.githubusercontent.com/98244339/150665307-7b5fc5b4-0dc9-470b-8fd7-27969843164b.jpeg)
- 아래그림은 딥러닝홀로서기에서 Regularization 에 대한 부가설명으로 그린 그림인데, 이해하기 좋아 첨부
- ![IMG_F99F26FA99C8-1 복사본](https://user-images.githubusercontent.com/98244339/150665662-fd69d119-430e-41eb-9ba4-8beff355736e.jpeg)


- DropOUT : Overfitting 방지하기 위해 일정 확률로 연결끊기
- Gradient Vanishing : Activation Function 을 sigmoid 사용했을때, 입력층으로 갈수록 gradient가 업데이트가 안됨 <br /> 
→ 해결책 : " Activaitaion Function ReLU로 변경 "


# Optimization
- Gradient Descent : gradient를 계산해서 loss를 찾아간다. GD의경우 한번 업데이트 될때, 전체 Train Data를 input으로넣어 전체 error를 구하는데에 시간이 오래듬
- Stochastic Gradient Descent : 전체 데이터가 아닌 샘플만 가지고 gradient 를 
