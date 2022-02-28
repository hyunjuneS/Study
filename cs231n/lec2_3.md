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
- loss function : "모델의 예측값이 정답값에 비해 얼마나 구린지" 측정
- SVM Loss : MAX ( 0 , 잘못예측한 클래스 점수 - 정답 클래스 점수 +1 )의 합계의 평균 / 1은 margin 으로 설정가능 / hinge loss
- Softmax : 스코어 자체애대한 해석은 고려하지 않음 / 클래스별 로그, 지수, 표준화 적용,  Cross Entropy /  확률 결과 return , 합은 항상 1
![cs231n_2017_lecture3-1-7](https://user-images.githubusercontent.com/98244339/155905775-700c6a05-3712-47b0-911f-af05913bea83.jpg)
![스크린샷 2022-02-28 오전 10 04 15](https://user-images.githubusercontent.com/98244339/155908503-5c984ed1-8e24-4c4c-b5f2-7835d49e47d8.png)


# Regularization : Loss에 특정 값을 붙여서 Overfitting 을 방지한다.( 모델이 Train data에 완벽히 FIT 하지 못하도록 모델의 복잡도에 패널티부여 )
- 보통 L2 Regularization을 사용한다. 
<img width="955" alt="스크린샷 2022-02-28 오전 10 36 39" src="https://user-images.githubusercontent.com/98244339/155910461-93652bf4-f931-46d2-9e4d-723d579ef5a5.png">
- 아래그림은 딥러닝홀로서기에서 Regularization 에 대한 부가설명으로 그린 그림인데, 이해하기 좋아 첨부.  
       
![IMG_F99F26FA99C8-1 복사본](https://user-images.githubusercontent.com/98244339/150665662-fd69d119-430e-41eb-9ba4-8beff355736e.jpeg)


# Optimization
- Gradient Descent : gradient를 계산해서 loss를 찾아간다. GD의경우 한번 업데이트 될때, 전체 Train Data를 input으로넣어 전체 error를 구하는데에 시간이 오래듬
- Stochastic Gradient Descent : 전체 데이터가 아닌 minibatch만 가지고 gradient를 계산해서 내려감 
- Momentum : SGD에 속도를 붙여서
- Nesterov : Momentum 은 이전속도의 영향을 받는데, 이전속도 영향 안받게 수정
- ADAGRAD : 처음에는 빠르게 학습하다가, 근처가서 천천히학습하도록함 ( gradient 의 제곱항을 더해감 )
- RMSProp : ADAGRAD 를 수정한것, ( 기존 누적값에 decay rate를 곱함 )
- ADAM : Momentum 과 ADAGRAD/RMSProp 합친것


# Network Training 
## Activation Function
- Sigmoid 가 가진 큰 문제 2가지 : Zero-Centered되지 않음  & 양/음값이 너무 커지면 기울기가 0으로 소실됨 ( Saturation )
- Zero-Centered : Activation Function 이 Zero-Centered 되어있지 않으면, gradient 값이 음수 or 양수로 치우쳐서 나온다.
- tanh 는 Zero-centered 되어 Simgoid 문제 많이 해결하지만, Saturation 때문에 여전히 Gradient 가 죽음 ( RNN에서 주로 쓰임 )
- ReLU 는 양의값에서는 saturation 되지않음(가장 큰 장점) , 계산 빠름 ( 매우쉬워, 잘 사용 , 신경과학적인 측면에서도 많이사용 ) , non - zerocentered / 음의경우에서 saturation 됨 
- dead ReLu : relu에서 절반만 active됨, 즉 일부만 update 되고 일부는 안됨 , 너무 날뛰는 경우도 발생, ==> Leaky Relu 나옴
- Gradient Vanishing : Activation Function 을 sigmoid 사용했을때, 입력층으로 갈수록 gradient가 업데이트가 안됨 <br /> 
   → 해결책 : " Activaitaion Function ReLU로 변경 "
![IMG_D0978629FB03-1](https://user-images.githubusercontent.com/98244339/150666080-27c36d73-b052-4125-a58d-09ca20913e7a.jpeg)

## Weight Initialization 가중치의 초기화가 반드시 필요함, 
- 초기에 weigth를 Gaussian 함수로 초기화 시켜도, Network가 깊어지면 문제가 생김
- 해결하기 위해 Xavier Initialization 을 사용
![IMG_15037652D26F-1](https://user-images.githubusercontent.com/98244339/150666917-e1ddf3ff-9bb3-42ca-b5f9-d814bbc4550d.jpeg)

## Batch Normalization : 네트워크 중간중간에 input 값들에 대해 Normalization 을 해줘야한다.
![IMG_63385333D2D2-1](https://user-images.githubusercontent.com/98244339/150666951-9b117b3f-7e05-461a-81b7-9cb947415e51.jpeg)

## Hyperparameter Optimization : 경험적으로 하는것이 대부분, 처음에는 넓은 범위에서 좁은범위로 좁혀가며 실행한다.
- Grid Search 와 Random Search 있지만, Random 이 성능 더 좋음
![IMG_04FBD192C240-1](https://user-images.githubusercontent.com/98244339/150667069-fdfb9900-d3c4-40fa-bf71-bff1f9b2e12b.jpeg)

## ETC..
- DropOUT : Overfitting 방지하기 위해 일정 확률로 연결끊기
- Data Augentation : CNN에서 데이터 약간씩 변형 ( ex. 반전 )


# CNN
- filter란?
<img width="669" alt="스크린샷 2022-02-02 오후 5 43 46" src="https://user-images.githubusercontent.com/98244339/152121214-5fd0d2bd-6d90-4d11-bafa-24bb6822401c.png">
- stride : filter 가 옮겨가는 보폭
- padding : 가장자리에 값들이 무시되지 않도록 0 을 추가
- pooling : fiter가 slide 되면서 계산한것중 최댓값 혹은 평균을 잡아서 사이즈 줄이고 특징을 잡아낸다. ( 일종의 Downsample )
![IMG_34B68039D87F-1](https://user-images.githubusercontent.com/98244339/150666461-65903c13-20db-4ab6-85c1-62b1c02bf1b3.jpeg)

## 중요예제 문제
- Input 7*7 , Filter 3*3 , Stride 1 , padding 1 ===> what is output?
- ANSWER IS : ( N - F ) / Stride + 1  = 7 
- N : 7 + 2 ( padding 1 이 2개 ) & F : 3  & Stride 1.
![연습장-11](https://user-images.githubusercontent.com/98244339/155914003-498ebfa8-f1cd-4544-b32c-f3b4ee4b4476.jpg)
## 실제 Assignment2 에서 구현했던 N=32 , F=5, Pad=2 , Stride =1  => output 32 나옴 




# RNN
- Truncated Backpropagation : batch 만큼 학습시키는것  ( gradient를 근사시키는것 )





