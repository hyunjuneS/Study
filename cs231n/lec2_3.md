# Image Classifier & Loss Functions

- Manhattan distance (L1) vs Euclidean distance (L2)
![IMG_CF046D7D79F0-1](https://user-images.githubusercontent.com/98244339/150664933-7639c166-8340-4ca6-92d4-1f8f8918510a.jpeg)


- Linear Classifier : y = wx + b
![IMG_B931D09E9F5B-1](https://user-images.githubusercontent.com/98244339/150664945-1222acc0-fba4-490b-a1ce-92eb764a7529.jpeg)


- Linear Classifier 의 loss는 SVM과 Softmax 크게 두종류

ⓐ SVM Loss : MAX ( 0 , 잘못예측한 클래스 점수 - 정답 클래스 점수 +1 )의 합계의 평균 <br /> 
ⓑ Softmax : 클래스별 로그, 지수, 표준화, 적용 & Cross Entropy 
![IMG_08EE75FEC7B6-1](https://user-images.githubusercontent.com/98244339/150665381-cc7e1b2b-f622-463b-98b4-ca23508c7cd2.jpeg)


- Regularization : Loss에 특정 값을 붙여서 Overfitting 을 방지한다. 
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
