# 헷갈린 의미 정리

## Epoch vs Iteration
- Epoch : 모든 Training Data를 가지고 파라미터 업데이트 한번 하면 1 epoch
- Iteration : Minibatch 가 업데이트 되면 1 Iteration
( EX. Training Data : 10000개 , mini-batch : 100개 
      → 100 iteration = 1 epoch )



# 까먹는 Python Code ( Numpy or Tensor ... )

- x = np.linspace(0,10,11) : 0부터 10까지 숫자를 11개로 나누어서 배열로 넣는다 [0,1,2,3,4,5,6,7,8,9,10]
- assert 가정문 : assert 뒤에 조건문이 참이 아니면 에러발생시킴 
- np.random.randn(3,2) : 3행 2열로, 0~1가우시안 분포값들 랜덤으로 뽑아냄
- np.random.randint(3 , size = (3,2)) : 3행2열로 0,1,2 값 랜덤으로 추출
- np.random.normal( loc , scale , size. ) : loc 평균, scale : 표준편차 , size 사이즈



