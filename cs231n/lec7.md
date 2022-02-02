- Momentum : SGD에 속도를 붙여서
- Nesterov : Momentum 은 이전속도의 영향을 받는데, 이전속도 영향 안받게 수정
- ADAGRAD : 처음에는 빠르게 학습하다가, 근처가서 천천히학습하도록함 ( gradient 의 제곱항을 더해감 )
- RMSProp : ADAGRAD 를 수정한것, ( 기존 누적값에 decay rate를 곱함 )
- ADAM : Momentum 과 ADAGRAD/RMSProp 합친것


- Data Augentation : CNN에서 데이터 약간씩 변형 ( ex. 반전 )

- filter란?
<img width="669" alt="스크린샷 2022-02-02 오후 5 43 46" src="https://user-images.githubusercontent.com/98244339/152121214-5fd0d2bd-6d90-4d11-bafa-24bb6822401c.png">
- stride : filter 가 옮겨가는 보폭
- padding : 가장자리에 값들이 무시되지 않도록 0 을 추가
- pooling : filter 로 계산한 값들중, 특성치를 뽑기위해 max / average 값을 가져오는것 
- 
