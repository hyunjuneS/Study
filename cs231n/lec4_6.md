# CNN

- Activation Function
![IMG_D0978629FB03-1](https://user-images.githubusercontent.com/98244339/150666080-27c36d73-b052-4125-a58d-09ca20913e7a.jpeg)

- POOLING : fiter가 slide 되면서 계산한것중 최댓값 혹은 평균을 잡아서 사이즈 줄이고 특징을 잡아낸다.
![IMG_34B68039D87F-1](https://user-images.githubusercontent.com/98244339/150666461-65903c13-20db-4ab6-85c1-62b1c02bf1b3.jpeg)

- Weight Initialization : 가중치의 초기화가 반드시 필요함, 
( 초기에 weigth를 Gaussian 함수로 초기화 시켜도, Network가 깊어지면 문제가 생김.)
해결하기 위해 Xavier Initialization 을 사용
![IMG_15037652D26F-1](https://user-images.githubusercontent.com/98244339/150666917-e1ddf3ff-9bb3-42ca-b5f9-d814bbc4550d.jpeg)

- Batch Normalization : 네트워크 중간중간에 input 값들에 대해 Normalization 을 해줘야한다.
![IMG_63385333D2D2-1](https://user-images.githubusercontent.com/98244339/150666951-9b117b3f-7e05-461a-81b7-9cb947415e51.jpeg)

- Hyperparameter Optimization : 경험적에 하는것이 대부분이고, 처음에는 넓은범위에서 좁은범위로 좁혀가며 실행
Grid Search 와 Random Search 있지만, Random 이 성능 더 좋음
![IMG_04FBD192C240-1](https://user-images.githubusercontent.com/98244339/150667069-fdfb9900-d3c4-40fa-bf71-bff1f9b2e12b.jpeg)




