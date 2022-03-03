# DeepLearning Basic

## Nearest Neighbor 
- Train 단계에서는 아무일도 하지 않고, 단지 모든 데이터를 기억만한다.
- Test 단계에서는 기억한 데이터중에서 가장 유사한데이터를 찾는다.
- 비교하는 방법에는 L1 Distance & L2 Distance 가 있다. ( 하기 참조 )
- Nearest Neighobors 의 단점을 보완하고자 K-Nearest Neigobors 등장 ( 가까운 이웃을 k개만큼 찾고 투표를 통해 결정 )

## Manhattan distance (L1) vs Euclidean distance (L2)
- L1 Distance는 좌표계에 많은 영향을 받는다. 그래서, 벡터간 요소들이 개별적 의미를 가지고 있다면, L1 Distance가 어울린다.
![IMG_CF046D7D79F0-1](https://user-images.githubusercontent.com/98244339/150664933-7639c166-8340-4ca6-92d4-1f8f8918510a.jpeg)

## Linear Classifier : y = wx + b
![IMG_B931D09E9F5B-1](https://user-images.githubusercontent.com/98244339/150664945-1222acc0-fba4-490b-a1ce-92eb764a7529.jpeg)

## Loss Function : SVM과 Softmax 크게 두종류
- loss function : "모델의 예측값이 정답값에 비해 얼마나 구린지" 측정
- SVM Loss : MAX ( 0 , 잘못예측한 클래스 점수 - 정답 클래스 점수 +1 )의 합계의 평균 / 1은 margin 으로 설정가능 / hinge loss
- Softmax : 스코어 자체애대한 해석은 고려하지 않음 / 클래스별 로그, 지수, 표준화 적용,  Cross Entropy /  확률 결과 return , 합은 항상 1
![cs231n_2017_lecture3-1-7](https://user-images.githubusercontent.com/98244339/155905775-700c6a05-3712-47b0-911f-af05913bea83.jpg)
![스크린샷 2022-02-28 오전 10 04 15](https://user-images.githubusercontent.com/98244339/155908503-5c984ed1-8e24-4c4c-b5f2-7835d49e47d8.png)


## Regularization 
- Loss에 특정 값을 붙여서 Overfitting 을 방지한다.( 모델이 Train data에 완벽히 FIT 하지 못하도록 모델의 복잡도에 패널티부여 )
- 보통 L2 Regularization을 사용한다.         
<img width="955" alt="스크린샷 2022-02-28 오전 10 36 39" src="https://user-images.githubusercontent.com/98244339/155910461-93652bf4-f931-46d2-9e4d-723d579ef5a5.png">
- 아래그림은 딥러닝홀로서기에서 Regularization 에 대한 부가설명으로 그린 그림인데, 이해하기 좋아 첨부.

![IMG_F99F26FA99C8-1 복사본](https://user-images.githubusercontent.com/98244339/150665662-fd69d119-430e-41eb-9ba4-8beff355736e.jpeg)


## Optimization
- Gradient Descent : gradient를 계산해서 loss를 찾아간다. GD의경우 한번 업데이트 될때, 전체 Train Data를 input으로넣어 전체 error를 구하는데에 시간이 오래듬
- Stochastic Gradient Descent : 전체 데이터가 아닌 minibatch만 가지고 gradient를 계산해서 내려감, 파라미터가 많아졌을때 불균형한 방향이 증가하여 매우 복잡해지고,로컬 minimum 또는 변곡점에서 멈춤 
- SGD with Momentum : SGD에 속도를 붙여서 변곡점이나 local minima에 대응
- Nesterov : Momentum 은 이전속도의 영향을 받는데, 이전속도 영향 안받게 수정 ( velocity 방향으로 이동한 후에, gradient를 계산하는 방식 )
- ADAGRAD : 처음에는 빠르게 학습하다가, 근처가서 천천히학습하도록함 ( gradient 의 제곱항을 누적해서 더해감 )
- RMSProp : ADAGRAD 를 수정한것, ( gradient 제곱항 누적값에 decay rate를 곱함 )
- ADAM : Momentum 과 ADAGRAD/RMSProp 합친것


## ============================================

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
- 해결하기 위해 Xavier Initialization 을 사용 ( 입출력의 분산을 맞춰줌 )
- ReLU 에서는 잘 작동하지 않음 ( 반을 죽이니깐 ) / 추가적으로 여러 방법이있는데, 표준편차에서 강의노트에서는 2를 나눠준다
![IMG_15037652D26F-1](https://user-images.githubusercontent.com/98244339/150666917-e1ddf3ff-9bb3-42ca-b5f9-d814bbc4550d.jpeg)

## Batch Normalization : 네트워크 중간중간에 input 값들에 대해 Normalization 을 해줘야한다.
![IMG_63385333D2D2-1](https://user-images.githubusercontent.com/98244339/150666951-9b117b3f-7e05-461a-81b7-9cb947415e51.jpeg)

## Hyperparameter Optimization : 경험적으로 하는것이 대부분, 처음에는 넓은 범위에서 좁은범위로 좁혀가며 실행한다.
- Grid Search 와 Random Search 있지만, Random 이 성능 더 좋음
![IMG_04FBD192C240-1](https://user-images.githubusercontent.com/98244339/150667069-fdfb9900-d3c4-40fa-bf71-bff1f9b2e12b.jpeg)

## lr decaying 
- 위에 Optimization 에는 모두 learning rate 라는 하이퍼파라미터가 존재한다. 
- lr 하이퍼 파라미터를 잘 잡는것은 매우 어려운일이고 중요한일이다.
- lr 점점 줄여가는 방법도 사용해 볼 수 있다.

![스크린샷 2022-02-28 오후 3 42 49](https://user-images.githubusercontent.com/98244339/155936524-a75f794f-a510-4f90-b63c-534debc6b516.png)

## Second-order Optimization 
- first-order optimization 은 1차 미분값으로 크게 변화할 수 없다.
![스크린샷 2022-02-28 오후 3 41 12](https://user-images.githubusercontent.com/98244339/155936354-e7366e80-b097-452e-a381-9beab5ee38a3.png)
- secon-order optimization 에서는 taylor 근사함수를 써서 minima에 더 잘 접근가능하다.
![스크린샷 2022-02-28 오후 3 41 34](https://user-images.githubusercontent.com/98244339/155936375-f93b4823-60e8-44c1-97da-0939b6772df5.png)

## Model Ensembles : 모델을 n개 학습시켜서 결과의 평균을 이용하자 

## Dropout : Regularization의 일종으로 Neural Net에서는 L2 보다, Dropout 이용 / forward pass 과정에서 임의로 일부뉴런을 0으로 만드는것 
![스크린샷 2022-02-28 오후 3 50 40](https://user-images.githubusercontent.com/98244339/155937457-b2b84260-4d30-42eb-9d7e-74bae401f2df.png)

## Transfer Learning 
- 기존에 개발된 레이어의 features를 가지고와서, 마지막 FC layer 초기화, 가중치행렬 초기화, 모든 레이어들의 가중치 freeze

## Data Augentation : CNN에서 데이터 약간씩 변형 ( ex. 반전 )


## ============================================


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
## 실제 Assignment2 에서 구현했던 N=32 , F=5, Pad=2 , Stride =1  => output 32 나옴 

![연습장-11](https://user-images.githubusercontent.com/98244339/155914003-498ebfa8-f1cd-4544-b32c-f3b4ee4b4476.jpg)


## ============================================


# CNN Architecture

## AlexNet
- 최초의 Large Scale CNN
- Convnet 연구 돌풍일으킴
- 파라미터 60M 개 
![스크린샷 2022-03-01 오전 8 54 15](https://user-images.githubusercontent.com/98244339/156078637-bbd590b1-a37b-4858-a280-9600e49196cf.png)

## VGGNet
- 더 깊어지고 더 작은 filter 사용.
- 메모리 사용량이 많은 편이라고 함.
- local Response Normalization 을 사용하지 않음 ( AlexNet 에서 사용되었음 ) 
- 16 or 19 layer 
- 3*3 filter 주로사용하고, pooling 사용 ( 작은 fiilter 사용하는이유 : 파라미터 수가 적고, depth 키울 수 있음. )
- 파라미터 128M 개
![스크린샷 2022-03-01 오전 9 02 44](https://user-images.githubusercontent.com/98244339/156079379-eb10fc34-b54d-4df1-93a8-23d3088a58b6.png)

## GoogleNet
- 22 layer
- FC layer 가 없음 ( 파라미터 줄이기 위해, 파라미터 5M개 )
- Inception module을 여러개 쌓아서 만듬
  ( Inception module ? : network 내의 network 개념으로 local network 라고 함 )
  ( 서로다른 필터들이 병렬로 계산하고, 한번에 합침 -> 계산량이 많아짐.. )
  Bottleneck Layers의 사용 ( 계산량이 많아지는 문제 -> 각 레이어의 계산량을  1*1 conv 를 통해서 dimension 을 줄여준다 )
  ![스크린샷 2022-03-01 오전 9 25 15](https://user-images.githubusercontent.com/98244339/156081354-c7d4a07a-645a-414b-8397-6f7302d2d82a.png)
![연습장-13](https://user-images.githubusercontent.com/98244339/156082013-1db3da9b-82ab-4511-b2ca-9d12bf4ab131.jpg)

## ResNet
- 매우 깊은 네트워크 152layer
- 가설 : 모델이 깊어질수록 최적화가 어렵다.
- Residual block : h(x) = f(x) + x 로두고, h(x) 학습시키는것은 어려우니 f(x) 학습시키고 x를 더하자 ( 입력값을 어떻게 수정해야하는지 보는것임 ) 
- Residual block 에서 모델 depth 가 50 이상일때는 GoogleNet과 같은 Bottleneck Layer 사용하자
- 아래와같은 reusiauel block 을 쌓아올리자 
![연습장-14](https://user-images.githubusercontent.com/98244339/156083242-b206b2a7-6f54-4242-b3d4-6b4fdb0b1711.jpg)
![스크린샷 2022-03-01 오전 9 54 42](https://user-images.githubusercontent.com/98244339/156084146-2331d052-1f63-42e9-9470-6b1727374012.png)


## ============================================

# RNN
- One to Many : Image Captioning
- Many to One : 문장의 감성분석
- Many to Many ( 입/출력 길이 다양 ) : Translation
- Many to Many ( 입/출력 길이 고정 ) : 프레임 수준에서의 비디오 분류
![다운로드](https://user-images.githubusercontent.com/98244339/156092212-37c0c9c5-45f5-4b94-a8e4-3c6a39b51ea2.png)

## Truncated Backpropagation : batch 만큼 학습시키는것  ( gradient를 근사시키는것 )
![스크린샷 2022-03-01 오전 11 26 49](https://user-images.githubusercontent.com/98244339/156093004-d9e9f1a2-e220-4cef-865d-b8c1a3915ec3.png)

## Image Captioning
- 그림의 정보를 CNN을 이용해서 이미지 정보를 요약하고, RNN을 통과시켜 문장을 만드는 구조
- 마지막 FC Layer 없애고, 그전에 4096 dim 을 가져와서 세번째 가중치 행렬을 더해서 다음 hidden 을 계산한다.
- end가 나올때까지 RNN 구조 반복 
![스크린샷 2022-03-01 오전 11 40 31](https://user-images.githubusercontent.com/98244339/156094381-648150e4-5101-4fa5-879c-92fe7b3446fe.png)

## LSTM 
- 한 cell당 2개의 hidden state 가 있음   
  ( h_t : 기존 RNN의 hidden )
  ( c_t : LSTM내부에만 존재하며 노출되지 않음 )
- ⓐ input x_t , h_t-1 을 받고 
- ⓑ h_t-1 과 x_t 와 큰 가중치 행렬을 곱하여 4개의 gate (i,f,o,g) 생성
     ( i : input gate , i 는 현재 입력 x_t에 대한 가중치 (cell에 얼마나 현재 input을 넣을지) , sigmoid 사용 )           
     ( g : gate gate , input cell을 얼마나 포함시킬지, tanh 사용 )            
     ( f : forget gate , 이전 스텝의 cell 정보를 얼마나 망각할지 , sigmoid 사용 )            
     ( o : output gate , c_t를 얼마나 밖으로 드러내보일지 (각 스템에서 hidden state를 계산할때 cell을 얼마나 노출시킬지), sigmoid 사용 )           
- ⓒ (i,f,o,g)를 이용하여 c_t 업데이트 -> c_t 이용하여 h_t 업데이트
![연습장-15](https://user-images.githubusercontent.com/98244339/156103249-eacce0f1-834b-4788-9ddb-4f52d18785e1.jpg)
![연습장-16](https://user-images.githubusercontent.com/98244339/156103259-d4adce7a-3bdd-4411-828a-1e912c647baf.jpg)


# Detection & Segmentation
![image](https://user-images.githubusercontent.com/98244339/156113335-4954c44c-d7c8-4d50-b06f-d042cea7f49c.png)

## Sementic Segmentation 
- 주로 마지막 FC-Layer 제외하고 conv만 써서 하는데, conv 하면서 pooling&stride 같은 downsampling 이 주로 쓰여서, Upsampling이 필요함
- Upsampling 을 하는 여러방법들이 있음 ( Unpooling , upsampling , Transpose convolution ) 
- Downsampling 은 strided convolution 이나 pooling 
 
## Classification + Localization 
- Localization 은 이미지내에 객체가 오직 한개라고 가정함
- FC Layer 하나 더 두고, output dimension 4로 함 ( width,height,x,y )

## Object Detection 
- 고양이, 개 물고기 등 고정된 카테고리 갯수만 고려
- 특정범위를 관찰하여 classification 하는 방법 -> 특정범위를 어떻게 잡을건데? 매우 난감...
- Region Proposal Network : Object 가 있을법한 BBox 제공
   ( Selective Search : 2000개의 Region Proposal 만듬 , 2000개 만들고서 input dimension은 고정해줘야함)                
   ( Region Proposal 가지고 CNN 하는것이 R-CNN )        
   ( ConvNet 통과시키고 Feature map 가지고 Region Proposal 만드는것이 Fast R-CNN )         
- YOLO : 입력이미지를 큼지막히 나눔, 각 셀에 Base BBox가 세가지 있고, 길쭉 넓죽 정사각형 , 으로 classification하면서 BBox 변형

## Instance Segmentation
- Sementic Segmentation 과 Object Detection 합작


## ============================================

# Visualizaing and Understanding 
- CNN의 filter가 이미지에 어떠한 특이점을 추출하는지 봄

## Occlusion Experiments
- 이미지에 특정 patch 가린것을 stride해가면서, patch의 위치에따른 prediction 확률변화를 봄
- 어디에 patch가 위치할때 스코어가 크게변하면, 그 부분이 classification 에 중요한 부분이었다는것을 짐작

## Saliency Maps
- 입력이미지를 classification 한 뒤에, 네트워크가 어떤 픽셀들을 보고서 classification 했는지 찾기위함
- 입력이미지의 각 픽셀들에 대해, 예측한 클래스 스코어의 그래디언트를 계산 -> 어떤 픽셀이 영향력 있는지 보여줌
- 입력이미지의 픽셀을 조금 바꾸었을때, 클래스의 스코어가 얼마나 바뀌는지 봄                  
( Guided backprop : 입력이미지의 각 픽셀에 대해, 중간 뉴런의 그래디언트를 계산해서 -> 어떤 픽셀이 중간 뉴런에 영향력이 있는지 보여주는것 )

## Gradient Ascent
- 네트워크의 가중치는 모두 고정
- Gradient Ascent 를 통해, 중간 뉴런 or 클래스스코어를 최대화 시키는 이미지 픽셀 만듬                   
- Regularization 을 추가 ( ⓶ 을 위해서 하는듯 .. )               
  ⓵ 이미지가 특정 뉴런의 값을 최대화시키는 방향으로 생성되길 원함                    
  ⓶ 이미지가 자연스러워야됨 ( 일반적으로 볼 수 있는 이미지이길 원하는것임 )                    


## ============================================

# Generative Models

## pixelRNN / CNN 
- likelihood 를 최대화 하기위해 chain-rule 사용 ( 관련있는 픽셀을 찾기위한 과정이라는 것인가...? / likelihood : 사건이 일어날 가능도 )  
- pixelRNN : 왼쪽 위 픽셀부터 대각선 아래방향으로 내려가면서 픽셀들을 생성 / feed forward문 여러번해야돼서 매우느림
- pixelCNN : pixelRNN과 유사하게 왼쪽 코너부터 픽셀 생성 / 모든 종속성을 고려하는것 대신 , 특정영역에서 CNN 사용 ( 픽셀을 생성할때 특정 픽셀만 고려 ) / 좀더 빠르긴함
![스크린샷 2022-03-02 오전 10 16 29](https://user-images.githubusercontent.com/98244339/156276168-3840c24a-c3fb-46ef-a659-90dcdf89fbbd.png)
![스크린샷 2022-03-02 오전 10 19 36](https://user-images.githubusercontent.com/98244339/156277206-b1258930-b28a-4702-ba52-a0195e316f4e.png)

## AutoEncoder
- 레이블되지 않은 train data로부터, 대표 features를, 학습하기위한 비지도학습 방법
- 입력을 복원하는 과정에서 특징을 잘 학습하고, 학습된 특징은 지도학습의 라벨 초기화에 이용
- AE룰 통해 차원축소의 효과가 있기도함               
[ AutoEncoder 세부 ]                    
- z가 데이터 x의 중요한 특징을 학습하기 원함
- 동일한차원의 데이터복원 
- 디코더와 인코더는 동일한구조, 대칭적, 주로 cnn 사용
- loss를 계산할때만 디코더가 쓰이고, 학습이 끝나면 디코더는 버린다. ( 학습할때는 l2 loss 사용 , 복원된 이미지의 픽셀과 입력이미지의 픽셀이 같았으면 좋겠음! ) 
- test에서는 classifier한다면, softmax 와 predicted라벨 연결해서 쓴다. 
![연습장-17](https://user-images.githubusercontent.com/98244339/156279738-3a1c6e53-d238-46c6-9ada-184c1bee85d7.jpg)



## Variational AutoEncoders
- 직접 계산이 불가능한 확률모델 정의

## GAN
- 입력으로 random noise 벡터 (z) -> Generator Network 통과하면 샘플출력 
- generator : 사실적인 이미지를 생성하여 discriminator 를 속이는것
- discriminator : 입력이미지가 실제인지 거짓인지 구분
- discriminaotr가 잘 구분한다면, generaotr는 discriminator 를 속이는 만큼 학습한다.
![스크린샷 2022-03-02 오전 11 37 55](https://user-images.githubusercontent.com/98244339/156284223-b23f5ac3-ad25-4371-bb4f-8c7cfed3f57f.png)


## ============================================


# Reinforcement Learning 
![스크린샷 2022-02-28 오후 5 12 39](https://user-images.githubusercontent.com/98244339/155947694-5a97dba4-dc5a-4e61-9658-cedf751fdb2f.png)


## Markov Decision Process ( MDP )
- Markov Property : 현재 상태만으로 전체 상태를 나타내는 성질
- 순차적으로 행동을 결정해야하는 문제

## Value function & Q-Value function
- Value function : 임의의 상태 s에 대해, 정책 兀(pi)가 주어졌을때 누적 보상의 기댓값
- Q-Value function : 임의의 상태 s에 대해, action(행동) , 정책 兀(pi) 가 주어졌을때 받을 수 있는 누적 보상의 기댓값
- Optimal Q-Value function : ( 상태 state , 행동 action )쌍으로 부터 얻을 수 있는 누적 보상의 기댓값 최대화
  [ 특정 상태에서 최상의 행동을 취할 수 있는 최적의 정책을 구할 수 있음 ]
  [ 어떤 행동을 취했을때 미래에 받을 보상의 최대치 ]

## Bellman equation 
- 어떤 ( 상태 state , 행동 action ) 이 주어지던간에, 다음 s' 으로 표현가능한 방정식 
- 현재 Value function 이 이전 state의 Value function 으로 표현가능하고, Q-Value function 도 이전 state로 표현 가능한 것

## Experience Replay
- 연속적인 샘플들로 학습하면 correlation 이 생겨서, 샘플을 랜덤하게 뽑아서 학습하는 방법
- Replay Memory 에는 ( 상태 , 행동 , 보상 , 다음상태 )로 table에 넣어놓음




