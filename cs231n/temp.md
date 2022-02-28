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
