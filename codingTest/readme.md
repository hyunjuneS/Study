# 프로그래머스 코딩테스트
## Sort
- sorted(dic_sum.items(), key=lambda x:x[1], reverse=True)
- sorted(dic['pop'], key=lambda x:x[1], reverse=True)

## zip
progresses = [93, 30, 55]<br/>
speeds = [1, 30, 5]                   
for p , s in zip(progresses, speeds):                              
    print(p, "    " , s)                           
=> 	93      1                        
    30      30                          
    55      5                           
    
    
## pop 
list = [1,2,3,4]
lst.pop(0) 

