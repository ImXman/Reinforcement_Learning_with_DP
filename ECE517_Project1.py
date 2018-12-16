# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:05:08 2018

@author: Yang Xu and Hui Liu
"""              
import sys
import numpy as np
#from os import arg

##define parameters
#n=2
#m=8
m=int(sys.argv[1])
n=int(sys.argv[2])
#c=1
c=float(sys.argv[3])
#l=-5
l=float(sys.argv[4])
dis=0.9
#p=[0.3,0.6]
p=[float(sys.argv[i]) for i in range(5,len(sys.argv)-1) ]
p = np.asarray(p)
theta=0.01
B=int(sys.argv[len(sys.argv)-1])
    
##create array to define the arriving people in each quene
people=[0 for i in range(n)]
for num in range(1,2**n):
    rate = str(bin(num).split('b')[1][::-1])
    rate = [int(i) for i in rate]
    while len(rate)<n:
        rate.append(0)
    people=np.vstack((people,rate))

aa=(p**people)*((1-p)**(1-people))
bb= np.ones((2**n,1))
for i in range(n):
    bb[:,0]*=aa[:,i]
##state_value=state_value,people=people,n=n,m=m,c=c,l=l,p=p,dis=dis,theta=theta

##create the state value matrix
def initiate_state_value():
    state_value = np.zeros(((m+1)**n,1))
    reshapes = [(m+1) for i in range(n)]
    state_value = state_value.reshape(reshapes)
    return state_value

##policy matrix has an extra dimension for action compared to state value matrix
def initiate_random_policy():
    policy = np.zeros(((m+1)**n*n,1))
    policy+= 1/n
    reshapes = [(m+1) for i in range(n)]
    reshapes.append(n)
    policy = policy.reshape(reshapes)
    return policy

##Policy evalution
def policy_evalution(state_value,policy):
    #iteration=True
    number=0
    while True:  
        number+=1
        estimation=[]
        for i in range((m+1)**n):
            
            ##make an array which can be used to retrive every state and find the current state
            state = np.zeros(((m+1)**n,1),dtype=bool)
            state[i]=True
            reshapes = [(m+1) for i in range(n)]
            state = state.reshape(reshapes)
            state=np.where(state==True)
            current_state=[]
            for j in range(n):
                current_state.append(state[j][0])
            
            ##create an empty list to store updated state value
            new_vs=[]
            
            for act in range(n):
                intmed_state = [i for i in current_state]
                intmed_state[act]-=1
                if -1 in intmed_state:
                    intmed_state[act]=0
                    intmed = intmed_state
                    for i in range(1,2**n):
                        intmed_state=np.vstack((intmed_state,intmed))
                    
                    next_state=intmed_state+people
                    vs=[]
                    for i in range(2**n):
                        if (m+1) in next_state[i,:]:
                            r=l*next_state[i,:].tolist().count((m+1))
                            new_state = next_state[i,:].tolist()
                            new_state=[m if i==(m+1) else i for i in new_state]
                            vs.append(r+dis*state_value[tuple(new_state)])
                        else:
                            r=0
                            new_state = next_state[i,:].tolist()
                            vs.append(r+dis*state_value[tuple(new_state)])
                    vs=np.asarray(vs)
                    vs=vs.reshape((2**n,1))
                    new_vs.append((bb*vs).sum())
                else:
                    intmed = intmed_state
                    for i in range(1,2**n):
                        intmed_state=np.vstack((intmed_state,intmed))
                    
                    next_state=intmed_state+people
                    vs=[]
                    for i in range(2**n):
                        if (m+1) in next_state[i,:]:
                            r=c+l*next_state[i,:].tolist().count((m+1))
                            new_state = next_state[i,:].tolist()
                            new_state=[m if i==(m+1) else i for i in new_state]
                            vs.append(r+dis*state_value[tuple(new_state)])
                        else:
                            r=c
                            new_state = next_state[i,:].tolist()
                            vs.append(r+dis*state_value[tuple(new_state)])
                    vs=np.asarray(vs)
                    vs=vs.reshape((2**n,1))
                    new_vs.append((bb*vs).sum())
            new_vs=np.asarray(new_vs).reshape((n,1))
            estimation.append(abs((policy[tuple(current_state)].reshape((n,1))*new_vs).sum()-state_value[tuple(current_state)]))
            state_value[tuple(current_state)]=(policy[tuple(current_state)].reshape((n,1))*new_vs).sum()
        if sum(estimation) < theta:
            break
    return state_value, policy, number
                    
def policy_improvement(state_value):
    
    new_policy = np.zeros(((m+1)**n*n,1))
    reshapes = [(m+1) for i in range(n)]
    reshapes.append(n)
    new_policy = new_policy.reshape(reshapes)
    
    for i in range((m+1)**n):
   
        ##make an array which can be used to retrive every state and find the current state
        state = np.zeros(((m+1)**n,1),dtype=bool)
        state[i]=True
        reshapes = [(m+1) for i in range(n)]
        state = state.reshape(reshapes)
        state=np.where(state==True)
        current_state=[]
        for j in range(n):
            current_state.append(state[j][0])
            
            ##create an empty list to store updated state value
        new_vs=[]
            
        for act in range(n):
            intmed_state = [i for i in current_state]
            intmed_state[act]-=1
            if -1 in intmed_state:
                intmed_state[act]=0
                intmed = intmed_state
                for i in range(1,2**n):
                    intmed_state=np.vstack((intmed_state,intmed))
                
                next_state=intmed_state+people
                vs=[]
                for i in range(2**n):
                    if (m+1) in next_state[i,:]:
                        r=l*next_state[i,:].tolist().count((m+1))
                        new_state = next_state[i,:].tolist()
                        new_state=[m if i==(m+1) else i for i in new_state]
                        vs.append(r+dis*state_value[tuple(new_state)])
                    else:
                        r=0
                        new_state = next_state[i,:].tolist()
                        vs.append(r+dis*state_value[tuple(new_state)])
                vs=np.asarray(vs)
                vs=vs.reshape((2**n,1))
                new_vs.append((bb*vs).sum())
            else:
                intmed = intmed_state
                for i in range(1,2**n):
                    intmed_state=np.vstack((intmed_state,intmed))
                
                next_state=intmed_state+people
                vs=[]
                for i in range(2**n):
                    if (m+1) in next_state[i,:]:
                        r=c+l*next_state[i,:].tolist().count((m+1))
                        new_state = next_state[i,:].tolist()
                        new_state=[m if i==(m+1) else i for i in new_state]
                        vs.append(r+dis*state_value[tuple(new_state)])
                    else:
                        r=c
                        new_state = next_state[i,:].tolist()
                        vs.append(r+dis*state_value[tuple(new_state)])
                vs=np.asarray(vs)
                vs=vs.reshape((2**n,1))
                new_vs.append((bb*vs).sum())
        #new_vs=np.asarray(new_vs).reshape((n,1))
        #take_act=(old_policy[tuple(current_state)].reshape((n,1))*new_vs).reshape((n)).tolist()
        index = []
        num=0
        for i in range(n):
            if new_vs[i] == max(new_vs):
                num+=1
                index.append(i)
            
        for i in index:
            new_policy[tuple(current_state)][i]+=1/num
    return new_policy

def policy_stable(new_policy,old_policy):
    stable=False

    num_unstable=0
    for i in range((m+1)**n):
   
        ##make an array which can be used to retrive every state and find the current state
        state = np.zeros(((m+1)**n,1),dtype=bool)
        state[i]=True
        reshapes = [(m+1) for i in range(n)]
        state = state.reshape(reshapes)
        state=np.where(state==True)
        current_state=[]
        for j in range(n):
            current_state.append(state[j][0])
            
        if new_policy[tuple(current_state)].tolist() != old_policy[tuple(current_state)].tolist():
            num_unstable+=1
    if num_unstable <= (m+1)**n*theta:
        stable=True
    return stable

def policy_iteration():
    epoch=0
    state_value=initiate_state_value()
    old_policy=initiate_random_policy()
    state_value,old_policy, number=policy_evalution(state_value,old_policy)
    epoch+=number
    new_policy =policy_improvement(state_value)
    stable = policy_stable(new_policy,old_policy)
    #stable=False
    while not stable:
        #state_value=initiate_state_value()
        state_value, old_policy,number=policy_evalution(state_value,new_policy)
        epoch+=number
        new_policy=policy_improvement(state_value)
        stable = policy_stable(new_policy,old_policy)
        
    return new_policy, old_policy, state_value, epoch

def initiate_state_value2():
    temp = list()
    for i in range(n):
        temp.append(m + 1)
    return np.zeros(temp)

def initiate_random_policy2():
    policy = np.zeros(((m+1)**n*n,1))
    reshapes = [(m+1) for i in range(n)]
    reshapes.append(n)
    policy = policy.reshape(reshapes)
    return policy

def value_iteration():
    value = initiate_state_value2()
    policy = initiate_random_policy2()
    new_value = value.copy()
    epoch = 0
    while True:
        for i in range((m + 1) ** n):
            state = np.zeros(((m + 1) ** n, 1), dtype=bool)
            state[i] = True
            reshapes = [(m + 1) for i in range(n)]
            state = state.reshape(reshapes)
            state = np.where(state == True)
            c_state = []
            for j in range(n):
                c_state.append(state[j][0])

            action_value = []
            for a in range(n):
                n_state = c_state[:]
                r0 = 0.0
                if c_state[a]:
                    n_state[a] -= 1
                    r0 += 1.0

                prob_index = 0
                action_value_temp = 0.0
                for line in people:
                    new_state = n_state[:]
                    r = r0
                    q_index=0
                    for q in line:
                        if q != 0:
                            new_state[q_index] += 1
                            if new_state[q_index] > m:
                                new_state[q_index] = m
                                r -= 5.0
                        q_index += 1
                    action_value_temp += bb[prob_index, 0] * (r + dis * value[tuple(new_state)])
                    prob_index += 1
                action_value.append(action_value_temp)
            new_value[tuple(c_state)] = max(action_value)
            policy[tuple(c_state)] = np.where(action_value==np.max(action_value),1,0)
            #policy[tuple(c_state)] = max(np.where(action_value == np.max(action_value)))
        deta = abs(new_value - value).sum()
        value = new_value.copy()
        epoch = epoch + 1
        print ("epoch:"+ str(epoch)) 

        if deta < theta:
            break
    return value, policy

      
def main():
    if B==0:
        policy,oldpolicy,value,number_iter=policy_iteration()
        print(number_iter)
        #np.savetxt("state_value_via_policy_iteration.txt",value,delimiter="\t")
    else:
        value,policy=value_iteration()       
        
if __name__ == '__main__':
    main()
    
