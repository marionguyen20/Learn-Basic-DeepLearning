#Find min f(x) = x^2 + 2x +5
#By calculator Min f(x) = 4 when x = -1

#Use Gradient Descent
import numpy as np 

learning_rate = 0.001
numberOfIteration = 5000

result = np.zeros ((numberOfIteration, 1))
x = 0;

for i in range (1, numberOfIteration):
    result[i] = x*x + 2*x + 5
    fPrime = 2*x + 2
    print(x,result[i])
    x -= learning_rate*fPrime
    
