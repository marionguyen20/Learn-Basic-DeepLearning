import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Read data
data = pd.read_csv('Linear Regression\data_square.csv').values
N = data.shape[0] #Number of rows
x = data[:,0].reshape(-1,1) #(-1) means unknow -> reshape with unknow rows (computer will calculate) and 1 column
y = data[:,1].reshape(-1,1)
x1 = (50-min(x))/(max(x)-min(x))
x_normalize = (x-min(x))/(max(x) - min(x))
y_normalize = (y-min(y))/(max(y) - min(y))

plt.scatter (x_normalize,y_normalize)
plt.xlabel('Square')
plt.ylabel('Price')
x = np.hstack((np.ones((N,1)),np.square(x_normalize),x_normalize))
w = np.array([0.,1.,0.],dtype=np.float64).reshape(-1,1)

numOfIteration = 5000
learning_rate = 0.01
cost = np.zeros((numOfIteration,1))

for i in range (1, numOfIteration):
    r = np.dot(x,w) - y_normalize
    cost[i] = 0.5*np.sum(r*r)
    w[0] -= learning_rate*np.sum(r)
    w[1] -= learning_rate*np.sum(np.multiply(r,x[:,1].reshape(-1,1)))
    w[2] -= learning_rate*np.sum(np.multiply(r,x[:,2].reshape(-1,1)))
    print(cost[i])

yPredict = np.dot(x,w)
plt.plot(x[:,2],yPredict,'r')

y1 = w[0] + w[1]*x1*x1 + w[2]*x1
print("Price of 50m^2 ",y1)
plt.scatter(x1,y1,color = 'red')
plt.show()








