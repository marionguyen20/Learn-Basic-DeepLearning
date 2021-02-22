import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

#Sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Change figure name
fig = plt.gcf()
fig.canvas.set_window_title('Logistic Regression')

#Load data from dataset
data = pd.read_csv('Logistic Regression/dataset.csv').values
N, d = data.shape
x = data[:,0:d-1].reshape(-1,d-1) #Get columns 0 and 1
y = data[:,2].reshape(-1,1) #Get column 2

#Scatter data
plt.xlabel('Salary (millions)')
plt.ylabel('Experiences (years)')
plt.scatter(x[:10,0], x[:10,1], c='red', edgecolors='none', s = 30, label = 'Accepted' ) #s: size of dot; edgecolor: around dot
plt.scatter(x[10:,0], x[10:,1], c='blue', edgecolors='none', s = 30, label = 'Rejected')
plt.legend(loc=1) #Location of label

#Create x and w matrix
x = np.hstack((np.ones((N,1)),x))
w = np.array([0.,0.1,0.1]).reshape(-1,1)

numOfIteration = 1000
learning_rate = 0.01
cost = np.zeros((numOfIteration,1))

for i in range (1, numOfIteration):
    #Compute the predict vale
    yPredict = sigmoid(np.dot(x,w))
    cost[i] = - np.sum(np.multiply(y, np.log(yPredict)) + np.multiply(1-y, np.log(1 - yPredict)))

    #Gradient descent
    w = w - learning_rate * np.dot(x.T, yPredict - y)
    print (cost[i])

#Draw the median line
t = 0.8
#x from 4 to 10 => draw y based on x
plt.plot((4,10), (-(w[0] + w[1]*4 + np.log(1/t - 1))/w[2], - (w[0] + w[1]*10 + np.log(1/t -1 ))/w[2]),color = 'green')
plt.show()







