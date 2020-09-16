import matplotlib.pyplot as plt
import numpy as np


X, y = [], []

with open('./dataset001.csv', 'r') as f:
    next(f)
    for line in f:
        line = line.strip()
        if line:
            values = line.split(',')
        else:
            continue
        X.append([float(i) for i in values[:2]])
        y.append(int(values[-1]))
        
print(len(X), len(y))


#############################################
import random


random.seed(123)

idx = list(range(len(X)))
random.shuffle(idx)

X_train = [X[i] for i in idx[:80]]
y_train = [y[i] for i in idx[:80]]
X_test = [X[i] for i in idx[80:]]
y_test = [y[i] for i in idx[80:]]

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

###########################################
plt.scatter([i[0] for idx, i in enumerate(X_train) if y_train[idx] == 0], 
            [i[1] for idx, i in enumerate(X_train) if y_train[idx] == 0],
            label='class 0', marker='o')

plt.scatter([i[0] for idx, i in enumerate(X_train) if y_train[idx] == 1], 
            [i[1] for idx, i in enumerate(X_train) if y_train[idx] == 1],
            label='class 1', marker='s')

plt.title('Training set')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.xlim([0.0, 7])
plt.ylim([-0.8, 0.8])
plt.legend()
plt.show()


W_out = [] ; b_out = []

#############################################
#####定义激活函数

def Linear(x):

    return x

def sigmoid(x):

    ex = np.exp(x)

    return ex / (ex + 1)

def tanh(x):

    return np.tanh(x)

def relu(x):

    return np.where(x > 0, x, 0)
##############################################
class Perceptron():
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = np.zeros((num_features,1),dtype=np.float) # <your code>
        self.bias = np.zeros(1,dtype=np.float)# <your code>


    def forward(self, x):
        linear = np.dot(x,self.weights) + self.bias# <your code>
        # prediction = np.where(linear > 0.,1,0)# <your code>
        # prediction = sigmoid(linear)# <your code>
        # prediction = tanh(linear)# <your code>
        prediction = relu(linear)# <your code>
        return prediction
        
    def backward(self, x, y):
        predictions = self.forward(x)
        errors = y - predictions
        return errors  
        # <your code> to compute the prediction error
        
    def train(self, x, y, epochs):

        for e in range(epochs):
            
            for i in range(len(y)):
                errors = self.backward(x[i].reshape(1, self.num_features), y[i]).reshape(-1)
                W_out.append(self.weights.tolist())
                b_out.append(self.bias.tolist())                
                self.weights += (errors*x[i]).reshape(self.num_features,1)
                self.bias += errors 



                # <your code> to update the weights and bias
                
    def evaluate(self, x, y):
        predictions = self.forward(x).reshape(-1)
        accuracy = np.sum(predictions == y)/y.shape[0]
        # <your code> to compute the prediction accuracy       
        return accuracy
##########################################
ppn = Perceptron(num_features=2)

ppn.train(X_train,y_train,epochs=1)

print(' Weights: %s\n' % ppn.weights)
print(' Bias: %s\n' % ppn.bias)


# <your code>

#############################################

# <your code>
train_acc = ppn.evaluate(X_train,y_train)
print('Train set accuracy: %.2f%%' % (train_acc*100))

# <your code>
test_acc = ppn.evaluate(X_test,y_test)
print('Test set accuracy: %.2f%%' % (test_acc*100))



#############################################
##########################
### 2D Decision Boundary
##########################

w = ppn.weights
b = ppn.bias

x_min = 0
y_min = ( (-(w[0] * x_min) - b[0]) 
          / w[1] )

x_max = 6
y_max = ( (-(w[0] * x_max) - b[0]) 
          / w[1] )


fig, ax = plt.subplots(1, 2, sharex=True, figsize=(7, 5))

ax[0].plot([x_min, x_max], [y_min, y_max])
ax[1].plot([x_min, x_max], [y_min, y_max])

ax[0].scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], label='class 0', marker='o')
ax[0].scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], label='class 1', marker='s')

ax[1].scatter(X_test[y_test==0, 0], X_test[y_test==0, 1], label='class 0', marker='o')
ax[1].scatter(X_test[y_test==1, 0], X_test[y_test==1, 1], label='class 1', marker='s')

ax[0].legend(loc='upper left')
ax[1].legend(loc='upper left')

plt.show()
# <your code>


####画W和b迭代图
# W_out,b_out

# nn = 81

# x = list(range(1,nn))

# W = np.array(W_out)

# y1 = W[:,0,:]
# y2 = W[:,1,:]
# y3 = np.array(b_out)

# fig, ax = plt.subplots(1, 2, sharex=True, figsize=(11, 5))

# ax[0].plot(x,y1,color='r')
# ax[0].plot(x,y2,color='b')
# ax[1].plot(x,y3)

# ax[0].set_title('Iteration of weights')
# ax[1].set_title('Iteration of bias')

# ax[0].set_xlabel('step')
# ax[0].set_ylabel('weights')

# ax[1].set_xlabel('step')
# ax[1].set_ylabel('bias')

# ax[0].legend(['feature1','feature2'],loc='upper right')

# plt.show()



# import torch 

# a = torch.arange(10).view(2,5)

# w = torch.tensor([4,3,2,1,0])

# w = w.view(-1,1)

# b = a.matmul(w)

# print(a)
# print(b)




# import torch 

# x = torch.arange(50,dtype=torch.float).view(10,5)

# fc_layer = torch.nn.Linear(in_features=5,out_features=3)


# print(fc_layer.weight)

# a = fc_layer(x)

# print(a)