#!/usr/bin/env python
# coding: utf-8

# Tensor Basics

# In[1]:


import torch


# In[8]:


x=torch.tensor([2.5, 0.1])
print(x)


# In[9]:


a=torch.rand(2,2)
b=torch.rand(2,2)
print(a)
print(b)


# In[11]:


b.add_(a)


# In[16]:


c=torch.rand(4,4)
print(c)


# In[17]:


d=c.view(-1,2)
print(d)


# In[18]:


import numpy as np


# In[22]:


a = np.ones(6)
print(a)


# In[23]:


b=torch.from_numpy(a)
print(type(b))


# Autograd

# In[25]:


import torch


# In[27]:


x=torch.randn(3, requires_grad=True)
print(x)


# In[28]:


y=x+2
print(y)


# In[31]:


y=y.mean()
y.backward()
print(x.grad)


# In[32]:


z=x*x


# In[33]:


v=torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)
print(x.grad)


# Ways to instruct tensor not to learn from the gradient history

# In[34]:


#x.requires_grad_(False)
#x.detach()
#with torch.nograd():


# Way to reset grad to zero

# In[35]:


weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_op=(weights*3).sum()
    print(model_op)
    model_op.backward()
    print(weights.grad)
    
    weights.grad.zero_()


# Backpropagation

# In[63]:


import torch


# In[64]:


x=torch.tensor(1.0)
y=torch.tensor(2.0)

w=torch.tensor(1.0, requires_grad=True)


# In[65]:


#forward pass
y_hat= w*x
loss=(y_hat-y)**2
print(loss)


# In[66]:


#backward pass
loss.backward()
print(w.grad)


# gradientdescent conventional method

# In[85]:


import numpy as np


# In[86]:


X=np.array([1,2,3,4])
Y=np.array([2,4,6,8])


# In[87]:


w=0.0


# In[88]:


#output
def forward(x):
    return w*x


# In[89]:


#loss=Mean squared Error
def loss(y, y_pred):
    return((y_pred-y)**2).mean()


# In[90]:


def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred-y).mean()


# In[91]:


print(f"Prection of model before training: Forward(5) = {forward(5):.4f}")


# In[92]:


#training
lr=0.01
iteration=20


# In[94]:


for epoch in range(iteration):
    y_prediction=forward(X)
    l=loss(Y,y_prediction)
    dw=gradient(X,Y, y_prediction)
    w-= lr*dw
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f"Prection of model after training: Forward(5) = {forward(5):.4f}")


# Gradient descent using pytorch

# In[95]:


import torch


# In[97]:


X=torch.tensor([1,2,3,4], dtype=torch.float32)
Y=torch.tensor([2,4,6,8], dtype=torch.float32)


# In[99]:


w=torch.tensor(0.0, dtype=torch.float32, requires_grad=True)


# In[100]:


#output
def forward(x):
    return w*x


# In[101]:


#loss=Mean squared Error
def loss(y, y_pred):
    return((y_pred-y)**2).mean()


# In[102]:


print(f"Prection of model before training: Forward(5) = {forward(5):.4f}")


# In[103]:


#training
lr=0.01
iteration=20


# In[106]:


for epoch in range(iteration):
    y_prediction=forward(X)
    l=loss(Y,y_prediction)
    l.backward()
    with torch.no_grad():
        w-= lr*w.grad
    w.grad.zero_()
    print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
print(f"Prection of model after training: Forward(5) = {forward(5):.4f}")


# Training pipeline

# In[110]:


import torch
import torch.nn as nn


# In[114]:


X=torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
Y=torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)


# In[115]:


n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')


# In[116]:


X_test=torch.tensor([5], dtype=torch.float32)


# In[117]:


input_size=n_features
output_size=n_features


# In[138]:


class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.lin=nn.Linear(input_dim, output_dim)
    
    def forward(self,x):
        return self.lin(x)


# In[139]:


model=LinearRegression(input_size, output_size)


# In[140]:


print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')


# In[141]:


learning_rate = 0.01
n_iters = 100


# In[142]:


loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[143]:


for epoch in range(n_iters):
    
    y_predicted = model(X)

    
    l = loss(Y, y_predicted)

    
    l.backward()


    optimizer.step()


    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)

print(f'Prediction after training: f(5) = {model(X_test).item():.3f}')    


# In[145]:


from sklearn import datasets


# In[146]:


X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)


# In[147]:


print(y_numpy)


# In[148]:


print(X_numpy)


# Logistic Regression

# In[187]:


import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[188]:


breast_cancer=datasets.load_breast_cancer()
X,y = breast_cancer.data, breast_cancer.target


# In[189]:


print(X)


# In[190]:


print(y)


# In[191]:


n_samples, n_features=X.shape


# In[192]:


print(n_samples)
print(n_features)


# In[193]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=1234)


# In[194]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[195]:


X_train=torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))


# In[196]:


y_train=y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# In[197]:


class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model,self).__init__()
        self.linear = nn.Linear(n_input_features,1)
    
    def forward(self, x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred


# In[198]:


model=Model(n_features)


# In[199]:


num_epochs = 100
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[204]:


for epoch in range(num_epochs):
    y_pred=model(X_train)
    loss=criterion(y_pred, y_train)
    loss.backward()
    optimizer.zero_grad()
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')


# Data Transformer and Data loader

# In[225]:


import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math


# In[226]:


class WineDataset(Dataset):
    def __init__(self, transform=None):
        dataset_loading=np.loadtxt('/home/gourav/Downloads/9408623-b237fa5848349a14a14e5d4107dc7897c21951f5/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples=dataset_loading.shape[0]
        
        self.x_data=dataset_loading[:,1:]
        self.y_data=dataset_loading[:,[0]]
        
        self.transform=transform
        
    def __getitem__(self, index):
        sample= self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.n_samples


# In[227]:


class ToTensor:
    def __call__(self, sample):
        inputs, results = sample
        return torch.from_numpy(inputs), torch.from_numpy(results)


# In[228]:


print('\nWith Tensor Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)


# In[229]:


train_loader=DataLoader(dataset=dataset,
                       batch_size=4,
                       shuffle=True,
                       num_workers=2)


# In[230]:


dataiterator=iter(train_loader)
data=dataiterator.next()
features, labels=data
print(features, labels)


# In[231]:


num_epochs=2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)


# In[ ]:




