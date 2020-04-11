#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD


# In[2]:


def PlotParameters(model): 
    W = model.state_dict()['linear.weight'].data
    w_min = W.min().item()
    w_max = W.max().item()
    fig, axes = plt.subplots(2, 5)
    fig.subplots_adjust(hspace=0.01, wspace=0.1)
    for i, ax in enumerate(axes.flat):
        if i < 10:
            
            # Set the label for the sub-plot.
            ax.set_xlabel("class: {0}".format(i))

            # Plot the image.
            ax.imshow(W[i, :].view(28, 28), vmin=w_min, vmax=w_max, cmap='seismic')

            ax.set_xticks([])
            ax.set_yticks([])

        # Ensure the plot is shown correctly with multiple plots
        # in a single Notebook cell.
    plt.show()


# In[3]:


train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
# transform is specify the all MNIST images loaded to pytorch tensor
validation_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())


# In[4]:


train_dataset[2]
# tensors is image, we mentioned that in before all MNIST images converted to a pytorch tensor. 4 is the actual class of image
# all dataset looks like this


# In[5]:


# type of output is FloatTensor
print(train_dataset[2][0]) # we taking the tensor side of tuple

# type of output of code cell below is LongTensor
print(train_dataset[2][1]) # there is actual output of image


# In[6]:


"""softmax classification from scratch"""
class SoftMax(nn.Module):
    def __init__(self, in_size, out_size):
        super(SoftMax,self).__init__() # super() allows us to avoid using the base class name explicitly
        self.linear = nn.Linear(in_size, out_size)
        
    def forward(self, x):
        out = self.linear(x)
        return out


# In[7]:


input_dimension = 28*28 # our imaged 28x28, that's why our input dimension will be 28*28 to the SoftMax classifier
output_dimension = 10 # output classes

# create model based on SoftMax class
model = SoftMax(input_dimension, output_dimension)


# In[8]:


""" Since we have 10 output classes we'll have 10 weights and 10 bias parameters for SoftMax.
    Each parameter will be 784 dimensions long(28*28)"""
print("W: ", list(model.parameters())[0].size())
print("b: ", list(model.parameters())[1].size())
PlotParameters(model) # randomly initalized numbers. weights and biases


# In[9]:


loss_criterion = nn.CrossEntropyLoss()
model_optimizer = SGD(model.parameters(), lr=0.01) # lr is learning rate


# In[10]:


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100)
validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000)


# In[11]:


"""zero_grad:
            Sets gradients of all model parameters to zero.
            we need to set the gradients to zero before starting to do backpropragation 
            because PyTorch accumulates the gradients on subsequent backward passes."""
n_epochs = 50
loss_list = []
accuracy_list = []
N_test = len(validation_dataset)

def train_model(n_epochs):
    for epoch in range(n_epochs):
        for x, y in train_loader:
            model_optimizer.zero_grad()
            z = model(x.view(-1, 28 * 28))
            loss = loss_criterion(z, y)
            loss.backward()
            model_optimizer.step()
            
        correct = 0
        # perform a prediction on the validationdata  
        for x_test, y_test in validation_loader:
            z = model(x_test.view(-1, 28 * 28))
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        loss_list.append(loss.data)
        accuracy_list.append(accuracy)

train_model(n_epochs)


# In[12]:


# Plot the loss and accuracy

fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.plot(loss_list,color=color)
ax1.set_xlabel('epoch',color=color)
ax1.set_ylabel('total loss',color=color)
ax1.tick_params(axis='y', color=color)
    
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('accuracy', color=color)  
ax2.plot( accuracy_list, color=color)
ax2.tick_params(axis='y', color=color)
fig.tight_layout()


# In[13]:


# Plot the parameters

PlotParameters(model)


# In[15]:


Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat != y:
        plt.show()
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break   


# In[17]:


Softmax_fn=nn.Softmax(dim=-1)
count = 0
for x, y in validation_dataset:
    z = model(x.reshape(-1, 28 * 28))
    _, yhat = torch.max(z, 1)
    if yhat == y:
        plt.show()
        print("yhat:", yhat)
        print("probability of class ", torch.max(Softmax_fn(z)).item())
        count += 1
    if count >= 5:
        break  

