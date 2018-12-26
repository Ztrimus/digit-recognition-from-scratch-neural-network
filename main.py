# Build a Fully Connected 2-Layer Neural Network to Classify Digits

from nnet import model

# import torch and torchvision libraries
# We will use torchvision's transforms and datasets
import math
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Defining torchvision transforms for preprocessing
transform_param = transforms.Compose([transforms.ToTensor()])

# Using torchvision datasets to load MNIST
trainset = datasets.MNIST('./data', train=True, download=True, transform=transform_param) 
validset = datasets.MNIST('./data', train=True, download=True, transform=transform_param) 
testset = datasets.MNIST('./data', train=False, download=True, transform=transform_param)

# Spliting training set of MNIST in 90% of Train set and 10% Validation set
valid_size = 0.1
num_train = len(trainset)
num_test = len(testset)
indices = torch.LongTensor(range(num_train))
split = int(math.floor(valid_size * num_train))

# Shuffle the Indices
torch.manual_seed(42)
indices = indices[torch.randperm(num_train)]

# Split the train and valid set
train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

# Using training batch size = 4 in train data loader.
batch_size = 4

# Using torch.utils.data.DataLoader to create loaders for train and test 
trainset_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
validset_loader = torch.utils.data.DataLoader(validset, batch_size=split, sampler=valid_sampler)
testset_loader = torch.utils.data.DataLoader(testset, batch_size=num_test)

# Don't change these settings
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Don't change these settings
# Layer size
N_in = 28 * 28 # Input size	
N_h1 = 256 # Hidden Layer 1 size
N_h2 = 256 # Hidden Layer 2 size
N_out = 10 # Output size
# Learning rate
lr = 0.01


# init model
net = model.FullyConnected(N_in, N_h1, N_h2, N_out, device=device)

# Define number of epochs
N_epoch = 7 # Or keep it as is


# Training and Validation Loop
# >>> for n epochs
for i in range(N_epoch):
    print("Train Epoch: ", i+1,end=' ')
# ## >>> for all mini batches
    for batch in trainset_loader:
        images = batch[0]
        images = images.reshape(images.size()[0], N_in)
        labels = batch[1]
### >>> net.train(...)
        net.train(images, labels, lr=lr, debug=False)
## at the end of each training epoch
    for batch in validset_loader:
        images = batch[0]
        images = images.reshape(images.size()[0], N_in)
        labels = batch[1]
## >>> net.eval(...)
        net.eval(images, labels, debug= True)

# End of Training
# make predictions on randomly selected test examples
# >>> net.predict(...)
for batch in testset_loader:
    images = batch[0]
    images = images.reshape(images.size()[0], N_in)
    labels = batch[1]
    score, idx = net.predict(images)
    prediction = sum((idx == labels).long()).item()
    print("=========== Traning Phase ===========")
    print("Correct Prediction : ", prediction , " / ", len(labels))
    print("Test Accuracy : ", prediction / len(labels))
