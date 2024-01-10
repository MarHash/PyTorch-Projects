import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm  # For nice progress bar!


# create a fully connected nn
class NN(nn.Module):
    ''' the class inherits the nn module from pytorch '''
    
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()

        # we will make a simple 1 hidden layer network that has 50 nodes (input:50:num_classes)
        self.fc1 = nn.Linear(input_size, 50) #fully connected layer - connects each input feature to 50 output features, using a linear transformation
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x): # x is our input data
        x = F.relu(self.fc1(x)) # relu will be applied to the output of fc1 - relu adds non-linearity - notice the operator overload with fc1
        x = self.fc2(x) # output of relu is passed to the final layer
        return x



# now let's use an available device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# our hyperparams
input_size = 784 # mnist images are 28X28
num_classes = 10 # 0 - 9 digits
lr = 0.001
batch_size = 64
epochs = 20


# Load our data
train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True) # transform the data to tensors
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True) # loader for training the data as it trains

test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True) # transform the data to tensors
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True) # loader for test data


# now let's initialise the network for training
model = NN(input_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss() # this will be the loss criterion for backprobagation
optimizer = optim.Adam(model.parameters(), lr=lr)

# train
for epoch in range(epochs):
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)): # unpack the data batch from the loader
        
        # map data and targets to cuda if available
        data = data.to(device)
        targets = targets.to(device)

        # print(data.shape) # should return [batch_size, number of input channels (1 for black and white in case of MNIST), img width, img height]

        data = data.reshape(data.shape[0], -1) # we want the correct shape

        #forward
        scores = model(data) # our forward method get called when data is passed to the instane
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad() # set all gradients to zero before backward
        loss.backward()

        # gradient descent or adam step - updates the weights depending on the backward call
        optimizer.step()


# check accuracy on training and test data

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('checking on train date')
    else:
        print('checking on test date')

    num_correct = 0
    num_samples = 0

    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train() # revert from eval mode


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)