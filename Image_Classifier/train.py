#train.py 
#This script trains a new network on dataset and save model on chekcpoint

#import libaries
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn
from torch import optim
import time
from workspace_utils import active_session
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json

#data directory
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

#transfrom on train, validation and test set. 
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229,0.224,0.225])] )


valid_test_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229,0.224,0.225])] )

#load image datasets and the transforms to define dataloaders.
train_data = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_test_transforms)
test_data = datasets.ImageFolder(test_dir, transform = valid_test_transforms)

# Using the image datasets and the trainforms to define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64)

#label mapping used to convert encoded categories to actual names of flowers.
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    

# Build network
## step 1: load a pretrained model. Using vgg16
vgg = models.vgg16(pretrained= True)

### keep the prtrained weights unchanged
for param in vgg.parameters():
    param.reuires_grad = False

### define a new untrained network as classifier. 
### Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> logsoftmax.
clf = nn.Sequential(nn.Linear(25088, 4096),
                    nn.ReLU("inplace"),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 512), 
                    nn.ReLU("inplace"),
                    nn.Dropout(0.5),
                    nn.Linear(512, 102), 
                    nn.LogSoftmax(dim = 1)) 

### replace the pretrained classifier with the newly defined classifier.
vgg.classifier = clf


## step 2: define device, optimizer and criterion
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
optimizer = optim.Adam(vgg.classifier.parameters(), lr = 0.001)

### pass model parameters and other tensors to GPU memory if available, otherwise to cpu.
vgg.to(device);

## step3: train the model and display train loss, validation loss and validation accuracy.
### define epochs and initialize counter. 
epochs = 3
steps = 0
running_loss = 0
print_every = 5


with active_session():
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps +=1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = vgg.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every ==0:
                valid_loss = 0
                accuracy = 0
                vgg.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = vgg.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim = 1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch +1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.."
                      f"validation accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                vgg.train()
                

# Save the model to checkpoint. 
checkpoint = {'n_epochs': epochs,
              'optimizer_state': optimizer.state_dict,
              'class_to_index': train_data.class_to_idx,
              'state_dict': vgg.state_dict()}


torch.save(checkpoint,'checkpoint.pth')
                





