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
from collections import OrderedDict
import argparse

#load and transform data
#return train loader, valid loader, and test loader
def load_transform():
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
    
    return trainloader, validloader, testloader, train_data, valid_data, test_data



#label mapping used to convert encoded categories to actual names of flowers.
def load_cat_file(file):
    
    #with open('cat_to_name.json', 'r') as f:
    with open(file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


# Build network
## step 1: load a pretrained model. allow user to specify architecture. Two architecture are made available: vgg 16 and alexnet. 
def model_select(arch): 
    
    available_arch = ['vgg16', 'alexnet']
    
    
    if arch in available_arch:
        string = 'models.' + arch + '(pretrained=True)'
    else:
       return 'architecture not available.'
    
    model = eval(string)

    ### keep the prtrained weights unchanged
    for param in model.parameters():
        param.reuires_grad = False

    return model
    
    
### define a new untrained network as classifier. Allow user to specify hidden units
### Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout -> Linear -> logsoftmax.
def define_classifier(model, hidden_units, output_size, drop_rate):
    #argument:
    #hidden_units: list of integers, ex [25088, 4096, 512]
    #each linear layer follows by a RELU and Dropout layers.
    layer = []
    
    for i in range(len(hidden_units)):
        
               
        if i != (len(hidden_units) - 1):
            
            layer.append(('fc' + str(i), nn.Linear(hidden_units[i],hidden_units[i+1] )))
            layer.append(('ReLU' + str(i), nn.ReLU("inplace")))
            layer.append(('Drop_rate' + str(i), nn.Dropout(drop_rate)))
                       
        else:                         
            layer.append(('output', nn.Linear(hidden_units[i], output_size)))


    #output layer
    layer.append(('softmax',nn.LogSoftmax(dim = 1)))
    
    clf = nn.Sequential(OrderedDict(layer))
    model.classifier = clf
    return 

## step 2: define device, optimizer and criterion
                         
def define_device_crit_Optim(model, dev, learn_rate):
    device = torch.device(dev)
    criterion = nn.NLLLoss()
    optimizier = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    model.to(device)
    return device, criterion, optimizier
                         

## step3: train the model and display train loss, validation loss and validation accuracy.
### define epochs and initialize counter. 
                         
def train_model(model, epochs, trainloader, validloader, dev, learn_rate, train_data):
    
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = 5
                         
    device, criterion, optimizer = define_device_crit_Optim(model, dev, learn_rate)


    with active_session():
        for epoch in range(epochs):
            for inputs, labels in trainloader:
                steps +=1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every ==0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in validloader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model.forward(inputs)
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
                    model.train()
        
    # Save the model to checkpoint. 
    checkpoint = {'model_arch': model,
                  'n_epochs': epochs,
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_index': train_data.class_to_idx,
                  'state_dict': model.state_dict(),
                  'hidden_layer': model.classifier,
                  'learn_rate': learn_rate}


    torch.save(checkpoint,'checkpoint.pth')
                
                         
def main(cat_file, arch, hidden_units, output_size, drop_rate, epochs, dev, learn_rate):
    trainloader, validloader, testloader, train_data, valid_data, test_data = load_transform()
    cat_to_name = load_cat_file(cat_file) #'cat_to_name.json'
    model = model_select(arch)
    define_classifier(model, hidden_units, output_size, drop_rate)
    
    
    train_model(model, epochs, trainloader, validloader, dev, learn_rate, train_data)
    
    return 

                   
                  
                         
if __name__ == '__main__':
                         
    parser = argparse.ArgumentParser()
    
    parser.add_argument('cat_file', default = 'cat_to_name.json')
    parser.add_argument('arch', default = 'vgg16')
    parser.add_argument('hidden_units', default = '25088,4096,512')
    parser.add_argument('output_size', type = int, default = 102)                             
    parser.add_argument('drop_rate', type = float, default = 0.5)
    parser.add_argument('epochs', type = int, default = 3)
    parser.add_argument('dev', default = 'cuda')
    parser.add_argument('learn_rate', type = float, default = 0.001)
    
                         
    input_args = parser.parse_args()
    hidden_units = []
    
    for i in range(len(input_args.hidden_units.split(','))):
        hidden_units.append(eval(input_args.hidden_units.split(',')[i]))
   
    
    
    main(input_args.cat_file, input_args.arch, hidden_units, input_args.output_size, input_args.drop_rate, input_args.epochs, input_args.dev, input_args.learn_rate)
    #python train.py 'cat_to_name.json' 'vgg16' '25088, 4096, 512' 102 0.5 3 'cuda' 0.001