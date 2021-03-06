# Predict.py
# this script load the trained network and predict the class for input image.

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
import argparse
import train as tn


# define location of data
data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    

# define functions

## 1. define the load checkpoint function. this function will be used to rebuild the model. 
def load_checkpoint(filepath, dev):
    
    checkpoint = torch.load(filepath)
    model = checkpoint['model_arch']
    learn_rate = checkpoint['learn_rate']
    model.class_to_idx = checkpoint['class_to_index']
    model.load_state_dict(checkpoint['state_dict'])
    
    #keep the pretrained weights unchanged
    for param in model.parameters():
        param.reuires_grad = False

    device , criterion, optimizer = tn.define_device_crit_Optim(model, dev, learn_rate)
    optimizer.state_dict = checkpoint['optimizer_state']
    
    #pass model parameters and other tensors to GPU memory if available, otherwise to cpu.
    model.to(device);
    
    return model

## 2. Define the process_image function. 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    im = Image.open(image)
    
    #resize image so shortest side is 256
    width, height = im.size
    shortest_dim = min(width, height)
    ratio = shortest_dim / 256.
    
    new_width  = int(width/ratio)
    new_height = int(height/ratio)
    
    
    new_dim = (new_width, new_height)
    
    new_im = im.resize(new_dim, Image.ANTIALIAS)
    
    #crop out the 224 * 224 center portion of the image
    crop_dim = 224
    
    left = int((new_width - crop_dim)/2)
    upper = int((new_height - crop_dim)/2)
    right = left + crop_dim
    lower = upper + crop_dim
    
    new_im = new_im.crop((left, upper, right, lower)) 
    
    #assign to numpy array    
    
    new_im_array = np.array(new_im)   #it has dimension (256,340,3)
    
    #scale by mean and std
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    new_im_array = new_im_array/255    #scale to (0,1)
       
    new_im_array = (new_im_array - mean) /std
    
    #reorder dimension
    new_im_array = new_im_array.transpose((2,0,1))

    return new_im_array

## 3. define the image show function. 
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

## define the predict function. 
def predict(image_path, model, topk, dev):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device(dev)
    model.eval()
    inputs = torch.tensor(np.expand_dims(process_image(image_path), axis = 0), dtype= torch.float)
    
    with torch.no_grad():
        inputs = inputs.to(device)
        logps = model.forward(inputs)  
        ps = torch.exp(logps)
        top_p, idx = ps.topk(topk, dim =1)
    
        #move back to cpu memory
        top_p = list(np.squeeze(top_p.cpu().numpy()))
        idx = np.squeeze(idx.cpu().numpy())

        
        #index to class
        top_cls = []
        for i in idx:
            for k, v in model.class_to_idx.items():
                if v == i:
                    top_cls.append(k)   
    
    #return inputs
    return top_p, top_cls  

# Main program
def main(image_path, checkpoint, cat_file, dev, top_k):
    model = load_checkpoint(checkpoint, dev)
    top_p, top_cls = predict(image_path, model, top_k, dev)
    
    cat_name = []
    cat_to_name = tn.load_cat_file(cat_file)
    
    for i in top_cls:
        cat_name.append(cat_to_name[i])
    
    return top_p, cat_name

# Argument parser

#test input: python predict.py 'flowers/test/43/image_02412.jpg' 'checkpoint.pth' 'cat_to_name.json' 'cuda' 5
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image_path', default = 'flowers/test/43/image_02412.jpg')
    parser.add_argument('checkpoint', default = 'checkpoint.pth')
    parser.add_argument('cat_file', default = 'cat_to_name.json')
    parser.add_argument('dev', default = 'cuda')
    parser.add_argument('top_k', type = int, default = 5)
    
    input_args = parser.parse_args()
    
    top_p, cat_name = main(input_args.image_path, input_args.checkpoint, input_args.cat_file, input_args.dev, input_args.top_k)
    
    print(f'\nimage: {input_args.image_path} ')
    print(f'\nFlower Category                            Probability\n')
    for i in range(len(top_p)):
        print(f'{cat_name[i]:<25}{top_p[i]:>25,.2f}')
    






