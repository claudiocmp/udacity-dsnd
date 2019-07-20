# standard imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as models
from torchvision import transforms, datasets
import json
import os
# module imports
from config import MEAN, STD, TRAIN, TEST, VALID


def load_data(directory, image_size=224):
    """Load data from directory and transform it to input to a NN.
    
    From torchvision docs: ```All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. 
    The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].```
    """
    _data_transforms = {
        'training' : transforms.Compose([
            transforms.RandomRotation(30),
            transforms.Resize(255),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'test' : transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ]),
        'validation' : transforms.Compose([
            transforms.Resize(255),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    }
    
    _image_datasets = {
        'training' :   datasets.ImageFolder(directory + TRAIN, transform=_data_transforms['training']),
        'test' :       datasets.ImageFolder(directory + TEST, transform=_data_transforms['test']),
        'validation' : datasets.ImageFolder(directory + VALID, transform=_data_transforms['validation'])
    }

    dataloaders = {
        'training' :   torch.utils.data.DataLoader(_image_datasets['training'],batch_size=32,shuffle=True),
        'test' :       torch.utils.data.DataLoader(_image_datasets['test'],batch_size=32,shuffle=True),
        'validation' : torch.utils.data.DataLoader(_image_datasets['validation'],batch_size=32,shuffle=True)
    }
    return dataloaders

def pick_architecture(arch_name):
    """Select the model architecture by name (string)
        """
    if arch_name == 'resnet18': 
        return models.resnet18(pretrained=True)
    elif arch_name == 'alexnet':
        return models.alexnet(pretrained=True)
    elif arch_name == 'squeezenet1_0':
        return models.squeezenet1_0(pretrained=True)
    elif arch_name == 'vgg16':
        return models.vgg16(pretrained=True)
    elif arch_name == 'densenet161':
        return models.densenet161(pretrained=True)
    elif arch_name == 'inception_v3':
        return models.inception_v3(pretrained=True)
    else:
        raise NameError('Architecture name {} is not defined'.format(arch_name))


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers - to be placed as classifier.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        return F.log_softmax(x, dim=1)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

def analyze_cats(cat_to_name):
    """
    Analyze categories and return their quantity
    """
    c = [int(s) for s in cat_to_name.keys()]
    # verify category indices range and if all indices ar taken (probablty 0 means no-cat)
    no_cats = abs(min(c)-max(c))
    all_taken = False not in [i in c for i in list(range(min(c),max(c)+1))]
    
    s = ''
    s += '\nCategory indices range: ({},{})'.format(min(c), max(c))
    s += '\nRange is continuous: {}'.format(all_taken)
    s += '\nTotal number of categories: {}'.format(no_cats)
    print(s)
    return no_cats



