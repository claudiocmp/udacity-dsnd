# standard imports
import torch
import torch.nn as nn
from torchvision import models as models
import numpy as np
from collections import OrderedDict
from PIL import Image
# modules
from util import load_data, cat_to_name, analyze_cats, pick_architecture, Network
from config import MEAN, STD

class Trainer():
    
    def __init__(self, feat_directory, arch_name, h_units, learn_rate, device):
        self.device = device
        self.architecture = arch_name
        self.model = pick_architecture(arch_name)
        self.dataloaders = load_data(feat_directory) # assumed input_size=224 all the times
        cats = analyze_cats(cat_to_name)
        self.input_size, self.output_size, self.dpout = self.model_setup(h_units, cats)
        self.criterion = torch.nn.NLLLoss()
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=learn_rate)
        self.epochs = None
        self.test_accuracy = None
    
    
    def model_setup(self, h_units, no_cats):
        """modify model classifier accordingly to model architecture, desired hidden units and categories to be generated
        """
        print(self.model)
        assert(isinstance(h_units,int)), "'{}' is not an integer".format(h_units)
        for p in self.model.parameters():
            p.requires_grad = False
        
        # get feature output layer size
        output = self.model.classifier.state_dict()
        keys = output.keys()
        try:
            x,y = output['0.weight'].shape
        except:
            x,y = output['weight'].shape
        output_dim = max(x,y)
        # remember: non-category hsould be included, thus output layer is no_cat++
        self.model.classifier = Network(output_dim, no_cats+1, [h_units], 0.35)
        return output_dim, no_cats+1, 0.35
    
    
    def train(self, epochs):
        # train
        print(f'\nTRAINING PHASE\nDevice used: {self.device}\n')
        self.model.to(self.device)
        
        steps = 0
        print_every = 20
        training_losses, test_losses, accuracies = [],[],[]
        
        for e in range(epochs):
            training_loss = 0.0
            for img,lbl in self.dataloaders['training']:
                # initial operations
                self.optimizer.zero_grad()
                img,lbl = img.to(self.device),lbl.to(self.device)
                steps +=1
                # forward, loss, step
                log_ps = self.model.forward(img)
                loss = self.criterion(log_ps, lbl)
                loss.backward()
                self.optimizer.step()
                self.epochs = e+1
                training_loss += loss.item()
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for t_img, t_lbl in self.dataloaders['validation']:
                            t_img,t_lbl = img.to(self.device),lbl.to(self.device)
                            log_ps = self.model.forward(t_img)
                            # accuracy
                            ps = torch.exp(log_ps)
                            top_p, top_cls = ps.topk(1,dim=1)
                            equals = top_cls == lbl.view(*top_cls.shape)
                            if self.device == 'cpu':
                                accuracy += torch.mean(equals.type(torch.FloatTensor))
                            else:
                                accuracy += torch.mean(equals.type(torch.cuda.FloatTensor))
                            loss = self.criterion(log_ps, lbl)
                            test_loss += loss.item()
                    print(f'epoch {e+1}/{epochs}',
                          f'Training Loss: {training_loss/print_every:.3f}',
                          f'Validation Loss: {test_loss/len(self.dataloaders["validation"]):.3f}',
                          f'Accuracy: {accuracy*100/len(self.dataloaders["validation"]):.1f}%'
                         )
                    # output data
                    training_losses.append(training_loss/print_every)
                    test_losses.append(test_loss/len(self.dataloaders['validation']))
                    accuracies.append(accuracy/len(self.dataloaders['validation']))
                    # to not cumulate between steps of an epoch
                    training_loss = 0.0
                    #set back to training mode
                    self.model.train()
    
    
    def test_model(self):
        """Do validation on the test set"""
        print(f'\nTESTING PHASE\n\nDevice used: {self.device}')
        self.model.to(self.device)
        self.model.eval()

        test_accuracy = 0.0

        with torch.no_grad():
            for i,(img,lbl) in enumerate(self.dataloaders['test']):
                img,lbl = img.to(self.device),lbl.to(self.device)

                log_ps = self.model.forward(img)
                ps = torch.exp(log_ps)#
                _, top_class = ps.topk(1,dim=1)
                equals = top_class == lbl.view(*top_class.shape)
                if self.device == 'cpu':
                    test_accuracy += torch.mean(equals.type(torch.FloatTensor))
                else:
                    test_accuracy += torch.mean(equals.type(torch.cuda.FloatTensor))

        self.test_accuracy = test_accuracy.item()/len(self.dataloaders['test'])
        print(f'Accuracy: {self.test_accuracy*100:.1f}%\n')
    
    
    def save_checkpoint(self, directory):
        """Save the checkpoint"""
        print(f'Saving model {self.architecture} with accuracy to directory: {directory}...')
        try:
            print(f'Model accuracy is {self.test_accuracy:.1f}%')
        except:
            print(f'Model has not been trained')
        self.model.to('cpu')
        checkpoint = {'architecture':self.architecture,
                      'epochs': self.epochs,
                      'input_size': self.input_size,
                      'output_size': self.output_size,
                      'hidden_layers': [hl.out_features for hl in self.model.classifier.hidden_layers],
                      'state_dict': self.model.state_dict(),
                      'optim_dict': self.optimizer.state_dict(),
                      'class_labels':self.dataloaders['training'].dataset.classes
                     }
        torch.save(checkpoint, directory)
        print('Model saved successfully')

class Predictor():
    
    def __init__(self, fp):
        """Load a model from file and makes prediction"""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loaded_ckpt = torch.load(f=fp)
        self.model = pick_architecture(self.loaded_ckpt['architecture'])
        self.model_setup()
        self.class_labels = self.loaded_ckpt['class_labels']
    
    def model_setup(self):
        self.model.classifier = Network(
            self.loaded_ckpt['input_size'], 
            self.loaded_ckpt['output_size'], 
            self.loaded_ckpt['hidden_layers'], 
            0.35
        )
        self.model.load_state_dict(self.loaded_ckpt['state_dict'])
        print('Model {} loaded with classifier {}'.format(self.loaded_ckpt['architecture'] , self.model.classifier))
    
    
    def process_image(self, image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        size_1 = 256
        size_2 = 224
        # Process a PIL image for use in a PyTorch model
        pimg = Image.open(image)
        pimg.thumbnail(size=(size_1,size_1))
        # (0,0) is upper left
        # left,upper,right,lower
        bbox = pimg.getbbox()
        new_bbox = np.array(bbox)*.5 + np.array([bbox[2]-size_2,bbox[3]-size_2,size_2,size_2])*.5
        cropped = pimg.crop(box=tuple(new_bbox)) #crop centred
        a_img = np.array(cropped)
        a_img = a_img/255.
        a_img = (a_img[:,:,:3]-MEAN)/STD #exclude alpha channel
        a_img = a_img.transpose(2,0,1)
        return torch.tensor(a_img, dtype=torch.float32) #model a doubletensor
    
    
    def predict(self, fp, topk=5, use_gpu=True):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        img = torch.unsqueeze(self.process_image(fp), 0) #input must be an array of tensors
        #not necessary as it's done in the img processing function - but probably good practice to do it here as we could check model weight/bias type
        if use_gpu:
            img.to('cuda')
            img = img.type(torch.cuda.FloatTensor)
            self.model.to('cuda')
        else:
            img.to('cpu')
            img = img.type(torch.FloatTensor)
            self.model.to('cpu')
        log_ps = self.model.forward(img.detach())
        ps = torch.exp(log_ps)
        top_ps, top_cat_idx = ps.topk(k=topk, dim=1)
        labels_index = [int(self.class_labels[int(i)]) for i in top_cat_idx.cpu().numpy().flatten()]
        return top_ps.cpu().detach().numpy().flatten(), np.array(labels_index)
    
    
    def inference(self, fp, top_k,use_gpu=True):
        """Returns a prediction for an input image"""
        ps, label_index = self.predict(fp,top_k,use_gpu)
        return ps, label_index
    
    
        
"""
Tests
"""
# t = Trainer(r'./flowers', 'vgg16', 4096, 1e-4)
# t.train(1)
# t.test_model()
# t.save_checkpoint(r'./checkpoint.pht')
# p = Predictor(r'./checkpoint.pht')