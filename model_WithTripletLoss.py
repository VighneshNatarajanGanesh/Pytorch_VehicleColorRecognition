from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import  models, transforms,datasets
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import time
import math
import numpy as np
import os,cv2
from torch.utils.data import Dataset
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from PIL import Image

#imports added by Vighnesh:
from dataset import ImageFolder
from torch.utils.data import DataLoader
import argparse
import datetime
from utils import *

# 定义一个类，需要创建模型的时候，就实例化一个对象

class VehicleColorRecognitionModel(nn.Module):
    def __init__(self, args, Load_VIS_URL=None):
        
        # arguments
        self.dataset = args.dataset # name of the dataset folder
        self.batch_size = args.batch_size
        self.triplet_batch_size = self.batch_size
        self.test_batch_size = args.test_batch_size
        self.NoClasses = args.NoClasses
        self.ipChannels  = args.ipChannels
        self.result_dir = args.result_dir # save the model
        self.resume = args.resume # resume model or restart training
        self.resume_folder = args.resume_folder
        self.useTripletLoss = args.useTripletLoss
        self.tripletLossRatio = args.tripletLossRatio
        
        # dataloader for cross entropy loss
        train_transform = transforms.Compose([ transforms.ToTensor() ])
        test_transform = transforms.Compose([ transforms.ToTensor() ])        
        self.trainFolder = ImageFolder(os.path.join('dataset', self.dataset, 'train'), train_transform)
        self.testFolder = ImageFolder(os.path.join('dataset', self.dataset, 'test'), test_transform)
        self.train_loader = DataLoader(self.trainFolder, batch_size=self.batch_size, shuffle=True,pin_memory=True)
        self.test_loader = DataLoader(self.testFolder, batch_size=self.test_batch_size, shuffle=True,pin_memory=True)
        
        # dataloader for triplet loss
        triplet_transform = transforms.Compose([ transforms.ToTensor() ])
        Triplet_loader = TripletLoader(os.path.join('dataset', self.dataset), transform=triplet_transform) #defined in utils.py
        self.triplet_loader = DataLoader(Triplet_loader, batch_size=self.triplet_batch_size, shuffle=True, num_workers=4)

        
        
        super(VehicleColorRecognitionModel,self).__init__()
        
        # ===============================  top ================================
        # first top convolution layer   
        self.top_conv1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(self.ipChannels, 48, kernel_size=(11,11), stride=(4,4)),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
       
        
        # first top convolution layer    after split
        self.top_top_conv2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.top_bot_conv2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        
        #  need a concat
        
        # after concat  
        self.top_conv3 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(128, 192, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        # fourth top convolution layer
        # split feature map by half
        self.top_top_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        self.top_bot_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        
        # fifth top convolution layer
        self.top_top_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.top_bot_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

#        # ===============================  bottom ================================
    
           
#         # first bottom convolution layer   
        self.bottom_conv1 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(self.ipChannels, 48, kernel_size=(11,11), stride=(4,4)),
            nn.ReLU(),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
       
        
        # first top convolution layer    after split
        self.bottom_top_conv2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.bottom_bot_conv2 = nn.Sequential(

            # 1-1 conv layer
            nn.Conv2d(24, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        
        #  need a concat
        
        # after concat  
        self.bottom_conv3 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(128, 192, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        # fourth top convolution layer
        # split feature map by half
        self.bottom_top_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        self.bottom_bot_conv4 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 96, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU()
        )
        
        
        # fifth top convolution layer
        self.bottom_top_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.bottom_bot_conv5 = nn.Sequential(
            # 1-1 conv layer
            nn.Conv2d(96, 64, kernel_size=(3,3), stride=(1,1),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        # Fully-connected layer
        self.classifier1 = nn.Sequential(
            nn.Linear(6*6*64*4, 4096)
        )              
        
        self.classifier2 = nn.Sequential(
            nn.ReLU(), 
            nn.Dropout(0.7),
            nn.Linear(4096, 4096),
            nn.ReLU(), 
            nn.Dropout(0.6),
            nn.Linear(4096, self.NoClasses)
        )
        
    def forward(self,x, intermediateOut = False):
        '''
        set intermediate = true if you want output of intermediate layer, (used in triplet loss for example)
        '''
        
        
        x_top = self.top_conv1(x)
        #print(x_top.shape)
                
        x_top_conv = torch.split(x_top, 24, 1)
        
        x_top_top_conv2 = self.top_top_conv2(x_top_conv[0])
        x_top_bot_conv2 = self.top_bot_conv2(x_top_conv[1])
        
        x_top_cat1 = torch.cat([x_top_top_conv2,x_top_bot_conv2],1)
        
        x_top_conv3 = self.top_conv3(x_top_cat1)
        
        x_top_conv3 = torch.split(x_top_conv3, 96, 1)
        
        x_top_top_conv4 = self.top_top_conv4(x_top_conv3[0])
        x_top_bot_conv4 = self.top_bot_conv4(x_top_conv3[1])
        
        x_top_top_conv5 = self.top_top_conv5(x_top_top_conv4)
        x_top_bot_conv5 = self.top_bot_conv5(x_top_bot_conv4)
        
        x_bottom = self.bottom_conv1(x)
        
        x_bottom_conv = torch.split(x_bottom, 24, 1)
        
        x_bottom_top_conv2 = self.bottom_top_conv2(x_bottom_conv[0])
        x_bottom_bot_conv2 = self.bottom_bot_conv2(x_bottom_conv[1])
        
        x_bottom_cat1 = torch.cat([x_bottom_top_conv2,x_bottom_bot_conv2],1)
        
        x_bottom_conv3 = self.bottom_conv3(x_bottom_cat1)
        
        x_bottom_conv3 = torch.split(x_bottom_conv3, 96, 1)
        
        x_bottom_top_conv4 = self.bottom_top_conv4(x_bottom_conv3[0])
        x_bottom_bot_conv4 = self.bottom_bot_conv4(x_bottom_conv3[1])
        
        x_bottom_top_conv5 = self.bottom_top_conv5(x_bottom_top_conv4)
        x_bottom_bot_conv5 = self.bottom_bot_conv5(x_bottom_bot_conv4)
        
        x_cat = torch.cat([x_top_top_conv5,x_top_bot_conv5,x_bottom_top_conv5,x_bottom_bot_conv5],1)
        
        flatten = x_cat.view(x_cat.size(0), -1)
        
        intermediateOutput = self.classifier1(flatten)
        
        if intermediateOut:
            return intermediateOutput
        
        output = self.classifier2(intermediateOutput)
        
        # the cross entropy loss automatically applies softmax, but during runtime add the softmax layer tho.
#         output = F.softmax(output, dim=0)
        
        
        return output
    
def train(model, optimizer, criterion, epochs=1000, print_freq=10, save_freq=1000, lrDecay=None):
    
    crit_train_losses = []
    crit_test_losses = []
    triplet_losses = []
    
    
    if(model.resume):
        # continue training:
        resume_folder = model.resume_folder
        print("dir:", resume_folder)
        resume_file = sorted( os.listdir(resume_folder) )[-1]
        subFolderName = "_".join( str(datetime.datetime.now()).split(" ") )
        SAVE_DIR = os.path.join( resume_folder, subFolderName)
        os.mkdir( SAVE_DIR )
        model.load_state_dict(torch.load( os.path.join(resume_folder, resume_file) ))
        
        print("[INFO] Continuing training from ", os.path.join(resume_folder, resume_file))
        
    else:
        # make the folder to save the model:
        if(not os.path.exists("results")):
            os.mkdir("results")
        if(not os.path.exists( os.path.join("results", model.dataset) )):
            os.mkdir( os.path.join("results", model.dataset) )
        if(not os.path.exists( os.path.join("results", model.dataset, "model") )):
            os.mkdir( os.path.join("results", model.dataset, "model") )   
        # save the model in folder with current (training start) date and time as name
        folderName = "_".join( str(datetime.datetime.now()).split(" ") ) # replace space with _
        SAVE_DIR = os.path.join("results", model.dataset, "model", folderName)
        os.mkdir( SAVE_DIR )
        
    
    if model.useTripletLoss:
        tripletLoss = TripletLoss()
    
    for epoch in range(epochs+1):
    
        model.train()
        tr_loss = 0

        ''' Criteron loss '''
        # get the training data
        try:
            X_data, file_paths = model.train_iter.next()
        except:
            train_iter = iter(model.train_loader)
            X_data, file_paths = train_iter.next()
        x_train = X_data
        y_train = np.array([int(path.split('/')[-1][:2]) for path in file_paths ]) #get the class from the file name
        y_train = Variable(torch.from_numpy(y_train)) # make y_train as a torch var
        # cross entropy loss does not need one hot encodings
#         y_train = torch.nn.functional.one_hot(y_train) # make one hot encodings

        # get the test data
        try:
            X_data, file_paths = model.test_iter.next()
        except:
            test_iter = iter(model.test_loader)
            X_data, file_paths = test_iter.next()

        x_test = X_data
        y_test = np.array([int(path.split('/')[-1][:2]) for path in file_paths ]) #get the class from the file name
        y_test = Variable(torch.from_numpy(y_test)) # make y_test as a torch var
        # cross entropy loss does not need one hot encodings
#         y_test = torch.nn.functional.one_hot(y_test) # make one hot encodings

        # transfer vals to gpu
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_test = x_test.cuda()
            y_test = y_test.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and test set
        output_train = model(x_train)
        output_test = model(x_test)

        # computing the training and test loss
        crit_loss_train = criterion(output_train, y_train)
        crit_loss_test = criterion(output_test, y_test)
        # before appending the losses detach it to free the reference to the computation graph
        crit_train_losses.append(crit_loss_train.cpu().detach().numpy())
        crit_test_losses.append(crit_loss_test.cpu().detach().numpy())

        
        ''' Triplet Loss'''
        if model.useTripletLoss == False:
            loss_train = crit_loss_train
        else:
            # use triplet loss:
            
            try:
                anchor_img, positive_img, negative_img, anchor_label = model.triplet_iter.next()
            except:
                triplet_iter = iter(model.triplet_loader)
                anchor_img, positive_img, negative_img, anchor_labels = triplet_iter.next()
                

            anchor_img = anchor_img.to(device)
            positive_img = positive_img.to(device)
            negative_img = negative_img.to(device)
            
            # calcukate the intermediate values
            anchor_out = model(anchor_img, intermediateOut=True)
            positive_out = model(positive_img, intermediateOut=True)
            negative_out = model(negative_img, intermediateOut=True)
        
            triplet_loss = tripletLoss(anchor_out, positive_out, negative_out)
            
            triplet_losses.append(triplet_loss.cpu().detach().numpy())
            
            # calculate the net loss:
            loss_train = crit_loss_train + model.tripletLossRatio*triplet_loss        

            
        ''' calculate backprop for the net loss'''
        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()
        
        # decay the lr if necessary
        if lrDecay is not None:
            lrDecay.step()
        
        # printing
        if model.useTripletLoss == False:
            if(epoch%print_freq==0):
                testPredictions = np.argmax(output_test.cpu().detach().numpy(), axis=1)
                testTruth = y_test.cpu().detach().numpy()
                NumCorrectPredictions = np.sum(testPredictions==testTruth)

                print("Epoch: %6d | training loss =  %3.4f | test loss = %3.4f | correct test predictions = %d/%d"
                      %(epoch, crit_loss_train, crit_loss_test, NumCorrectPredictions, len(testTruth) ))
        else:
            if(epoch%print_freq==0):
                testPredictions = np.argmax(output_test.cpu().detach().numpy(), axis=1)
                testTruth = y_test.cpu().detach().numpy()
                NumCorrectPredictions = np.sum(testPredictions==testTruth)

                print("Epoch: %6d | training loss =  %3.4f | test loss = %3.4f | train Triplet Loss = %3.4f | correct test predictions = %d/%d"
                      %(epoch, crit_loss_train, crit_loss_test, triplet_loss, NumCorrectPredictions, len(testTruth) ))
    
    
        # saving the model
        if(epoch%save_freq==0):
            params = model.state_dict()
            torch.save(params, os.path.join(SAVE_DIR + '/_params_%07d.pt' % epoch))
    
    
    
    
# mode = VehicleColorRecognitionModel()
# inputs = torch.rand(64,3,224,224)
# mode(inputs)
# x = torch.rand(1,48,27,27)
# print(x.shape)
# z = torch.split(x,24,1)
# z[0].shape



''' Arguments '''
parser = argparse.ArgumentParser(description="VehicleColorRecognitionModel")
parser.add_argument('--dataset', type=str, default='default')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch size')
parser.add_argument('--test_batch_size', type=int, default=32, help='The size of batch size for test data')
parser.add_argument('--NoClasses', type=int, default=10, help='The number of output classes')
parser.add_argument('--ipChannels', type=int, default=4, help='The number of channels in input image')
parser.add_argument('--result_dir', type=str, default="results", help='directory to srore the results (like the trained model)')
parser.add_argument('--resume', type=str2bool, default=False)
parser.add_argument('--resume_folder', type=str, default="", help='directory to load the results (like the trained model)')
parser.add_argument('--learningRate', type=float, default=0.001, help='Learning Rate, default=0.001')
parser.add_argument('--epochs', type=int, default=25000, help='The number of epochs')
parser.add_argument('--lrDecay', type=str2bool, default=False, help='set to true if you want Exponential Learning Rate Decay')
parser.add_argument('--lrDecayRate', type=float, default=0.999)
parser.add_argument('--useTripletLoss', type=str2bool, default=False, help='set to true if you want to activate triplet loss')
parser.add_argument('--tripletLossRatio', type=float, default=1)
args = parser.parse_args()

''' trying model '''
model = VehicleColorRecognitionModel(args)
# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = args.learningRate)
if(args.lrDecay==True):
    lrDecay = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.lrDecayRate)

# Define loss function
criterion = torch.nn.CrossEntropyLoss()
# see if gpu is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.get_device_name()

    
print(model)
print("------------------------")
print("[NOTE] Normalization was removed from the hdf5 loader, uncomment it if it is needed")
print("------------------------")
if(args.lrDecay==True):
    train(model, optimizer, criterion, args.epochs, 25, 500, lrDecay)
else:
    train(model, optimizer, criterion, args.epochs, 25, 500)