from distutils.command.config import config
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import numpy as np
import PIL.Image
import sklearn.metrics
from typing import Callable, Optional

from RainforestDataset import RainforestDataset, ChannelSelect, get_classes_list
from YourNetwork import SingleNetwork, TwoNetworks

def train_epoch(model, trainloader, criterion, device, optimizer):
    model.train()
 
    losses = []
    for batch_idx, data in enumerate(trainloader):
        if (batch_idx %100==0) and (batch_idx>=100):
          print('at batchidx',batch_idx)
        
        inputs = data['image'].to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()

        output = model(inputs)
        loss = criterion(output, labels)
  
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (batch_idx %100==0) and (batch_idx>=100):
            print('current mean of losses ',np.mean(losses))
      
    return np.mean(losses)

def train_epoch_twonetwork(model, trainloader, criterion, device, optimizer):
    model.train()
 
    losses = []
    for batch_idx, data in enumerate(trainloader):
        if (batch_idx %100==0) and (batch_idx>=100):
          print('at batchidx',batch_idx)
        
        inputs = data['image']#.to(device)
        inputs1 = inputs[:,[0,1,2],:,:]
        inputs2 = inputs[:,[3],:,:]
        inputs2 = inputs2.repeat(1,3,1,1)

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels = data['label'].to(device)

        optimizer.zero_grad()

        output = model(inputs1, inputs2)
        loss = criterion(output, labels)
  
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (batch_idx %100==0) and (batch_idx>=100):
            print('current mean of losses ',np.mean(losses))
      
    return np.mean(losses)

def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):
    model.eval()

    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs= np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)
      
          inputs = data['image'].to(device)        
          outputs = model(inputs)
          labels = data['label']

          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

          cpuout = outputs.to('cpu')

          concat_pred = np.vstack((concat_pred, cpuout))
          concat_labels = np.vstack((concat_labels, labels))
          fnames = fnames + data['filename']

    np.seterr(invalid='ignore')
    for c in range(numcl):
      avgprecs[c] = sklearn.metrics.average_precision_score(concat_labels[:,c], concat_pred[:,c])

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames

def evaluate_meanavgprecision_twonetwork(model, dataloader, criterion, device, numcl):
    model.eval()

    concat_pred = np.empty((0, numcl)) #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels = np.empty((0, numcl)) #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs= np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):
          if (batch_idx%100==0) and (batch_idx>=100):
            print('at val batchindex: ', batch_idx)
      
          inputs = data['image']#.to(device)
          inputs1 = inputs[:,[0,1,2],:,:]
          inputs2 = inputs[:,[3],:,:]
          inputs2 = inputs2.repeat(1,3,1,1)

          inputs1 = inputs1.to(device)
          inputs2 = inputs2.to(device)
          labels = data['label']

          outputs = model(inputs1, inputs2)
          loss = criterion(outputs, labels.to(device))
          losses.append(loss.item())

          cpuout = outputs.to('cpu')

          concat_pred = np.vstack((concat_pred, cpuout))
          concat_labels = np.vstack((concat_labels, labels))
          fnames = fnames + data['filename']

    np.seterr(invalid='ignore')
    for c in range(numcl):
      avgprecs[c] = sklearn.metrics.average_precision_score(concat_labels[:,c], concat_pred[:,c])

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames

def traineval2_model_nocv(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl, filename):
  best_measure = 0
  best_epoch = -1

  trainlosses=[]
  testlosses=[]
  testperfs=[]
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    avgloss=train_epoch(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)
    
    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss, concat_labels, concat_pred, fnames = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)
    
    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    
    avgperfmeasure = np.mean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure:
      bestweights = model.state_dict()
      best_measure = avgperfmeasure
      torch.save(model, filename + '.pt')
      # Save values from best epoch
      vals = {'epoch':np.asarray(epoch), 'AP':np.asarray(perfmeasure), 'mAP':np.asarray(avgperfmeasure), 'labels':np.asarray(concat_labels), 'preds':np.asarray(concat_pred), 'fnames':np.asarray(fnames)}
  
  # save vals to file
  vals['trainlosses'] = np.asarray(trainlosses); vals['testlosses'] = np.asarray(testlosses); vals['testperfs'] = np.asarray(testperfs)
  outfile = filename + '.npz'
  np.savez(outfile, **vals)

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs

def traineval_twonetwork(dataloader_train, dataloader_test ,  model ,  criterion, optimizer, scheduler, num_epochs, device, numcl, filename):
  best_measure = 0
  best_epoch = -1

  trainlosses=[]
  testlosses=[]
  testperfs=[]
  
  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    avgloss=train_epoch_twonetwork(model,  dataloader_train,  criterion,  device , optimizer )
    trainlosses.append(avgloss)
    
    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss, concat_labels, concat_pred, fnames = evaluate_meanavgprecision_twonetwork(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)
    testperfs.append(perfmeasure)
    
    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    
    avgperfmeasure = np.nanmean(perfmeasure)
    print('at epoch: ', epoch,' avgperfmeasure ', avgperfmeasure)

    if avgperfmeasure > best_measure: 
      bestweights = model.state_dict()
      best_measure = avgperfmeasure
      torch.save(model, filename + '.pt')
      # Save values from best epoch
      vals = {'epoch':np.asarray(epoch), 'AP':np.asarray(perfmeasure), 'mAP':np.asarray(avgperfmeasure), 'labels':np.asarray(concat_labels), 'preds':np.asarray(concat_pred), 'fnames':np.asarray(fnames)}
  
  # save vals to file
  vals['trainlosses'] = np.asarray(trainlosses); vals['testlosses'] = np.asarray(testlosses); vals['testperfs'] = np.asarray(testperfs)
  outfile = filename + '.npz'
  np.savez(outfile, **vals)

  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs

def runstuff():
  config = dict()
  config['use_gpu'] = True # change this to True for training on the cluster
  config['lr'] = 0.005
  config['batchsize_train'] = 16
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 12
  config['scheduler_stepsize'] = 5
  config['scheduler_factor'] = 0.3

  # This is a dataset property.
  config['numcl'] = 17

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(),
          transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]) # RGB
          #transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # RGBa
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          ChannelSelect(),
          transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]) # RGB
          #transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # RGBa
      ]),
  }

  #root_dir = str(pathlib.Path(__file__).parent.resolve())
  root_dir = '/itf-fi-ml/shared/IN5400/2022_mandatory1/'
  # Datasets
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir=root_dir, trvaltest=0, transform=data_transforms['train'])
  image_datasets['val'] = RainforestDataset(root_dir=root_dir, trvaltest=1, transform=data_transforms['val'])

  # Dataloaders
  dataloaders = {}
  dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batchsize_train'], shuffle=True, num_workers=1)
  dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=config['batchsize_val'], shuffle=False, num_workers=1)

  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  # Model
  pretrained_net = models.resnet18(pretrained=True)
  model = SingleNetwork(pretrained_net)
  model = model.to(device)

  lossfct = nn.BCELoss()

  lr = config['lr']
  someoptimizer = optim.SGD(model.net.fc.parameters(), lr=lr, momentum=0.9)

  # Decay LR by a factor of config['scheduler_factor'] every config['scheduler_factor'] epochs
  somelr_scheduler = torch.optim.lr_scheduler.StepLR(someoptimizer, step_size=config['scheduler_stepsize'], gamma=config['scheduler_factor'])

  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'], filename='Task1')

  return

def runstuff2():
  config = dict()
  config['use_gpu'] = True #change this to True for training on the cluster
  config['lr'] = 0.005
  config['batchsize_train'] = 16
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 12
  config['scheduler_stepsize'] = 5
  config['scheduler_factor'] = 0.3

  # This is a dataset property.
  config['numcl'] = 17

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          #ChannelSelect(),
          #transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]) # RGB
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # RGBa
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          #ChannelSelect(),
          #transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]) # RGB
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # RGBa
      ]),
  }

  #root_dir = str(pathlib.Path(__file__).parent.resolve())
  root_dir = '/itf-fi-ml/shared/IN5400/2022_mandatory1/'
  # Datasets
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir=root_dir, trvaltest=0, transform=data_transforms['train'])
  image_datasets['val'] = RainforestDataset(root_dir=root_dir, trvaltest=1, transform=data_transforms['val'])

  # Dataloaders
  dataloaders = {}
  dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batchsize_train'], shuffle=True, num_workers=1)
  dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=config['batchsize_val'], shuffle=False, num_workers=1)

  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  # Models
  pretrained_net1 = models.resnet18(pretrained=True)
  pretrained_net2 = models.resnet18(pretrained=True)
  model = TwoNetworks(pretrained_net1, pretrained_net2)
  model = model.to(device)

  lossfct = nn.BCELoss()

  lr = config['lr']
  someoptimizer = optim.SGD(model.linear.parameters(), lr=lr, momentum=0.9)

  # Decay LR by a factor of config['scheduler_factor'] every config['scheduler_factor'] epochs
  somelr_scheduler = torch.optim.lr_scheduler.StepLR(someoptimizer, step_size=config['scheduler_stepsize'], gamma=config['scheduler_factor'])

  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval_twonetwork(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'], filename='Task3')

  return

def runstuff3():
  """
  Same as runstuf() but for task4 with; weight_init = "kaiminghe"
  """
  config = dict()
  config['use_gpu'] = True #change this to True for training on the cluster
  config['lr'] = 0.005
  config['batchsize_train'] = 16
  config['batchsize_val'] = 64
  config['maxnumepochs'] = 12
  config['scheduler_stepsize'] = 5
  config['scheduler_factor'] = 0.3

  # This is a dataset property.
  config['numcl'] = 17

  # Data augmentations.
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          #ChannelSelect(),
          #transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]) # RGB
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # RGBa
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          #ChannelSelect(),
          #transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]) # RGB
          transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # RGBa
      ]),
  }


  #root_dir = str(pathlib.Path(__file__).parent.resolve())
  root_dir = '/itf-fi-ml/shared/IN5400/2022_mandatory1/'
  # Datasets
  image_datasets={}
  image_datasets['train'] = RainforestDataset(root_dir=root_dir, trvaltest=0, transform=data_transforms['train'])
  image_datasets['val'] = RainforestDataset(root_dir=root_dir, trvaltest=1, transform=data_transforms['val'])

  # Dataloaders
  dataloaders = {}
  dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=config['batchsize_train'], shuffle=True, num_workers=1)
  dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=config['batchsize_val'], shuffle=False, num_workers=1)

  # Device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  # Model
  pretrained_net = models.resnet18(pretrained=True)
  model = SingleNetwork(pretrained_net, weight_init="kaiminghe")
  model = model.to(device)

  lossfct = nn.BCELoss()

  lr = config['lr']
  someoptimizer = optim.SGD(model.net.fc.parameters(), lr=lr, momentum=0.9)

  # Decay LR by a factor of config['scheduler_factor'] every config['scheduler_factor'] epochs
  somelr_scheduler = torch.optim.lr_scheduler.StepLR(someoptimizer, step_size=config['scheduler_stepsize'], gamma=config['scheduler_factor'])

  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs= config['maxnumepochs'], device = device , numcl = config['numcl'], filename='Task4')
  
  return

if __name__=='__main__':
  torch.manual_seed(0)
  runstuff()
  runstuff2()
  runstuff3()