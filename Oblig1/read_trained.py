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
import matplotlib.pyplot as plt
import warnings

from RainforestDataset import RainforestDataset, ChannelSelect, get_classes_list
from YourNetwork import SingleNetwork, TwoNetworks
from train_pytorch_in5400_studentversion import evaluate_meanavgprecision, evaluate_meanavgprecision_twonetwork

def plot_10_pics(names, title, root_dir, filename):
    fig = plt.figure(figsize=(8, 8))
    for i in range(10):
        img = PIL.Image.open(root_dir + 'train-jpg/' + names[i].removesuffix('.tif') + '.jpg')
        fig.add_subplot(5, 2, i+1)
        plt.imshow(img)
    plt.savefig('img/' + filename + '.png')
    plt.close()

def tailacc(values):
    # Values from best epoch
    AP = values['AP']
    labels = values['labels']
    preds = values['preds']
    fnames = values['fnames']

    steps = 10
    tail_accs = np.zeros((len(preds[0,:]), steps))
    threshold_values = np.zeros((len(preds[0,:]), steps))
    for c in range(len(preds[0,:])): # for classes
        chanel_preds = preds[:,c].flatten()
        chanel_labels = labels[:,c].flatten()

        sorting_idx = np.argsort(chanel_preds)
        sorted_preds = chanel_preds[sorting_idx]
        sorted_labels = chanel_labels[sorting_idx]
        sorted_fnames = fnames[sorting_idx]
        
        t_start = 0.5
        t_end = sorted_preds[-1] # Highest prediction score for class c
        if t_end < t_start:
            t_array = np.linspace(0.5, 1, num=steps)
        else:
            t_array = np.linspace(t_start, t_end, num=steps)

        threshold_values[c,:] = t_array
        for i, t in enumerate(t_array):
            thresholded_pred = np.where(sorted_preds >= t, 1, 0)
            tail_idx = np.nonzero(thresholded_pred) # indexes wherer thresholded_pred==1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                acc = sklearn.metrics.accuracy_score(sorted_labels[tail_idx], thresholded_pred[tail_idx])
            if acc != acc: # acc is nan
                acc = 0

            tail_accs[c, i] = acc

    # Plot tailacc
    classes, num_classes = get_classes_list()
    plt.figure()
    ax = plt.subplot(111)
    for i in range(len(tail_accs[:,0])):
        line, = ax.plot(threshold_values[i,:], tail_accs[i,:], label=classes[i], alpha=0.6)
    # mean taiacc
    line, = ax.plot(np.mean(threshold_values, axis=0), np.linspace(0.5, 1, num=steps), linewidth=2, label='Mean over all classes')
    plt.xlabel('Threshold values')
    plt.ylabel('Tail accuracy')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('img/tailacc.png')
    plt.close()
    
    # Best and worst pictures of high AP class
    best_AP_index = np.where(AP==np.nanmax(AP))
    chanel_preds = preds[:,best_AP_index].flatten()
    chanel_labels = labels[:,best_AP_index].flatten()
    sorting_idx = np.argsort(chanel_preds)
    sorted_preds = chanel_preds[sorting_idx]
    sorted_labels = chanel_labels[sorting_idx]
    sorted_fnames = fnames[sorting_idx]

    worst_pics = sorted_fnames[0:10]
    best_pics = sorted_fnames[-11:-1]

    filename = '10_worst'
    plot_10_pics(worst_pics, '10 worst pictures', root_dir='/itf-fi-ml/shared/IN5400/2022_mandatory1/', filename=filename)
    filename = '10_best'
    plot_10_pics(best_pics, '10 best pictures', root_dir='/itf-fi-ml/shared/IN5400/2022_mandatory1/', filename=filename)
    
    return

def load_model_and_data(taskname):
    model_path = './' + taskname + '.pt'
    model = torch.load(model_path)

    data_transforms = {
        'train_rgb': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ChannelSelect(),
            transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]) # RGB
        ]),
        'val_rgb': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ChannelSelect(),
            transforms.Normalize([0.7476, 0.6534, 0.4757], [0.1677, 0.1828, 0.2137]) # RGB
        ]),
        'train_all': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # RGBa
        ]),
        'val_all': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.7476, 0.6534, 0.4757, 0.0960], [0.1677, 0.1828, 0.2137, 0.0284]) # RGBa
        ]),
    }

    root_dir = '/itf-fi-ml/shared/IN5400/2022_mandatory1/'
    if taskname=='Task1':
        image_datasets = RainforestDataset(root_dir=root_dir, trvaltest=1, transform=data_transforms['val_rgb'])
    else:
        image_datasets = RainforestDataset(root_dir=root_dir, trvaltest=1, transform=data_transforms['val_all'])
    
    dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=config['batchsize_val'], shuffle=False, num_workers=1)
    
    values = np.load('./' + taskname + '.npz')

    return model, dataloader, values

def compare_pred_scores(model, dataloader, criterion, device, numcl, values, taskname):
    if taskname=='Task1' or taskname=='Task4':
        perfmeasure, testloss, concat_labels, concat_pred, fnames = evaluate_meanavgprecision(model, dataloader, criterion, device, numcl)
    else:
        perfmeasure, testloss, concat_labels, concat_pred, fnames = evaluate_meanavgprecision_twonetwork(model, dataloader, criterion, device, numcl)

    avgperfmeasure = np.nanmean(perfmeasure)

    # compare predictions
    old_preds = values['preds']
    pred_differance = np.abs(old_preds - concat_pred)
    print(taskname ,' - Differace in AP: ', abs(perfmeasure - values['AP']))
    print(taskname ,' - Differace in mAP: ', abs(avgperfmeasure - values['mAP']))
    print(taskname ,' - Mean differace in predictions: ', np.mean(pred_differance))
    print(taskname ,' - Max differace in predictions: ', np.max(pred_differance))
    print('\n')

    return

def loss_plots(values, num_epochs, filename):
    trainlosses = values['trainlosses']
    testlosses = values['testlosses']
    testperfs = values['testperfs']

    plt.plot(range(num_epochs), trainlosses)
    plt.xlabel('Epoch')
    plt.ylabel('Train loss')
    plt.title('BCE loss per epoch')
    plt.savefig('img/' + filename + '_trainlosses' + '.png')
    plt.close()

    plt.plot(range(num_epochs), testlosses)
    plt.xlabel('Epoch')
    plt.ylabel('Test loss')
    plt.title('BCE loss per epoch')
    plt.savefig('img/' + filename + '_testlosses' + '.png')
    plt.close()

    plt.plot(range(num_epochs), np.mean(testperfs, axis=1))
    plt.xlabel('Epoch')
    plt.ylabel('Test mAP')
    plt.title('Test mAP per epoch')
    plt.savefig('img/' + filename + '_mAP' + '.png')
    plt.close()

    classes, num_classes = get_classes_list()
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(testperfs.shape[1]):
        AP = testperfs[:,i]
        line, = ax.plot(range(num_epochs), AP, label=classes[i])
    plt.xlabel('Epoch')
    plt.ylabel('Test AP')
    plt.title('Classwise test AP per epoch')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('img/' + filename + '_AP' + '.png')
    plt.close()

    return

def best_epoch(values, taskname):
    print(taskname ,' - Best epoch: ', values['epoch'])
    print(taskname ,' - Best epoch AP: ', values['AP'])
    print(taskname ,' - Best epoch mAP: ', values['mAP'])

if __name__=='__main__':
    torch.manual_seed(0)

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

    # Device
    if True == config['use_gpu']:
        device= torch.device('cuda:0')
    else:
        device= torch.device('cpu')

    tasknames = ['Task1', 'Task3', 'Task4']
    for taskname in tasknames:
        model, dataloader, values = load_model_and_data(taskname=taskname)
        model = model.to(device)
        criterion = nn.BCELoss()
        best_epoch(values, taskname)
        compare_pred_scores(model, dataloader, criterion, device, config['numcl'], values, taskname)
        loss_plots(values, config['maxnumepochs'], taskname)
        if taskname=='Task1':
            tailacc(values)