import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
from unet import UNet
import shutil
import os
import time

from additionals import *

# Specify data and output folders
pth_data = '/itf-fi-ml/shared/courses/IN5400/tranfers/in5400_2022_lab_segmentation_data'
pth_work = None  # [optional] '/path-to-somewhere-you-can-store-intermediate-results/'

if pth_work is not None:
    # Make a copy of the current script
    copied_script_name = time.strftime("%Y%m%d_%H%M") + '_' + os.path.basename(__file__)  # generate filename with timestring
    shutil.copy(__file__, os.path.join(pth_work, copied_script_name))  # copy script
    print('Wrote file: {}'.format(copied_script_name))

# Read list of files used to train
with open(os.path.join(pth_data, 'annotations/trainval.txt')) as f:
    lines = f.read().splitlines()
im_list_train = [line.split(' ')[0]+'.jpg' for line in lines]

# Read list of files used for validation
with open(os.path.join(pth_data, 'annotations/test.txt')) as f:
    lines = f.read().splitlines()
im_list_val = [line.split(' ')[0]+'.jpg' for line in lines]

mpl.use('Agg')

# Create data loaders for both the training and validation set
pth_images = os.path.join(pth_data, 'images/')
pth_gt = os.path.join(pth_data, 'annotations/trimaps/')
datasets = {'train': PetDataset(im_list_train, pth_images, pth_gt), 'val': PetDataset(im_list_val, pth_images, pth_gt)}
dataloaders = {split: torch.utils.data.DataLoader(datasets[split], batch_size=3, shuffle=True, pin_memory=True)
               for split in ('train', 'val')}

if 0:
    # ### Visualize an example image and ground-truth mask from the training data
    ind = random.choice(range(len(datasets['train'])))
    im, im_gt = datasets['train'][ind]

    im = np.transpose(im.detach().numpy(), [1, 2, 0])
    plt.imshow(im)
    plt.show()
    plt.imshow(im_gt)
    plt.show()

device = 'cuda'  # 'cuda'/'cpu'

model = UNet(chs=(3, 64, 128, 256), num_class=3)

# Verify that the model produces an output layer having the correct shape
out = model(torch.rand((1, 3, 224, 224))).detach()  # Run a single dummy-image through the model
assert out.shape == (1, 3, 224, 224),\
    "The output of the model for a 224 x 224 input image should be num_class x 224 x 224."

model.to(device=device)  # Move the model onto GPU if so opted for

# Specify the loss function
class_weights = torch.tensor([1, 1, 1], dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device=device)

# Specify the optimizer and the learning-rate scheduler
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=1e-5)
# Move from zero to 1e-2 lr in 500 iterations, then back down in 1000 iterations; iterate while halving the
# maximum amplitude
scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0, max_lr=1e-2, step_size_up=500, step_size_down=1000,
                                        base_momentum=0.9, max_momentum=0.9, mode='triangular2')

total_epochs = 150  # Specify for how long we are prepared to train the model

epoch_loss = {'train': [], 'val': []}

# Train the model by looping through all the images 'total_epochs' times
for epoch in range(total_epochs):
    for phase in ('train', 'val'):
        is_training = phase == 'train'
        model.train(is_training)
        torch.set_grad_enabled(is_training)
        torch.cuda.empty_cache()

        start_time = time.time()

        total_loss = 0
        total_images = 0

        for ix, (ims, ims_gt) in enumerate(tqdm(dataloaders[phase])):

            if ix > 100:   # Cut the epochs short, a mere 100 iterations
                break

            # Put the images and targets on GPU if so opted for
            ims = ims.to(device=device)
            targets = ims_gt.type(torch.long).to(device=device)

            # Run images through the model, producing a num_class x 224 x 224 output for each
            outputs = model(ims)

            # During the first iteration, write a sample image, output and target
            if ix == 0:
                write_sample(ims, outputs, targets, pth_work, epoch, phase)

            # Calculate the loss
            loss = criterion(outputs, targets)

            # Accumulate losses for later display
            total_loss += loss.item()
            total_images += ims.shape[0]

            if is_training:
                # Update the weights by doing back-propagation
                # [Note that by doing this last, the CPU can work on loading images for the next iteration while the
                # GPU handles this task]
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            del loss, outputs  # Make sure we allow freeing these on GPU to save on memory

        elapsed_time = time.time() - start_time
        print('[{}]\tEpoch: {}/{}\t loss: {:.5f}\tlr: {:.7f}'.format(phase, epoch+1, total_epochs,
                                                                     total_loss/total_images, get_lr(optimizer)))

        epoch_loss[phase].append(total_loss/total_images)

    if pth_work is not None:
        # Plot train / val losses and write it to file
        plt.clf()
        plt.plot(epoch_loss['train'], label='train')
        plt.plot(epoch_loss['val'], label='val')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.grid(True)
        fname1 = os.path.join(pth_work, 'fig_train_val_loss.png')
        plt.savefig(fname1)
        print('Wrote {}'.format(fname1))

