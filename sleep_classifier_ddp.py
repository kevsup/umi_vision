import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.multiprocessing as mp
import torch.distributed as dist
import matplotlib.pyplot as plt
import time
import copy
import argparse
import os
import pdb

root = 'data'
#device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(gpu, args):
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(                                   
    	backend='gloo',                                         
    	world_size=args.world_size,                              
    	rank=rank                                               
    )    

    print('Umi')

    torch.manual_seed(0)
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    #model.fc = nn.Linear(num_ftrs, 2)
    model.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs // 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(num_ftrs // 8, 2))

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    print(model)

    transform = transforms.Compose([transforms.Lambda(cropBottomLeft), transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(root, transform)

    train_set, val_set = torch.utils.data.random_split(dataset, [0.7, 0.3])

    # DDP
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas = args.world_size,
        rank = rank
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle=False, num_workers=0, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32, shuffle=False, num_workers=0)

    class_names = ('sleep', 'wake')
    '''
    # Get a batch of training data
    inputs, classes = next(iter(train_loader))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
    pdb.set_trace()
    '''
    weights = torch.tensor([0.35, 0.65]).cuda()
    criterion = nn.CrossEntropyLoss(weight=weights).cuda(gpu)
    # try focal loss later: https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    dataloaders={'train':train_loader, 'val':val_loader}
    dataset_sizes={'train':len(train_set),'val':len(val_set)}

    num_epochs = args.epochs

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        # Each epoch has a training and validation phase
        phases = ['train']
        if rank == 0:
            phases = ['train', 'val']
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            neg_recall = [0, 0]
            pos_recall = [0, 0]

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                neg_recall[0] += torch.sum(preds[labels.data == 0] == labels.data[labels.data == 0])
                pos_recall[0] += torch.sum(preds[labels.data == 1] == labels.data[labels.data == 1])
                neg_recall[1] += torch.sum(labels.data == 0)
                pos_recall[1] += torch.sum(labels.data == 1)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | Neg recall: {neg_recall[0]/neg_recall[1]:.4f} | Pos recall: {pos_recall[0]/pos_recall[1]:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    if rank == 0:
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model, 'ddp_test.pth')

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.show()

def cropBottomLeft(image):
    return transforms.functional.crop(image, 450, 190, 720-450, 425)

def visualize_model(model, dataloader, class_names, num_images=1):
    was_training = model.training
    model.eval()
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                if j % 10 == 0:
                    imshow(inputs.cpu().data[j])
                    print('model predicts:', class_names[preds[j]], ' | actual:', class_names[labels[j]])

        model.train(mode=was_training)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '2601:647:5e80:4d60:7a30:73a0:f9d6:1102'
    os.environ['MASTER_PORT'] = '8888'
    train_model(0, args)
    #mp.spawn(train_model, nprocs = args.gpus, args=(args,))