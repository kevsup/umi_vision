import torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import pdb

root = 'data'
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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
                inputs = inputs.to(device)
                labels = labels.to(device)

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
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc.cpu().numpy()

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
    transform = transforms.Compose([transforms.Lambda(cropBottomLeft), transforms.Resize((224, 224)), transforms.ToTensor()])

    dataset = torchvision.datasets.ImageFolder(root, transform)

    k = 2
    highest_acc = 0
    accuracies = []
    for i in range(k):
        train_set, val_set = torch.utils.data.random_split(dataset, [0.7, 0.3])

        train_loader = torch.utils.data.DataLoader(train_set, batch_size = 32, shuffle=True, num_workers=0)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32, shuffle=True, num_workers=0)

        class_names = ('sleep', 'wake')
        '''
        # Get a batch of training data
        inputs, classes = next(iter(train_loader))
        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)
        imshow(out, title=[class_names[x] for x in classes])
        pdb.set_trace()
        '''

        model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        #model.fc = nn.Linear(num_ftrs, 2)
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_ftrs // 8), nn.ReLU(), nn.Dropout(0.5), nn.Linear(num_ftrs // 8, 2))
        print(model)

        model = model.to(device)
        weights = torch.tensor([0.35, 0.65]).cuda()
        criterion = nn.CrossEntropyLoss(weight=weights)
        # try focal loss later: https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html 
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.01)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        best_model, best_acc = train_model(model, criterion, optimizer, \
                    exp_lr_scheduler, dataloaders={'train':train_loader, 'val':val_loader}, \
                    dataset_sizes={'train':len(train_set),'val':len(val_set)}, num_epochs=15)
        accuracies.append(best_acc)
        if highest_acc < best_acc:
            highest_acc = best_acc
            torch.save(best_model, 'sleep_model.pth')
    print("accuracies:", accuracies)
    alpha = 0.95
    p = ((1 - alpha)/2.0)*100
    lower = max(0.0, np.percentile(accuracies, p))
    p = (alpha+((1-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(accuracies, p))
    print("95 percent confidence interval:", lower, "to", upper)
