import os
import shutil
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models

## Dataset
images_dir = r'..\dataset'

categories = os.listdir(images_dir)
m = 0
for category in categories:
    category_dir = os.path.join(images_dir, category)
    class_size = len(os.listdir(category_dir))
    print('Images belonging to class "', category, '":', class_size)
    m += class_size
print('We have ', m, ' images') # 1800

val_dir = r'..\val'
os.mkdir(val_dir)
test_dir = r'..\test'
os.mkdir(test_dir)

print(categories) # ["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
print(len(categories), 'classes') # 6

for category in categories:
    path = os.path.join(train_dir, category)
    os.mkdir(path)
    path = os.path.join(val_dir, category)
    os.mkdir(path)
    path = os.path.join(test_dir, category)
    os.mkdir(path)
    
split = int((m*.2)//6)
print(split)

for class_name in categories:
    src_dir = os.path.join(images_dir, class_name)
    relocated_images = os.listdir(src_dir)[:split]
    for image in relocated_images:
        src = os.path.join(src_dir, image)
        dst = os.path.join(test_dir, class_name, image)
        shutil.move(src, dst)
        
for class_name in categories:
    src_dir = os.path.join(images_dir, class_name)
    relocated_images = os.listdir(src_dir)[:split]
    for image in relocated_images:
        src = os.path.join(src_dir, image)
        dst = os.path.join(val_dir, class_name, image)
        shutil.move(src, dst)

## Check
for category in categories:
    category_dir = os.path.join(test_dir, category)
    class_size = len(os.listdir(category_dir))
    print('In test set, images belonging to class "', category, '":', class_size)

for category in categories:
    category_dir = os.path.join(val_dir, category)
    class_size = len(os.listdir(category_dir))
    print('In validation set, images belonging to class "', category, '":', class_size)
            
train_dir = r'..\dataset'
for category in categories:
    category_dir = os.path.join(train_dir, category)
    class_size = len(os.listdir(category_dir))
    print('In training set, images belonging to class "', category, '":', class_size)

## Preprocessing
transforms = transforms.Compose([transforms.Resize(100), transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                       std=[0.229, 0.224, 0.225])])

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=transforms)
val_data = torchvision.datasets.ImageFolder(root=val_dir, transform=transforms)
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=transforms)

batch_size = 8
train_data_loader = DataLoader(train_data, batch_size=batch_size)
val_data_loader = DataLoader(val_data, batch_size=batch_size)
test_data_loader = DataLoader(test_data, batch_size=batch_size)

class CNNNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNNet, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding='same'), nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2),
                                      
                                      nn.Conv2d(16, 32, kernel_size=3, padding='same'), nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2),
                                      
                                      nn.Conv2d(32, 64, kernel_size=3, padding='same'), nn.ReLU(),
                                      nn.MaxPool2d(kernel_size=2))
        self.classifier = nn.Sequential(nn.Linear(9216, 128), nn.ReLU(),
                                        nn.Linear(128, 6)) # classification layer
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # flattening the output of convolution part of the network
        x = self.classifier(x)
        return x
    
cnn = CNNNet()

optimizer = optim.Adam(cnn.parameters(), lr=5e-5)

device = torch.device("cuda")

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs=60):
    # training
    model.to(device)
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for inputs, targets in train_loader: # for a batch
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device) # utilizing the GPU
            output = model(inputs) # obtaining outputs of the network
            loss = loss_fn(output, targets) # computing loss
            loss.backward() # computing the gradients
            optimizer.step() # using gradients for optimization of parameters
            training_loss += loss.data.item()
        training_loss /= len(train_loader) # computing training loss at the end of an epoch
    
        model.eval()
        correct = 0
        total = 0

        # validation
        for inputs, targets in val_loader: # for a batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item()
            # computing accuracy
            _, y_pred_tags = output.max(1)
            total += targets.size(0)
            correct += y_pred_tags.eq(targets).sum().item()
        acc = 100.*correct/total
        valid_loss /= len(val_loader)
        print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, Accuracy: {:.2f}'.format(epoch, training_loss,
                                                                                                    valid_loss, acc))


train(cnn, optimizer, torch.nn.CrossEntropyLoss(), train_data_loader, val_data_loader)

torch.save(cnn, r'..\cnn')

cnn = torch.load(r'..\cnn')

## Final Evaluation
test_loss = 0
total = 0
correct = 0
acc = list()
for inputs, targets in test_data_loader:
    inputs = inputs.to(device)
    output = cnn(inputs)
    targets = targets.to(device)
    batch_loss = torch.nn.CrossEntropyLoss()(output, targets)
    test_loss += batch_loss.data.item() # cumulatively adding losses on batches
    _, y_pred_tags = output.max(1)
    total += targets.size(0)
    correct += y_pred_tags.eq(targets).sum().item()
    acc.append(100.*correct/total) # accuracy on a batch

test_loss /= len(test_data_loader) # total test loss
avg_acc = np.mean(acc)

print('Test loss: {:.2f}, Avg accuracy: {:.2f}'.format(test_loss, avg_acc))  