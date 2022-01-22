import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck

num_classes = 1000
 
class ParallelResnet(ResNet):
    def __init__(self, dev0, dev1, *args, **kwargs):
        super(ParallelResnet, self).__init__(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)
        # dev0 and dev1 each point to a GPU device (usually gpu:0 and gpu:1)
        self.dev0 = dev0
        self.dev1 = dev1
 
        # splits the model in two consecutive sequences : seq0 and seq1 
        self.seq0 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2
        ).to(self.dev0)  # sends the first sequence of the model to the first GPU
 
        self.seq1 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to(self.dev1)  # sends the second sequence of the model to the second GPU
 
        self.fc.to(self.dev1)  # last layer is on the third, fourth GPU
 
    def forward(self, x):
        x= self.seq0(x)     # apply first sequence of the model on input x
        x= x.to(self.dev1)  # send the intermediary result to the second GPU
        x = self.seq1(x)    # apply second sequence of the model to x
        return self.fc(x.view(x.size(0), -1))  


class PipelinedResnet(ResNet):
    def __init__(self, dev0, dev1, split_size=8, *args, **kwargs):
        super(PipelinedResnet, self).__init__(
            Bottleneck, [3, 4, 23, 3], num_classes=num_classes, *args, **kwargs)
        # dev0 and dev1 each point to a GPU device (usually gpu:0 and gpu:1)
        self.dev0 = dev0
        self.dev1 = dev1
        self.split_size = split_size
 
        # splits the model in two consecutive sequences : seq0 and seq1 
        self.seq0 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            self.layer1,
            self.layer2
        ).to(self.dev0)  # sends the first sequence of the model to the first GPU
 
        self.seq1 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool,
        ).to(self.dev1)  # sends the second sequence of the model to the second GPU
 
        self.fc.to(self.dev1)  # last layer is on the second GPU
 
    def forward(self, x):
        # split setup for x, containing a batch of (image, label) as a tensor
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        # initialisation: 
        # - first mini batch goes through seq0 (on dev0)
        # - the output is sent to dev1
        s_prev = self.seq0(s_next).to(self.dev1)
        ret = []
 
        for s_next in splits:
            # A. s_prev runs on dev1
            s_prev = self.seq1(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
 
            # B. s_next runs on dev0, which can run concurrently with A
            s_prev = self.seq0(s_next).to(self.dev1)
 
        s_prev = self.seq1(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)))
 
        return torch.cat(ret)


def train(model, optimizer, criterion, train_loader, batch_size, dev0, dev1, dev2, dev3):
    model.train()
    for batch_counter, (images, labels) in enumerate(train_loader):
        # images are sent to the first GPU
        images = images.to(dev0, non_blocking=True)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        outputs = model(images)
        # labels (ground truth) are sent to the GPU where the outputs of the model
        # reside, which in this case is the second GPU 
        labels = labels.to(outputs.device, non_blocking=True)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()