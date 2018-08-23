## Define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

dtype = torch.cuda.FloatTensor
device = torch.device("cuda:0")
#nn = nn.cuda()

class BasicNN(nn.Module):
    def __init__(self):
        super(BasicNN, self).__init__()
        self.net = nn.Linear(28 * 28, 10)
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.net(x)

class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()

        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F)/S +1 = (28-3)/1 +1 = 26
        # the output Tensor for one image, will have the dimensions: (10, 26, 26)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(1, 10, 3)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(10, 20, 3)

        # 20 outputs * the 5*5 filtered/pooled map size
        # 10 output channels (for the 10 classes)
        self.fc1 = nn.Linear(20*5*5, 10)


    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)

        # one linear layer
        x = F.relu(self.fc1(x))
        # a softmax layer to convert the 10 outputs into a distribution of class scores
        x = F.log_softmax(x, dim=1)

        # final output
        return x


class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()

        ## Define all the layers of this CNN, the only requirements are:
        ## This network takes in a square (224 x 224), RGB image as input
        ## 120 output channels/feature maps

        # 1 - input image channel (RGB), 32 output channels/feature maps, 4x4 square convolution kernel
        # 2x2 max pooling with 10% droupout
        # ConvOut: (32, 221, 221) <-- (W-F+2p)/s+1 = (224 - 4)/1 + 1
        # PoolOut: (32, 110, 110) <-- W/s
        self.conv1 = nn.Conv2d(3, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p = 0.1)

        # 2 - 64 output channels/feature maps, 3x3 square convolution kernel
        # 2x2 max pooling with 20% droupout
        # ConvOut: (64, 108, 108) <-- (W-F+2p)/s+1 = (110 - 3)/1 + 1
        # PoolOut: (64, 54, 54) <-- W/s
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(p = 0.2)

        # 3 - 128 output channels/feature maps, 2x2 square convolution kernel
        # 2x2 max pooling with 30% droupout
        # ConvOut: (128, 53, 53) <-- (W-F+2p)/s+1 = (54 - 2)/1 + 1
        # PoolOut: (128, 26, 26) <-- W/s
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(p = 0.3)

        # 4 - 256 output channels/feature maps, 3x3 square convolution kernel
        # 2x2 max pooling with 30% droupout
        # ConvOut: (256, 24, 24) <-- (W-F+2p)/s+1 = (24 - 3)/1 + 1
        # PoolOut: (256, 12, 12) <-- W/s
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout(p = 0.4)

        # Fully Connected Layers
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.dropout5 = nn.Dropout(p = 0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.dropout6 = nn.Dropout(p = 0.6)
        self.fc3 = nn.Linear(1000, 250)
        self.dropout7 = nn.Dropout(p = 0.7)
        self.fc4 = nn.Linear(250, 120)


    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # convolutions
        x = self.dropout1(self.pool1(F.relu(self.bn1(self.conv1(x)))))
        x = self.dropout2(self.pool2(F.relu(self.bn2(self.conv2(x)))))
        x = self.dropout3(self.pool3(F.relu(self.bn3(self.conv3(x)))))
        x = self.dropout4(self.pool4(F.relu(self.bn4(self.conv4(x)))))

        #flatten
        x = x.view(x.size(0), -1)

        #fully connected
        x = self.dropout5(self.fc1(x))
        x = self.dropout6(self.fc2(x))
        x = self.dropout7(self.fc3(x))
        x = self.fc4(x)
        # a softmax layer to convert the 120 outputs into a distribution of class scores
        x = F.log_softmax(x, dim=1)

        # a modified x, having gone through all the layers of your model, should be returned
        return x
