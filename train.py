from __future__ import print_function

import os
import argparse

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.utils import linear_assignment_
from sklearn.manifold import TSNE
from scipy.stats import itemfreq
from sklearn.cluster import KMeans
from itertools import chain

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn

import configs
import utils
from data.load import load_datasets
import models

assert torch.cuda.is_available(), 'Error: CUDA not found!'

def main():
    args = parse_args()
    train_model(batch_size=args.batch_size,
                n_epochs=args.n_epochs,
                learning_rate = args.learning_rate,
                saved_epoch = args.saved_epoch,
                run_id=args.run_id)
    print("ok")

def train_model(batch_size, n_epochs, learning_rate,
                saved_epoch,
                run_id="def", set_name="stanford_dogs",
                save_every=1000, save_path=configs.models,
                plot_every=500, plot_path=configs.plots):
    # Setup save directories
    if save_path:
        save_path = os.path.join(save_path, "run_{}".format(run_id))
        os.makedirs(save_path, exist_ok=True)
    if plot_path:
        plot_path = os.path.join(plot_path, "run_{}".format(run_id))
        os.makedirs(plot_path, exist_ok=True)

    # Load network and use GPU
    net = models.Net2().cuda()
    cudnn.benchmark = True

    # Load dataset
    train_data, test_data, classes = load_datasets(set_name)
    #train_y, test_y = utils.get_labels(train_data), utils.get_labels(test_data)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # obtain one batch of training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    images = np.swapaxes(np.swapaxes(images.numpy(), 1, 2), 2, 3)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    for idx in np.arange(batch_size):
        ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(classes[labels[idx]], {'fontsize': batch_size/5}, pad=0.4)
    plt.tight_layout(pad=1, w_pad=0, h_pad=0)
    if plot_path:
        plt.savefig(os.path.join(plot_path, "Initial_Visualization"))
    else:
        plt.show()
    plt.clf()

    # cross entropy loss combines softmax and nn.NLLLoss() in one single class.
    criterion = nn.NLLLoss()

    # stochastic gradient descent with a small learning rate
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # ToDo: Add to utils
    # Calculate accuracy before training
    correct = 0
    total = 0

    # Iterate through test dataset
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()

        # forward pass to get outputs
        # the outputs are a series of class scores
        outputs = net(images)

        # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)

        # count up total number of correct labels
        # for which the predicted and true labels are equal
        total += labels.size(0)
        correct += (predicted == labels).sum()

    # calculate the accuracy
    # to convert `correct` from a Tensor into a scalar, use .item()
    accuracy = 100.0 * correct.item() / total

    # print('Accuracy before training: ', accuracy)

    def train(n_epochs):
        net.train()
        loss_over_time = [] # to track the loss as the network trains

        for epoch in range(n_epochs):  # loop over the dataset multiple times
            output_epoch = epoch + saved_epoch
            running_loss = 0.0

            for batch_i, data in enumerate(train_loader):
                # get the input images and their corresponding labels
                inputs, labels = data
                inputs, labels = inputs.cuda(), labels.cuda()

                # zero the parameter (weight) gradients
                optimizer.zero_grad()

                # forward pass to get outputs
                outputs = net(inputs)

                # calculate the loss
                loss = criterion(outputs, labels)

                # backward pass to calculate the parameter gradients
                loss.backward()

                # update the parameters
                optimizer.step()

                # print loss statistics
                # to convert loss into a scalar and add it to running_loss, we use .item()
                running_loss += loss.item()

                if batch_i % 45 == 44:    # print every 45 batches
                    avg_loss = running_loss/45
                    # record and print the avg loss over the 100 batches
                    loss_over_time.append(avg_loss)
                    print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(output_epoch + 1, batch_i+1, avg_loss))
                    running_loss = 0.0
            if output_epoch % 100 == 99: # save every 100 epochs
                torch.save(net.state_dict(), 'saved_models/Net2_{}.pt'.format(output_epoch + 1))

        print('Finished Training')
        return loss_over_time

    if saved_epoch:
        net.load_state_dict(torch.load('saved_models/Net2_{}.pt'.format(saved_epoch)))

    # call train and record the loss over time
    training_loss = train(n_epochs)

    # visualize the loss as the network trained
    fig = plt.figure()
    plt.plot(45*np.arange(len(training_loss)), training_loss)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.xlabel('Number of Batches', fontsize=12)
    plt.ylabel('loss', fontsize=12)
    plt.ylim(0, 5.5) # consistent scale
    plt.tight_layout()
    if plot_path:
        plt.savefig(os.path.join(plot_path, "Loss_Over_Time"))
        print("saved")
    else:
        plt.show()
    plt.clf()

    # initialize tensor and lists to monitor test loss and accuracy
    test_loss = torch.zeros(1).cuda()
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    # set the module to evaluation mode
    # used to turn off layers that are only useful for training
    # like dropout and batch_norm
    net.eval()

    for batch_i, data in enumerate(test_loader):

        # get the input images and their corresponding labels
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        # forward pass to get outputs
        outputs = net(inputs)

        # calculate the loss
        loss = criterion(outputs, labels)

        # update average test loss
        test_loss = test_loss + ((torch.ones(1).cuda() / (batch_i + 1)) * (loss.data - test_loss))

        # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)

        # compare predictions to true label
        # this creates a `correct` Tensor that holds the number of correctly classified images in a batch
        correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))

        # calculate test accuracy for *each* object class
        # we get the scalar value of correct items for a class, by calling `correct[i].item()`
        for l, c in zip(labels.data, correct):
            class_correct[l] += c.item()
            class_total[l] += 1

    print('Test Loss: {:.6f}\n'.format(test_loss.cpu().numpy()[0]))

    for i in range(len(classes)):
        if class_total[i] > 0:
            print('Test Accuracy of %30s: %2d%% (%2d/%2d)' % (
                classes[i], 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print('Test Accuracy of %5s: N/A (no training examples)' % (classes[i]))


    print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

    # Visualize Sample Results (Runs until a batch contains a )
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(batch_size/4+5, batch_size/4+5))
    misclassification_found = False
    while(not misclassification_found):
        fig.clf()
        # obtain one batch of test images
        dataiter = iter(test_loader)
        images, labels = dataiter.next()
        images, labels = images.cuda(), labels.cuda()
        # get predictions
        preds = np.squeeze(net(images).data.max(1, keepdim=True)[1].cpu().numpy())
        images = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, 2), 2, 3)
        for idx in np.arange(batch_size):
            ax = fig.add_subplot(batch_size/8, 8, idx+1, xticks=[], yticks=[])
            ax.imshow(images[idx])
            if preds[idx]==labels[idx]:
                ax.set_title("{}".format(classes[preds[idx]], classes[labels[idx]]), color="green")
            else:
                ax.set_title("({})\n{}".format(classes[labels[idx]], classes[preds[idx]]), color="red", pad=.4)
                misclassification_found = True
    if plot_path:
        plt.savefig(os.path.join(plot_path, "Results Visualization"))
    else:
        plt.show()
    plt.clf()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch DML Training')
    parser.add_argument('--batch_size', help='Batch_size', default=64, type=int)
    parser.add_argument('--n_epochs', help='Number of Epochs', default=5, type=int)
    parser.add_argument('--learning_rate', help='Learning Rate', default=0.01, type=float)
    parser.add_argument('--saved_epoch', help='epoch of saved model', default=None, type=int)
    parser.add_argument('--run_id', help='Used to help identify artifacts', default=0, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
