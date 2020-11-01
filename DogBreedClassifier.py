from abc import ABC

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import tqdm
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFile
from path_names import model_path
from parameters import kernel_size, pool_size, stride, padding_size, dropout_probability


def print_error_check(train_data, test_data, val_data):
    """ Loads all the images in datasets, and checks if they are not corrupted.
    Prints a warning message if any of the files are corrupted
    inputs: train_data -> Training Data Loader
            test_data-> Testing Data Loader
            val_data-> Validation Data loader"""

    for DUT in [train_data, test_data, val_data]:
        for fn, label in tqdm.tqdm(DUT.imgs):
            try:
                im = ImageFile.Image.open(fn)
                im2 = im.convert('RGB')
            except OSError:
                print("Cannot load : {}".format(fn))


def load_data(train_dir, test_dir, val_dir, batch_size=20, error_check=0, input_size=256):
    """
    Loads data from specified path to tensors.
    inputs:
    train_dir: Path for training images
    testing_dir: Path for testing images
    Validation Images: Path for validation images
    batch_size: Images to be processed in one batch
    error_check: if this flag is set, the images from directories will be checked for error.
    Prints warning in case of load error
    input_size: Sice of the CNN input image
    """

    num_workers = 0
    size = (input_size, input_size)
    data_transforms = transforms.Compose([transforms.Resize(size),
                                          transforms.RandomAffine(10),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(20),
                                          transforms.RandomResizedCrop(256),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    data_transforms_val = transforms.Compose([transforms.Resize(size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=data_transforms_val)
    val_data = datasets.ImageFolder(val_dir, transform=data_transforms_val)
    if error_check:
        print_error_check(train_data, test_data, val_data)

    train_data = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers)
    test_data = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)
    val_data = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers)
    return train_data, test_data, val_data


def visualize_data(data_loader, batch_size=20):
    """
    Data visualization from a data loader
    :param data_loader: The torch data loader
    :param batch_size :Number of images in one batch
    :return: None
    """
    data_iter = iter(data_loader)
    images, labels = data_iter.next()
    images = images.numpy()
    fig = plt.figure(figsize=(25, 4))
    for index in np.arange(batch_size):
        ax = fig.add_subplot(2, batch_size / 2, index + 1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[index], (1, 2, 0)))


class DogBreedNet(nn.Module, ABC):
    """
    The model class:
    """

    def __init__(self):
        super(DogBreedNet, self).__init__()
        # Define Network Architecture
        self.con_1_1 = nn.Conv2d(3, 16, kernel_size, padding=padding_size)
        self.con_2_1 = nn.Conv2d(16, 32, kernel_size, padding=padding_size)
        self.con_3_1 = nn.Conv2d(32, 64, kernel_size, padding=padding_size)
        self.con_4_1 = nn.Conv2d(64, 128, kernel_size, padding=padding_size)
        self.pool = nn.MaxPool2d(pool_size, pool_size)
        self.fc_1 = nn.Linear(128 * 32 * 32, 1024)
        self.fc_2 = nn.Linear(1024, 512)
        self.fc_3 = nn.Linear(512, 120)
        self.dropout = nn.Dropout(dropout_probability)

    def forward(self, x):
        """:param x  : Input image
           :returns : model output
        """
        x = self.pool(F.relu(self.con_1_1(x)))
        x = self.pool(F.relu(self.con_2_1(x)))
        x = self.pool(F.relu(self.con_3_1(x)))
        x = F.relu(self.con_4_1(x))
        x = x.view(x.shape[0], -1)
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.dropout(F.relu(self.fc_2(x)))
        x = self.dropout(F.log_softmax(self.fc_3(x), dim=1))
        return x


def get_device():
    """
    Function verifies if a GPU ia available.
    If available it returns a GPU, else it returns a CPU device
        :param: None
        :returns GPU or CPU device, a flag designating the same
    """
    train_on_gpu = torch.cuda.is_available()
    use_gpu = False
    if not train_on_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        use_gpu = True
    return device, use_gpu


def save_model(model, valid_loss_min, valid_loss):
    """ Saves the trained model
        :param model  : The CNN model class
        :param valid_loss_min : The minimum validation loss that triggered save function
        :param  valid_loss : Previous validation loss
    """
    torch.save(model.state_dict(), model_path)
    print('Validation loss decreased ({:.6f} --> {:.6f}).Saving model ...'.format(valid_loss_min, valid_loss))


def load_model(path):
    """Loads the trained model from stae dict
       :param path: Path load the model
       :returns : loaded model
    """
    model = DogBreedNet()
    try:
        model.load_state_dict(torch.load(path))
    except OSError:
        model = None
        print("Could not open/read file")

    return model


def find_accuracy(val_output, val_labels):
    """
        :Calculates the accuracy of current batch fw propagation through the model
        :param val_output : The output of validation batch after fw propagation
        :param val_labels : The actual label of the input batch
        :returns : Accuracy
    """
    probabilities = torch.exp(val_output)
    _, top_classes = probabilities.topk(1, dim=1)
    equals = top_classes == val_labels.view(*top_classes.shape)
    accuracy = torch.mean(torch.tensor(equals.type(torch.FloatTensor)))
    return accuracy


def train_network(train_data_loader, val_data_loader, model, device, criterion, optimizer, epochs=20):
    """ Trains the model, for a given number of epochs. Plots the training loss,
    validation loss and accuracy Vs Epoch
    :param train_data_loader : Training torch data loader object
    :param val_data_loader : Validation Data loader Object
    :param model : Network model to be trained
    :param device : Training device, GPU if available
    :param criterion : The loss criterion of the network
    :param optimizer : The optimizer algorithm
    :param epochs : Training Epochs , defaulted to 20
    :returns  training_loss : The training loss list, for all the epochs
    :returns  validation_loss: Validation loss for each epochs
    :returns validation_accuracy: Validation Accuracy for Each Epochs
    """
    model.to(device)
    over_all_progress = 0
    training_loss, validation_loss, validation_accuracy = [], [], []
    val_loss_min = np.inf
    for epoch in range(epochs + 1):
        over_all_progress += 1
        train_loss = 0
        val_loss = 0
        train_progress = 0
        model.train()
        desc = "Epoch " + str(epoch)
        for loop_index in tqdm.tqdm(range(len(train_data_loader)), desc=desc):
            data_iter = iter(train_data_loader)
            try:
                images, labels = data_iter.next()
            except OSError:
                continue
            train_progress += 1
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model.forward(images)
            batch_loss = criterion(out, labels)
            train_loss += batch_loss.item()
            # images.to("cpu")
            batch_loss.backward()
            optimizer.step()
        else:
            model.eval()
            val_accuracy = 0
            report_accuracy = 0
            with torch.no_grad():
                for val_images, val_labels in val_data_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(device)
                    val_output = model.forward(val_images)
                    val_batch_loss = criterion(val_output, val_labels)
                    val_loss += val_batch_loss.item()
                    val_accuracy += find_accuracy(val_output, val_labels)
            train_loss /= len(train_data_loader)
            val_loss /= len(val_data_loader)
            val_accuracy /= len(val_data_loader)
            print('\nVal Accuracy%:  {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'
                  .format(val_accuracy * 100, train_loss, val_loss))
            training_loss.append(train_loss)
            validation_loss.append(val_loss)
            validation_accuracy.append(val_accuracy)
            if val_loss < val_loss_min:
                x = val_loss_min
                val_loss_min = val_loss
                save_model(model, x, val_loss)
    plt.plot(training_loss, label='Training Loss')
    plt.plot(validation_loss, label='Validation Loss')
    plt.plot(validation_accuracy, label='Validation Accuracy')
    plt.legend(frameon=False)

    return training_loss, validation_loss, validation_accuracy


def test(loaders, model, criterion, device):
    """
    Test the network using torch test loaders created from test data sets
    :param loaders : The test Data loader
    :param model : Network Model
    :device : Testing device, GPU if available
    """
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    model.to(device)
    model.eval()
    for batch_idx, (data, target) in enumerate(loaders):
        # move to GPU if available
        data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model.forward(data)
        # calculate the loss
        loss = criterion(output, target)
        # update average test loss
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        prediction = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(prediction.eq(target.data.view_as(prediction))).cpu().numpy())
        total += data.size(0)
    print('Test Data Loss: {:.6f}\n'.format(test_loss))
    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
