# TODO: Import your dependencies.
# For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import logging
import argparse
import sys
import os
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn.functional as F
import torch.utils.data.distributed

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    logger.info("starting testing ...")
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logger.info(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer, device, epochs=5):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    logger.info("starting training...")
    logger.info(device)
    for epoch in range(epochs):
        running_corrects = 0
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
#             logger.info(batch_idx)
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            loss = criterion(output, target)
            preds = output.argmax(dim=1, keepdim=True)
            running_loss += loss.item() * data.size(0)
            running_corrects += preds.eq(target.view_as(preds)).sum().item()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item()))

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        logger.info('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    return model


def net(num_classes=133):
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dataset_path = os.path.join(data, "Training")
    test_dataset_path = os.path.join(data, "Testing")

    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()])

    testing_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor()])

    train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=testing_transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_data_loader, test_data_loader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    logger.info("hany starting...")
    save_path = args.model_dir + "/resnet50.pth"
    model = net()
    model.to(device)
    '''
    TODO: Create your loss and optimizer
    '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''

    #     bucket_name = "sagemaker-us-east-1-436014510024"

    #     downloadDirectoryFroms3(bucket_name, "data")
    #     # region = "us-east-1"

    #     train_dir = "data/train"
    #     test_dir = "data/test"

    train_loader, test_loader = create_data_loaders(args.train_dir, batch_size)

    #     train_loader = create_data_loaders(train_data, batch_size,training_transform)

    #     test_loader = create_data_loaders(test_dir, batch_size,testing_transform)

    model = train(model, train_loader, criterion, optimizer, device)

    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion, device)

    '''
    TODO: Save the trained model
    '''
    torch.save(model, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )

    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    # parser.add_argument("--test_dir", type=str, default=os.environ["SM_CHANNEL_TESTING"])

    args = parser.parse_args()

    main(args)
