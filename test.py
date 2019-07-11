from pathlib import Path
import gzip
import pickle
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

dev = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    # TODO
    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def preprocess(x, y):
    # use GPU
    return x.view(-1, 1, 28, 28).to(dev), y.to(dev)
    # use CPU
    # return x.view(-1, 1, 28, 28), y


def get_data(x_train, y_train, x_valid, y_valid, x_test, y_test, bs):
    return (
        DataLoader(TensorDataset(x_train, y_train), batch_size=bs, shuffle=True, num_workers=8),
        DataLoader(TensorDataset(x_valid, y_valid), batch_size=bs * 2, num_workers=8),
        DataLoader(TensorDataset(x_test, y_test), batch_size=bs * 2, num_workers=8)
    )


def loss_batch(model, loss_func, x_bs, y_bs, opt=None):
    pred = model(x_bs)
    loss_bs = loss_func(pred, y_bs)
    acc = accuracy(pred, y_bs)
    assert (not opt == True)
    if opt is not None:
        loss_bs.backward()
        opt.step()
        opt.zero_grad()

    return loss_bs.item(), len(x_bs), acc  # TODO


def accuracy(pred, yb):
    preds = torch.argmax(pred, dim=1)
    return (preds == yb).float().mean()


def fit(model, epochs, loss_func, opt, train_dl, valid_dl, test_dl, print_freq=500):
    plt.figure()
    train_acc_history = []
    valid_acc_history = []
    train_loss_history = []
    valid_loss_history = []
    for epoch in range(epochs):
        epoch_tic = time.time()
        model.train()
        batch_num = 0
        total_batch_num = len(train_dl)
        for x_bs, y_bs in train_dl:
            batch_num += 1
            batch_tic = time.time()
            loss, _, acc = loss_batch(model, loss_func, x_bs, y_bs, opt)
            batch_toc = time.time()
            # if batch_num % print_freq == 0:
            # print('Epoch [{}][{}/{}] Time: {}s Loss: {} Acc: {}'.format(epoch, batch_num, total_batch_num,
            #                                                             (batch_toc - batch_tic), loss, acc))

        epoch_toc = time.time()

        model.eval()
        with torch.no_grad():
            # TODO zip and *
            losses, nums, acc = zip(
                *[loss_batch(model, loss_func, x_valid_bs, y_valid_bs) for x_valid_bs, y_valid_bs in valid_dl]
            )
            train_losses, train_nums, train_acc = zip(
                *[loss_batch(model, loss_func, x_valid_bs, y_valid_bs) for x_valid_bs, y_valid_bs in train_dl]
            )
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        train_acc = np.sum(np.multiply(train_acc, train_nums)) / np.sum(train_nums)

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        val_acc = np.sum(np.multiply(acc, nums)) / np.sum(nums)

        train_acc_history.append(train_acc)
        valid_acc_history.append(val_acc)

        train_loss_history.append(train_loss)
        valid_loss_history.append(val_loss)

        print(
            'Epoch [{}] Train-loss: {}\tvalidation-loss: {}\tTrain-acc: {}\tValidation-acc: {}\ttrain-time: {} s\tvalidation-time: {} s'.format(
                epoch, train_loss, val_loss, train_acc, val_acc,
                epoch_toc - epoch_tic, time.time() - epoch_toc))

        torch.save(model.state_dict(), 'weight/model-%03d.pth' % epoch)

    plt.plot(range(len(train_acc_history)), train_acc_history, color='red')
    plt.plot(range(len(valid_acc_history)), valid_acc_history, color='orange')
    plt.show()


class Lambda(nn.Module):

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        a = self.func(x)
        # print(a.size())
        return a


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


def main():
    DATA_PATH = Path('data')
    PATH = DATA_PATH / 'mnist'
    FILENAME = 'mnist.pkl.gz'

    with gzip.open((PATH / FILENAME).as_posix(), 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding='latin-1')

    x_train, y_train, x_valid, y_valid, x_test, y_test = map(torch.tensor,
                                                             (x_train, y_train, x_valid, y_valid, x_test, y_test))

    train_dl, valid_dl, test_dl = get_data(x_train, y_train, x_valid, y_valid, x_test, y_test, 250)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    valid_dl = WrappedDataLoader(valid_dl, preprocess)
    test_dl = WrappedDataLoader(test_dl, preprocess)

    model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 1024, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(1024),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(1024,512, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Conv2d(512, 10, kernel_size=3, stride=2, padding=1),
        nn.AdaptiveAvgPool2d(1),  # TODO
        Lambda(lambda x: x.view(x.size(0), -1)),  # TODO
        # nn.Linear(256, 1024),
        # nn.BatchNorm2d(1024),
        # nn.ReLU(),
        # nn.Linear(1024, 256),
        # nn.BatchNorm2d(256),
        # nn.ReLU(),
        # nn.Linear(256, 64),
        # nn.BatchNorm2d(64),
        # nn.ReLU(),
        # nn.Linear(64, 10),
        nn.Softmax(1)
    )
    # model = AlexNet()
    # use Multi-GPUs
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).cuda()
    # print(next(model.parameters()).is_cuda)

    # TODO
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    fit(model, 300, F.cross_entropy, opt, train_dl, valid_dl, test_dl)


if __name__ == '__main__':
    main()
    # TODO one GPU
    # TODO multi-GPUs
    # TODO save and load models
    # TODO compare w/ BatchNorm with w/o BatchNorm
    # TODO Refactor our architecture
