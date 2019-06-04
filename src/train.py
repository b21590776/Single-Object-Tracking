import os
import time
from src import dataset
from src import model
import torch
from torch.autograd import Variable
from torchvision import transforms
from src.operations import ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D


use_gpu = torch.cuda.is_available()

# parameters
epochs = 5
batch_size = 64
learning_rate = 0.001
model_save_directory = '../save/pa3/'


def train(model, dataloader, criterion, optimizer, num_epochs, model_save_dir):
    since = time.time()
    dataset_size = dataloader.dataset.len
    train_l = []
    valid_l = []
    running_loss = 0.0
    i = 0
    x = list()
    for epoch in range(num_epochs):
        print('------------------EPOCH {}/{}------------------'.format(epoch+1, num_epochs))
        model.train()
        x.append(epoch + 1)
        # iterating over data
        for data in dataloader:
            # getting the inputs and labels
            x1, x2, y = data['previmg'], data['currimg'], data['currbb']
            # wrapping them in variable
            if use_gpu:
                x1, x2, y = Variable(x1.cuda()), Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
            else:
                x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            output = model(x1, x2)
            loss = criterion(output, y)
            # backward + optimize
            loss.backward()
            optimizer.step()
            print('training epoch : %d, step : %d, loss : %f' % (epoch+1, i, loss.data.item()))
            i = i + 1
            running_loss += loss.data.item()

        epoch_loss = running_loss / dataset_size
        train_l.append(epoch_loss)
        print('-------------Loss: {:.4f} in epoch: {}-------------'.format(epoch_loss, epoch+1))
        val_loss = validation(model, criterion, epoch+1)
        print('Validation Loss: {:.4f}'.format(val_loss))
        valid_l.append(val_loss)

    path = model_save_dir + 'model_n_epoch_' + str(num_epochs) + '.pth'
    torch.save(model.state_dict(), path)

    # plotting the loss graphics both for validation and training.
    plot_loss_table(x, train_l, valid_l)

    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model


def plot_loss_table(x, train_l, valid_l):

    fig = plt.figure(0)
    fig.canvas.set_window_title('Train MSE loss vs Validation MSE loss')
    plt.axis([0, epochs + 1, 0, 2])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(1.05, 5), loc=2, borderaxespad=0.)
    train_graph, = plt.plot(x, train_l, 'b--', label='Train MSE loss')
    plt.plot(x, valid_l, 'g', label='Validation MSE loss')
    plt.legend(handler_map={train_graph: HandlerLine2D(numpoints=3)})
    plt.show()


def validation(model, criterion, epoch):

    # evaluation mode
    model.eval()
    transform = transforms.Compose([Normalize(), ToTensor()])
    data = dataset.Datasets("../dataset/videos/val/", "../dataset/annotations/",transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)
    dataset_size = dataloader.dataset.len
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    running_loss = 0.0
    i = 0
    # iterate over data
    for data in dataloader:
        # getting the inputs and labels
        x1, x2, y = data['previmg'], data['currimg'], data['currbb']
        # wrapping them in Variable
        if use_gpu:
            x1, x2, y = Variable(x1.cuda()), Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
        else:
            x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        output = model(x1, x2)
        loss = criterion(output, y)
        # backward + optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.data.item()
        print('Validation epoch : %d, step : %d, loss : %f' % (epoch , i, loss.data.item()))
        i = i + 1

    val_loss = running_loss/dataset_size
    return val_loss


if __name__ == "__main__":
    # loading dataset
    transform = transforms.Compose([Normalize(), ToTensor()])
    data = dataset.Datasets("../dataset/videos/train/","../dataset/annotations/", transform)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=4)

    # loading model
    net = model.ConNet()
    # MSE loss
    loss_fn = torch.nn.MSELoss()
    if use_gpu:
        net = net.cuda()
        loss_fn = loss_fn.cuda()
    optimizer = optim.Adam(net.classifier.parameters(), lr=learning_rate)

    if os.path.exists(model_save_directory):
        print('Directory %s already exists' % (model_save_directory))
    else:
        os.makedirs(model_save_directory)

    # start training
    net = train(net, dataloader, loss_fn, optimizer, epochs, model_save_directory)
