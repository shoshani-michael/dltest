import numpy
import pandas as pd
import numpy as np
import torch
from torch import nn
import scipy.linalg
from torch.nn import ModuleList
from torch.utils.data import DataLoader, TensorDataset, random_split
# from torch.utils.data.datapipes import dataframe
from torchvision import datasets
from torchvision.datasets.utils import download_url
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from sklearn.svm import SVC

import pickle
import os
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from google.colab import files
import itertools

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")
N = 2
BATCH_SIZE = 64

DEFAULT_NUM_OF_EPOCHS = 20

#DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
DATA_FILENAME = "insurance.csv"
#download_url(DATASET_URL, '.')

dataframe_raw = pd.read_csv(DATA_FILENAME)
dataframe_raw.head()
input_cols = list(dataframe_raw.drop('charges', axis=1).columns)
categorical_cols = list(dataframe_raw.select_dtypes(include='object').columns)
output_cols = [dataframe_raw.columns[-1]]


def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    # inputs_array = np.array(dataframe1[input_cols])
    inputs_array = dataframe1.drop('charges', axis=1).values
    # targets_array = np.array(dataframe1[output_cols])
    targets_array = dataframe1[['charges']].values
    return inputs_array, targets_array


inputs_array, targets_array = dataframe_to_arrays(dataframe_raw)
inputs = torch.from_numpy(inputs_array).to(torch.float32)
targets = torch.from_numpy(targets_array).to(torch.float32)

dataset = TensorDataset(inputs, targets)

val_percent = 0.1  # between 0.1 and 0.2
num_rows = len(dataframe_raw)
val_size = int(num_rows * val_percent)
print(val_size)
train_size = num_rows - val_size
print(train_size)

input_size = len(input_cols)
print(input_size)
output_size = len(output_cols)
print(output_size)


TRAIN_SIZE = train_size
TEST_SIZE = val_size

# training_data = datasets.CIFAR10(
#     root="data",
#     train=True,
#     download=True,
#     transform=ToTensor(),
# )
#
# # Download test data from open datasets.
# test_data = datasets.CIFAR10(
#     root="data",
#     train=False,
#     download=True,
#     transform=ToTensor(),
# )


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        linears = nn.ModuleList([nn.Linear(input_size, 30)])
        for i in range(max(N-2 , 0)):
            linears.extend([nn.Linear(30, 30)])
        linears.extend([nn.Linear(30, output_size)])
        self.linear_stack = linears

    def forward(self, x):
        for layer in self.linear_stack:
            x = layer(x)
        return x


def init_weights_wrapper(deviation):
    def init_weights(m):
        if (isinstance(m, nn.Linear)) or (isinstance(m, nn.Conv2d)):
            torch.nn.init.normal_(m.weight, std=deviation)
            # torch.nn.init.normal_(m.bias , std=deviation)
            m.bias.data.fill_(0)

    return init_weights


# Train the model for a single epoch
# Returns accuracy and loss
def epoch_train(dataloader, model, loss_fn, optimizer, num_of_batches=-1, verbose=False):
    model.train()
    print(model)
    for param in model.parameters():
        print(param)
    # Calculate the size of data we run on
    if num_of_batches > 0:
        size = num_of_batches * BATCH_SIZE
    else:
        size = TRAIN_SIZE

    # loss and accuracy of the entire epoch
    epoch_loss, epoch_accuracy = 0, 0
    for batch, (X, y) in enumerate(dataloader):

        e2e = np.dot(model.linear_stack[1].weight.cpu().detach().numpy(), model.linear_stack[0].weight.cpu().detach().numpy())
        if N == 3:
            e2e = np.dot(model.linear_stack[2].weight, e2e)
        norm = np.linalg.norm(e2e)
        # Dont run more than num_of_batches batches (-1 means to run all)
        if batch == num_of_batches:
            break
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        print(pred)
        loss = loss_fn(pred, y)
        print(loss)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch % 100 == 0) and verbose:
            current = (batch + 1) * len(X)
            print(f"Train loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
        epoch_loss += loss.item()
        epoch_accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()

    # Calculate the total loss and accuracy
    if num_of_batches > 0:
        loss = epoch_loss / num_of_batches
    else:
        loss = epoch_loss / train_size
    accuracy = epoch_accuracy / size
    print("loss in epoch: " + str(loss))
    return norm, loss


def test(dataloader, model, loss_fn, num_of_batches=-1, verbose=False):
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            # Dont run more than num_of_batches batches (-1 means to run all)
            if batch == num_of_batches:
                break
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    # Calculate the total loss and accuracy
    if num_of_batches > 0:
        size = num_of_batches * BATCH_SIZE
        test_loss = test_loss / num_of_batches
    else:
        size = TEST_SIZE  # len(dataloader.dataset)
        test_loss = test_loss / val_size
    accuracy = correct / size
    if verbose:
        print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return accuracy, test_loss


# Run train and test for num_of_epochs
def train_and_test(train_dataloader, test_dataloader, model, loss_fn, optimizer,
                   num_of_train_batches=-1, num_of_test_batches=-1, num_of_epochs=DEFAULT_NUM_OF_EPOCHS, verbose=False):
    all_loss, all_accuracy, all_test_accuracy, all_test_loss = [], [], [], []
    for i in range(num_of_epochs):
        epoch_accuracy, epoch_loss = epoch_train(train_dataloader, model, loss_fn,
                                                 optimizer, verbose=verbose)
        test_accuracy, test_loss = test(test_dataloader, model, loss_fn,
                                        verbose=verbose)
        all_test_accuracy.append(test_accuracy)
        all_test_loss.append(test_loss)
        all_accuracy.append(epoch_accuracy)
        all_loss.append(epoch_loss)
    return all_accuracy, all_loss, all_test_accuracy, all_test_loss


# Plot train and test accuracy on the same graph,
# and train and test losses on the same graph.
def plot_graphs(num_of_epochs, train_accuracy, train_loss, test_accuracy, test_loss, suptitle):
    epochs = [(x + 1) for x in range(num_of_epochs)]
    fig, axs = plt.subplots(2)
    fig.suptitle(suptitle)
    fig.tight_layout()  # pad=3)
    # Plot
    # axs[0].set_title("Accuracy as func of epochs")
    axs[0].plot(epochs, train_accuracy)
    axs[0].plot(epochs, test_accuracy)
    axs[0].set(xlabel='epochs', ylabel='accuracy')
    axs[0].grid()
    axs[0].legend(["train", "test"], loc="best")

    # axs[1].set_title("Loss as func of epochs")
    axs[1].plot(epochs, train_loss)
    axs[1].plot(epochs, test_loss)
    axs[1].set(xlabel='epochs', ylabel='loss')
    axs[1].grid()
    axs[1].legend(["train", "test"], loc="best")

    plt.show()


def pretty_print_results(train_accuracy, train_loss, test_accuracy, test_loss):
    # Print the results at the end of optimization
    print(f"Train Error: \n Accuracy: {(100 * train_accuracy[-1]):>0.1f}%, Avg loss: {train_loss[-1]:>8f} \n")
    print(f"Test Error: \n Accuracy: {(100 * test_accuracy[-1]):>0.1f}%, Avg loss: {test_loss[-1]:>8f} \n")



def run_baseline(config, train_dataloader, test_dataloader, num_of_epochs=DEFAULT_NUM_OF_EPOCHS, verbose=False):
    # Train the network with the best parameters and then test it
    model = NeuralNetwork().to(device)
    model.apply(init_weights_wrapper(config["deviation"]))

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config["step_size"], momentum=config["momentum"])
    train_accuracy, train_loss, test_accuracy, test_loss = \
        train_and_test(train_dataloader, test_dataloader, model, loss_fn, optimizer, num_of_epochs=num_of_epochs,
                       verbose=verbose)

    plot_graphs(num_of_epochs, train_accuracy, train_loss, test_accuracy, test_loss, "2.1 baseline")

    print("Baseline results: ")
    pretty_print_results(train_accuracy, train_loss, test_accuracy, test_loss)


def gf_pred(X, e2e):
    temp = []
    for x in X:
        temp.append(np.dot(e2e.detach().numpy() , x))
    res = torch.tensor(np.array(temp) , requires_grad=True)
    print(res)
    return res

def get_fractional_power(t , p):
    evals, evecs = torch.linalg.eig(t)  # get eigendecomposition
    print(evals)
    evals = [n for n in evals if n.isreal][0]  # get real part of (real) eigenvalues

    # rebuild original matrix
    mchk = torch.matmul(evecs, torch.matmul(torch.diag(evals), torch.inverse(evecs)))

    print(mchk - t)  # check decomposition

    evpow = evals ** (p % 1)  # raise eigenvalues to fractional power

    # build exponentiated matrix from exponentiated eigenvalues
    mpow = torch.matmul(evecs, torch.matmul(torch.diag(evpow), torch.inverse(evecs)))
    return torch.matmul(torch.pow(t , int(p)), mpow)
    #return torch.tensor(scipy.linalg.fractional_matrix_power(t ,p))

def epoch_gf(dataloader, e2e , loss_fn , lr):
    for batch, (X, y) in enumerate(dataloader):
        #pred = gf_pred(X , e2e)
        #loss = loss_fn(pred , y)
        loss = torch.sum(torch.stack([(e2e*X[i] - y[i])**2 for i in range(len(X))]))/len(X)
        loss.backward()
        #epoch_grad = e2e.grad
        temp1 = torch.mul(e2e, torch.transpose(e2e, 0 , 1))
        temp2 = torch.mul(torch.transpose(e2e, 0 , 1), e2e)
        sum = torch.tensor([0,0,0,0,0,0])
        for i in range(1,N+1):
            temp11 = torch.mul(get_fractional_power(temp1 , i/N) , e2e.grad)
            e2e.retain_grad()
            sum = torch.add(sum , torch.mul(temp11 , get_fractional_power(temp2 , (N-i)/N)))
        e2e = torch.subtract(e2e , lr * sum)
        e2e.requires_grad_()
        print("hey")
    return loss.item , torch.norm(e2e)


def discreet_gradient_flow(train_dataloader, lr , num_of_epochs=DEFAULT_NUM_OF_EPOCHS, verbose=False):
    all_loss, all_norm = [], []
    loss_fn = nn.MSELoss()
    model = NeuralNetwork().to(device)
    model.apply(init_weights_wrapper(lr))
    e2e = np.dot(model.linear_stack[1].weight.cpu().detach().numpy(),
                 model.linear_stack[0].weight.cpu().detach().numpy())
    if N == 3:
        e2e = np.dot(model.linear_stack[2].weight, e2e)
    e2e = torch.tensor(e2e , requires_grad=True)
    for i in range(num_of_epochs):
        train_loss, all_norm = epoch_gf(train_dataloader, e2e, loss_fn , lr)
        all_loss.append(train_loss)
        all_norm.append(all_norm)
    plot_graphs(num_of_epochs , all_loss, all_norm , all_loss, all_norm)


# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    # Use the random_split function to split dataset into 2 parts of the desired length
    # Loaders for NN
    num_workers = 1
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=num_workers)
    test_dataloader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=num_workers)

    best_config = {"step_size": 0.0000001, "momentum": 0.8, "deviation": 0.0001}

    #run_baseline(best_config, train_dataloader, test_dataloader)
    discreet_gradient_flow(train_dataloader, best_config["step_size"])
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
