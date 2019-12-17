from model import Net
from train import train
from test import test


import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)


def main():      
    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=False, num_workers=2)

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate,
                      momentum=momentum)

    #what we have
    examples = enumerate(testloader)
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(4):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
        plt.show()
    fig


    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(trainloader.dataset) for i in range(n_epochs + 1)]
    
 
    for epoch in range(1, n_epochs + 1):
        train(trainloader, network, optimizer, n_epochs, log_interval, train_losses, train_counter)
        test(network, test_losses, testloader)

if __name__ == '__main__':
    main()
    