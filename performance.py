import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from model import Net
import matplotlib.pyplot as plt

network = Net()

network_state_dict = torch.load('/Users/jessicayang/Workspace/Research/pytorch/results/model.pth')
network.load_state_dict(network_state_dict)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.1307,), (0.3081,))])
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                            shuffle=True, num_workers=2)

examples = enumerate(testloader)
batch_idx, (example_data, example_targets) = next(examples)

with torch.no_grad():
  output = network(example_data)

fig = plt.figure()
for i in range(4):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
  plt.title("Prediction: {}".format(
    output.data.max(1, keepdim=True)[1][i].item()))
  plt.xticks([])
  plt.yticks([])
  plt.show()
fig