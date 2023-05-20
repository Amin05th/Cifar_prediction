import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# preprocessed data
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# loaded data
train_dataset = datasets.CIFAR10("data", train=True, download=True, transform=transform)
train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)


test_dataset = datasets.CIFAR10("data", train=False, download=True, transform=transform)
test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(1024, 500)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


model = Network()
model.cuda()

torch.save(model, "cifar_network.pt")

if os.path.isfile("cifar_network.pt"):
    model = torch.load("cifar_network.pt")
    model.cuda()


optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    model.eval()
    for batch_id, (data, target) in enumerate(train_dataset):
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        out = model(data)
        criterion = F.cross_entropy
        loss = criterion(out, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f'Train Epoch: {epoch} [{batch_id * len(data)}/{len(train_dataset.dataset)} ({100. * batch_id / len(train_dataset):.0f}%)]\tLoss: {loss.item():.6f}')

        return out, loss


def test():
    model.eval()
    loss = 0
    correct = 0
    for data, target in test_dataset:
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        out = model(data)
        loss += F.cross_entropy(out, target, reduction="sum").item()
        prediction = out.data.max(1, keepdim=True)[1]
        reshaped_target = target.data.view_as(prediction)
        correct += prediction.eq(reshaped_target).sum().item()
    loss += loss / len(test_dataset.dataset)
    print("Mean-Loss: ", loss)
    print("Loss in percent: ", 100. / len(test_dataset.dataset) * correct, "%")


sum = 0
avg = []
for epoch in range(1, 15):
    out, loss = train(epoch)
    test()
    sum = sum + loss.data.item()

    if epoch % 100 == 0:
        avg.append(sum / 100)
        sum = 0
        print(epoch / 100, "% done.")

plt.figure()
plt.plot(avg)
plt.show()
