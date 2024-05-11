import torch.nn as nn


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(20 * 4 * 4, 50),
            nn.Linear(50, 10),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x

    def name(self, ):
        return self.__class__.__name__


if __name__ == '__main__':
    lenet = LeNet()
    print(lenet.name())
    print('over')
