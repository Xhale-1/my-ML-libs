
from torch import nn

class net(nn.Module):
  def __init__(self):
    super().__init__()

    self.fc1 = nn.Linear(5, 100)
    self.fc2 = nn.Linear(100, 60)
    self.fc3 = nn.Linear(60, 30)
    self.fc4 = nn.Linear(30, 20)
    self.fc5 = nn.Linear(20, 10)
    self.fc6 = nn.Linear(10, 5)
    self.fc7 = nn.Linear(5, 3)
    self.fc8 = nn.Linear(3, 2)
    self.fc9 = nn.Linear(2, 1)

    self.act = nn.LeakyReLU()

  def forward(self,x):
    x = self.act(self.fc1(x))
    x = self.act(self.fc2(x))
    x = self.act(self.fc3(x))
    x = self.act(self.fc4(x))
    x = self.act(self.fc5(x))
    x = self.act(self.fc6(x))
    x = self.act(self.fc7(x))
    x = self.act(self.fc8(x))
    return self.fc9(x)


model = net().to(device)
#print(model(torch.tensor(data.iloc[:5,:5].to_numpy(), dtype=torch.float32).to(device)))
