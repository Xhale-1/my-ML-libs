
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


#model = net().to(device)
#print(model(torch.tensor(data.iloc[:5,:5].to_numpy(), dtype=torch.float32).to(device)))





# по результатам тоже самое в среднем
class net2(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc0 = nn.Linear(4,30)
    self.fc01 = nn.Linear(30,15)
    self.fc02 = nn.Linear(15,6)
    self.fc03 = nn.Linear(6,3)
    self.fc04 = nn.Linear(3,1)

    self.fc1 = nn.Linear(2, 100)
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
    emb = self.act(self.fc0(x[:,:4]))
    emb = self.act(self.fc01(emb))
    emb = self.act(self.fc02(emb))
    emb = self.act(self.fc03(emb))
    emb = self.act(self.fc04(emb))


    x_t = torch.cat((emb,x[:,4].reshape(-1,1)), axis = 1)
    x = self.act(self.fc1(x_t))
    x = self.act(self.fc2(x))
    x = self.act(self.fc3(x))
    x = self.act(self.fc4(x))
    x = self.act(self.fc5(x))
    x = self.act(self.fc6(x))
    x = self.act(self.fc7(x))
    x = self.act(self.fc8(x))
    return self.fc9(x)


    


# веса есть более менее хорошие, но повторно обучить не получается + долгое обучение
from torch import nn

class net(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 100)
        self.bn01 = nn.BatchNorm1d(100, momentum = 0.055)      # После fc1

        self.fc2 = nn.Linear(100, 60)
        self.bn12 = nn.BatchNorm1d(60, momentum = 0.055)       # После fc2

        self.fc3 = nn.Linear(60, 30)
        self.bn23 = nn.BatchNorm1d(30, momentum = 0.055)       # После fc3

        self.fc4 = nn.Linear(30, 20)
        self.bn34 = nn.BatchNorm1d(20, momentum = 0.055)       # После fc4

        self.fc5 = nn.Linear(20, 10)
        self.bn45 = nn.BatchNorm1d(10, momentum = 0.055)       # После fc5

        self.fc6 = nn.Linear(10, 5)
        self.bn56 = nn.BatchNorm1d(5, momentum = 0.055)        # После fc6

        self.fc7 = nn.Linear(5, 3)
        self.bn67 = nn.BatchNorm1d(3, momentum = 0.055)        # После fc7

        self.fc8 = nn.Linear(3, 2)
        self.bn78 = nn.BatchNorm1d(2, momentum = 0.055)        # После fc8

        self.fc9 = nn.Linear(2, 1)           # Без BN после fc9

        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn01(x)         # BN после fc1
        x = self.act(x)

        x = self.fc2(x)
        x = self.bn12(x)         # BN после fc2
        x = self.act(x)

        x = self.fc3(x)
        x = self.bn23(x)         # BN после fc3
        x = self.act(x)

        x = self.fc4(x)
        x = self.bn34(x)         # BN после fc4
        x = self.act(x)

        x = self.fc5(x)
        x = self.bn45(x)         # BN после fc5
        x = self.act(x)

        x = self.fc6(x)
        x = self.bn56(x)         # BN после fc6
        x = self.act(x)

        x = self.fc7(x)
        x = self.bn67(x)         # BN после fc7
        x = self.act(x)

        x = self.fc8(x)
        x = self.bn78(x)         # BN после fc8
        x = self.act(x)

        x = self.fc9(x)          # Без BN после fc9
        return x

model = net().to(device)

