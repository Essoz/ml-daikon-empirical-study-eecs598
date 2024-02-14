
import torch
from itertools import repeat
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = nn.RNN(input_dim, output_dim,
                            batch_first=True,
                            dtype=torch.cfloat)
        self.loss = nn.L1Loss()  # Use SmoothL1Loss instead of L1Loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)  # Increase the learning rate

    def forward(self, x):
        x = torch.unsqueeze(x, 0)
        return self.model(x)[0]

    def fit(self, x, y):
        self.optimizer.zero_grad()
        z = torch.squeeze(self(x), 0)
        loss = self.loss(z, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
class RandomDataset(torch.utils.data.IterableDataset):
    def __init__(self):
        super(RandomDataset, self).__init__()

    def __iter__(self):
        return repeat(torch.rand(100, dtype=torch.cfloat))    

model = CRNN(100, 100)
dataset = RandomDataset()
loader = torch.utils.data.DataLoader(dataset, batch_size=30)

for i, data in enumerate(loader):
    loss = model.fit(data, data)
    print(i, loss)
    if i > 10000:
        break