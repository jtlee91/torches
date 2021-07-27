import torch
import torch.nn as nn


class AN(nn.Module):
    def __init__(self):
        super(AN, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc(x)
        return x


device = torch.device("cuda")

model = AN().to(device)

X = torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]]).to(device)
Y = 2 * X + 0.1

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.25)

for _ in range(1000):
    preds = model(X)

    loss = loss_fn(preds, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(model(torch.tensor([[0.6], [0.7], [0.8], [0.9]]).to(device)))
