import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.fc = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        x = self.fc(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = torch.tensor([[0.1]]).to(device)
Y = 2 * X - 0.1

model = Linear().to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(500):
    preds = model(X)
    loss = loss_fn(preds, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch == 99:
        print(epoch, loss.item())


print(model(torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9]]).to(device)))

print()
for i in model.parameters():
    print(i)