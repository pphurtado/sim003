import torch
import torch.nn as nn

def train_perceptron(x, F, lr=0.01, epochs=500):
    model = nn.Linear(1, 1, bias=False)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    loss_history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, F)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
    return model, loss_history
