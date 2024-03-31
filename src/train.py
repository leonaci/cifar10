import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import get_dataloader
from model import ImageClassifier
from evaluate import Evaluator
import time

num_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = get_dataloader("train", batch_size=32)
valid_dataloader = get_dataloader("valid", batch_size=32)

model = ImageClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

evaluator = Evaluator(model, train_dataloader, valid_dataloader, criterion, num_epochs)

evaluator.eval()

print("Starting Training...")

for epoch in range(num_epochs):
    model.train()
    print(f"---> Epoch {epoch + 1}, lr = {optimizer.param_groups[0]['lr']:.2e}")
    start_time = time.time()
    running_loss = 0.0
    train_acc = 0.0
    total = 0

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()
        total += labels.size(0)
    end_time = time.time()
    evaluator.eval((end_time - start_time, running_loss, train_acc / total))

print("Finished Training!")
