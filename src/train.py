import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import get_dataloader
from model import ImageClassifier
from evaluate import evaluate
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = get_dataloader("train", batch_size=32)
valid_dataloader = get_dataloader("valid", batch_size=32)

model = ImageClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_history = []
train_acc_history = []
valid_loss_history = []
valid_acc_history = []

num_epochs = 50

train_loss, train_acc, valid_loss, valid_acc = evaluate(model, train_dataloader, valid_dataloader, criterion)
train_loss_history.append(train_loss)
train_acc_history.append(train_acc)
valid_loss_history.append(valid_loss)
valid_acc_history.append(valid_acc)

print("Starting Training...")

min_valid_loss = float('inf')

for epoch in range(num_epochs):
    print(f"---> Epoch {epoch + 1}")
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    train_loss, train_acc, valid_loss, valid_acc = evaluate(model, train_dataloader, valid_dataloader, criterion)
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    valid_loss_history.append(valid_loss)
    valid_acc_history.append(valid_acc)
    
    if valid_loss < min_valid_loss:
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), "../weights/model.pth")

print("Finished Training!")


plt.figure(figsize=(8, 6))
plt.ylim(0, 2.5)
plt.plot(train_loss_history, label='train')
plt.plot(valid_loss_history, label='valid')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.title('Loss')
plt.grid(True)
plt.savefig('../data/loss_graph.png')

plt.figure(figsize=(8, 6))
plt.ylim(0, 100)
plt.plot(train_acc_history, label='train')
plt.plot(valid_acc_history, label='valid')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('Accuracy')
plt.grid(True)
plt.savefig('../data/accuracy_graph.png')

#plt.show()
