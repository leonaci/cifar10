import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import get_dataloader
from model import ImageClassifier, evaluate
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataloader = get_dataloader("train", batch_size=32)

model = ImageClassifier()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

loss_history = []

initial_loss = 0.0
for inputs, labels in dataloader:
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    initial_loss += loss.item()
loss_history.append(initial_loss)
accuracy = evaluate(model, dataloader)
print(f"Initial Loss: {initial_loss / len(dataloader):.4f}")
print(f"Initial Accuracy: {accuracy:.2f}%")

num_epochs = 100

print("Starting Training...")

for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_history.append(running_loss)
    accuracy = evaluate(model, dataloader)
    print(f"---> Epoch {epoch + 1}")
    print(f"        Loss: {running_loss / len(dataloader):.4f}")
    print(f"    Accuracy: {accuracy:.2f}%")
    torch.save(model.state_dict(), "../weights/model.pth")

print("Finished Training!")


plt.figure(figsize=(8, 6))
plt.plot(range(0, num_epochs+1), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.grid(True)
plt.savefig('../data/loss_graph.png')
plt.show()


