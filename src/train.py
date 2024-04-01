import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import get_dataloader
from model import ImageClassifier
from evaluate import Evaluator

num_epochs = 160
batch_size = 128
initial_lr = 0.01
depth = 12

train_dataloader = get_dataloader("train", batch_size=batch_size)
valid_dataloader = get_dataloader("valid", batch_size=batch_size)

print(f"Train Num Batches: {len(train_dataloader)}")
print(f"Valid Num Batches: {len(valid_dataloader)}")
print(f"Iterations: {num_epochs * len(train_dataloader)}")

model = ImageClassifier(depth=depth)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)

evaluator = Evaluator(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs)

evaluator.output_stats(evaluator.eval_model(train_dataloader))

print("Starting Training...")

for epoch in range(num_epochs):
    print(f"---> Epoch {epoch + 1} / {num_epochs}, lr = {scheduler.get_lr()[0]:.2e}")

    _, train_loss, _ = train_result = evaluator.train_model()

    evaluator.output_stats(train_result)

    scheduler.step()

print("Finished Training!")
