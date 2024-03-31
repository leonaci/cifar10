import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import get_dataloader
from model import ImageClassifier
from evaluate import Evaluator

num_epochs = 100
batch_size = 32
initial_lr = 0.01
depth = 6

train_dataloader = get_dataloader("train", batch_size=batch_size)
valid_dataloader = get_dataloader("valid", batch_size=batch_size)

model = ImageClassifier(depth=depth)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, eps=1e-24)

evaluator = Evaluator(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs)

evaluator.output_stats(evaluator.eval_model(train_dataloader))

print("Starting Training...")

for epoch in range(num_epochs):
    print(f"---> Epoch {epoch + 1}, lr = {optimizer.param_groups[0]['lr']:.2e}")

    _, train_loss, _ = train_result = evaluator.train_model()

    evaluator.output_stats(train_result)

    scheduler.step(train_loss)

print("Finished Training!")
