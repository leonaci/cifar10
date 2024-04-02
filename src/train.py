import torch
import torch.nn as nn
import torch.optim as optim
from load_dataset import get_dataloader
from model import ImageClassifier
from evaluate import Evaluator
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--num-epochs", type=int, default=500)
argparser.add_argument("--batch-size", type=int, default=256)
argparser.add_argument("--initial-lr", type=float, default=0.01)
argparser.add_argument("--num-layers", type=str, default="")
argparser.add_argument("--data-dir", type=str, default="")
argparser.add_argument("--csv-path", type=str, default="loss_and_error")
argparser.add_argument("--plot-path", type=str, default="loss_and_error")
argparser.add_argument("--weight-dir", type=str, default="")
argparser.add_argument("--weight-path", type=str, default="model")
argparser.add_argument("--suffix", type=str, default="")
argparser.add_argument("--output-weight", type=bool, default=False)

args = argparser.parse_args()

num_layers = list(map(int, args.num_layers.split(",")))

num_epochs = args.num_epochs
batch_size = args.batch_size
initial_lr = args.initial_lr

train_dataloader = get_dataloader("train", batch_size=batch_size)
valid_dataloader = get_dataloader("valid", batch_size=batch_size)

print(f"Train Num Batches: {len(train_dataloader)}")
print(f"Valid Num Batches: {len(valid_dataloader)}")
print(f"Iterations: {num_epochs * len(train_dataloader)}")

model = ImageClassifier(num_layers=num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=initial_lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[num_epochs//2, num_epochs//4*3], gamma=0.1)

evaluator = Evaluator(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, args)

evaluator.output_stats(evaluator.eval_model(train_dataloader))

print("Starting Training...")

for epoch in range(num_epochs):
    print(f"---> Epoch {epoch + 1} / {num_epochs}, lr = {scheduler.get_lr()[0]:.2e}")

    _, train_loss, _ = train_result = evaluator.train_model()

    evaluator.output_stats(train_result)

    scheduler.step()

print("Finished Training!")
