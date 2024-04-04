import torch.nn as nn
import torch.optim as optim
from .load_dataset import get_dataloader
from .model import ImageClassifier
from .evaluate import Evaluator
from .config import load_config
import argparse

def main(args):
    if args.config is None:
        raise ValueError("Config file must be provided.")

    config = load_config(args.config)

    num_epochs = config.num_epochs

    train_dataloader = get_dataloader("train", batch_size=config.batch_size)
    valid_dataloader = get_dataloader("valid", batch_size=config.batch_size)

    print(f"Train Num Batches: {len(train_dataloader)}")
    print(f"Valid Num Batches: {len(valid_dataloader)}")
    print(f"Iterations: {num_epochs * len(train_dataloader)}")

    model = ImageClassifier(config)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.initial_lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=map(lambda x: x * num_epochs, config.milestones), gamma=0.1)

    evaluator = Evaluator(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, config)

    evaluator.output_stats(evaluator.eval_model(train_dataloader))

    print("Starting Training...")

    for epoch in range(num_epochs):
        if epoch == 20 and (train_loss > 2.0 or train_error > 80):
            print("Early stopping...")
            break

        print(f"---> Epoch {epoch + 1} / {num_epochs}, lr = {scheduler.get_lr()[0]:.2e}")

        _, train_loss, _ = train_result = evaluator.train_model()

        evaluator.output_stats(train_result)

        scheduler.step()

    print("Finished Training!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    main(args)
