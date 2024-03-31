from typing import Tuple
import torch
import matplotlib.pyplot as plt
import csv
import time

class Evaluator:
    def __init__(self, model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        
        self.train_loss_history = []
        self.train_acc_history = []
        self.valid_loss_history = []
        self.valid_acc_history = []
        self.min_valid_loss = float('inf')

        model.to(self.device)

    def train_model(self):
        self.model.train()

        dataloader = self.train_dataloader
        running_loss = 0.0; num_correct = 0.0; total = 0

        start_time = time.time()
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            num_correct += (predicted == labels).sum().item()
            total += labels.size(0)
        end_time = time.time()

        return 1000 * (end_time - start_time) / len(dataloader), running_loss / len(dataloader), 100 * num_correct  / total

    def eval_model(self, dataloader):
        self.model.eval()

        running_loss = 0.0; num_correct = 0.0; total = 0

        start_time = time.time()
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            num_correct += (predicted == labels).sum().item()
            total += labels.size(0)
        end_time = time.time()

        return 1000 * (end_time - start_time) / len(dataloader), running_loss / len(dataloader), 100 * num_correct / total

    def output_stats(self, train_result:Tuple[float, float, float]):
        self.model.eval()

        train_time, train_loss, train_acc = train_result
        eval_time, valid_loss, valid_acc = self.eval_model(self.valid_dataloader)

        print(f"    Train Time: {train_time:.2f}ms")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Train Accuracy: {train_acc:.2f}%")
        print(f"    Eval Time: {eval_time:.2f}ms")
        print(f"    Valid Loss: {valid_loss:.4f}")
        print(f"    Valid Accuracy: {valid_acc:.2f}%")

        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            try:
                torch.save(self.model.state_dict(), "../weights/model.pth")
            except IOError as e:
                print(f"saving model failed: {e}")

        self.train_loss_history.append(train_loss)
        self.train_acc_history.append(train_acc)
        self.valid_loss_history.append(valid_loss)
        self.valid_acc_history.append(valid_acc)

        self._save_graph()
        self._save_csv()

    def _save_graph(self):

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()

        train_loss_line = ax1.plot(self.train_loss_history, linewidth=2, color='#FDB813', label='Train Loss')
        valid_loss_line = ax1.plot(self.valid_loss_history, linewidth=2, color='#B2D732', label='Valid Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_xlim(1, self.num_epochs)
        ax1.set_ylim(ymax=1)
        ax1.grid(True)

        train_acc_line = ax2.plot(self.train_acc_history, linewidth=1, linestyle="dashed", color='#FD7F20', label='Train Accuracy')
        valid_acc_line = ax2.plot(self.valid_acc_history, linewidth=1, linestyle="dashed", color='#87CB16', label='Valid Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlim(1, self.num_epochs)
        ax2.set_ylim(0, 100)

        lines = train_loss_line + valid_loss_line + train_acc_line + valid_acc_line
        labels = map(lambda line: line.get_label(), lines)
        plt.legend(lines, labels, loc='lower left')

        plt.title('Loss and Accuracy')
        plt.savefig('../data/loss_and_accuracy.png')
        plt.close()

    def _save_csv(self):
        with open('../data/loss_and_accuracy.csv', 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            epochs = range(0, len(self.train_loss_history) + 1)
            for epoch, train_loss, valid_loss, train_acc, valid_acc in zip(epochs, self.train_loss_history, self.valid_loss_history, self.train_acc_history, self.valid_acc_history):
                writer.writerow({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'train_accuracy': train_acc,
                    'valid_accuracy': valid_acc,
                })
