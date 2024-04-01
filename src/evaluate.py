from typing import Tuple
import torch
import matplotlib.pyplot as plt
import csv
import time

class Evaluator:
    def __init__(self, model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

        suffix = "" if args.suffix == '' else f"-{args.suffix}"
        self.csv_path = f"../data/{args.csv_path}{suffix}.csv"
        self.plot_path = f"../data/{args.plot_path}{suffix}.png"
        self.weight_path = f"../weights/{args.weight_path}{suffix}.pth"

        self.train_loss_history = []
        self.train_err_history = []
        self.valid_loss_history = []
        self.valid_err_history = []
        self.min_valid_loss = float('inf')

        model.to(self.device)

    def train_model(self):
        self.model.train()

        dataloader = self.train_dataloader
        running_loss = 0.0; num_incorrect = 0; total = 0

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
            num_incorrect += (predicted != labels).sum().item()
            total += labels.size(0)
        end_time = time.time()

        return 1000 * (end_time - start_time) / len(dataloader), running_loss / len(dataloader), 100 * num_incorrect  / total

    def eval_model(self, dataloader):
        self.model.eval()

        running_loss = 0.0; num_incorrect = 0; total = 0

        start_time = time.time()
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            num_incorrect += (predicted != labels).sum().item()
            total += labels.size(0)
        end_time = time.time()

        return 1000 * (end_time - start_time) / len(dataloader), running_loss / len(dataloader), 100 * num_incorrect / total

    def output_stats(self, train_result:Tuple[float, float, float]):
        self.model.eval()

        train_time, train_loss, train_err = train_result
        eval_time, valid_loss, valid_err = self.eval_model(self.valid_dataloader)

        print(f"    Train Time: {train_time:.2f}ms")
        print(f"    Train Loss: {train_loss:.4f}")
        print(f"    Train Error: {train_err:.2f}%")
        print(f"    Eval Time: {eval_time:.2f}ms")
        print(f"    Valid Loss: {valid_loss:.4f}")
        print(f"    Valid Error: {valid_err:.2f}%")

        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            try:
                torch.save(self.model.state_dict(), "../weights/model.pth")
            except IOError as e:
                print(f"saving model failed: {e}")

        self.train_loss_history.append(train_loss)
        self.train_err_history.append(train_err)
        self.valid_loss_history.append(valid_loss)
        self.valid_err_history.append(valid_err)

        self._save_graph()
        self._save_csv()

    def _save_graph(self):
        num_batches = len(self.train_dataloader)

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()

        train_loss_line = ax1.plot(_lerp_data(self.train_loss_history, num_batches), linewidth=2, alpha=0.5, color='#FDB813', label='Train Loss')
        valid_loss_line = ax1.plot(_lerp_data(self.valid_loss_history, num_batches), linewidth=2, alpha=0.5, color='#B2D732', label='Valid Loss')
        ax1.set_xlabel('Iterations (10^4)')
        ax1.set_ylabel('Loss')
        ax1.set_xlim(0, self.num_epochs * num_batches)
        ax1.set_ylim(0, 1)
        # ax1.set_ylim(ymax=1)
        # ax1.set_yscale('log')

        xticks = ax1.get_xticks()
        xtick_labels = map(lambda tick : int(tick / 1000) / 10, xticks)
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xtick_labels)

        train_err_line = ax2.plot(_lerp_data(self.train_err_history, num_batches), linewidth=1, linestyle="dashed", color='#FD7F20', label='Train Error')
        valid_err_line = ax2.plot(_lerp_data(self.valid_err_history, num_batches), linewidth=1, linestyle="dashed", color='#87CB16', label='Valid Error')
        ax2.set_ylabel('Error')
        ax2.set_xlim(0, self.num_epochs * num_batches)
        ax2.set_ylim(0, 50)
        ax2.grid(True)

        lines = train_loss_line + valid_loss_line + train_err_line + valid_err_line
        labels = map(lambda line: line.get_label(), lines)
        plt.legend(lines, labels, loc='lower left')

        plt.title('Loss and Error')
        plt.savefig(self.plot_path)
        plt.close()

    def _save_csv(self):
        with open(self.csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'train_loss', 'valid_loss', 'train_error', 'valid_error']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            epochs = range(0, len(self.train_loss_history) + 1)
            for epoch, train_loss, valid_loss, train_err, valid_err in zip(epochs, self.train_loss_history, self.valid_loss_history, self.train_err_history, self.valid_err_history):
                writer.writerow({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'train_error': train_err,
                    'valid_error': valid_err,
                })

def _lerp_data(data, N):
    length = len(data)
    lerp_data = [0] * (N * (length - 1) + 1)

    for i in range(length - 1):
        start = data[i]
        end = data[i + 1]
        step = (end - start) / N

        for j in range(N):
            lerp_data[i * N + j] = start + j * step

    lerp_data[-1] = data[-1]

    return lerp_data
