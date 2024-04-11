import os
from . import DEVICE
from typing import Tuple
import torch
import matplotlib.pyplot as plt
import csv
import time
import math

class Evaluator:
    def __init__(self, model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs, config):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_epochs = num_epochs

        csv_path = [config.csv_path]
        plot_path = [config.plot_path]

        if config.suffix != "":
            csv_path.append(config.suffix)
            plot_path.append(config.suffix)

        csv_path = "-".join(csv_path) + ".csv"
        plot_path = "-".join(plot_path) + ".png"

        if not os.path.exists(config.data_dir):
            os.makedirs(config.data_dir)

        self.csv_path = os.path.join(config.data_dir, csv_path)
        self.plot_path = os.path.join(config.data_dir, plot_path)

        self.weight_path = None
        if config.weight_path is not None:
            weight_path = [config.weight_path]

            if config.suffix != "":
                weight_path.append(config.suffix)
            
            weight_path = "-".join(weight_path)

            if not os.path.exists(config.weight_dir):
                os.makedirs(config.weight_dir)

            self.weight_path = os.path.join(config.weight_dir, weight_path)

        self.train_loss_history = []
        self.train_err_history = []
        self.valid_loss_history = []
        self.valid_err_history = []
        self.min_valid_loss = float('inf')
        self.min_valid_err = 100

        model.to(DEVICE)

    def train_model(self):
        self.model.train()

        dataloader = self.train_dataloader
        running_loss = 0.0; num_incorrect = 0; total = 0

        start_time = time.time()
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

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
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

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

        self.output_model(valid_loss, valid_err)

        self.train_loss_history.append(train_loss)
        self.train_err_history.append(train_err)
        self.valid_loss_history.append(valid_loss)
        self.valid_err_history.append(valid_err)

        self._save_graph()
        self._save_csv()

    def output_model(self, valid_loss, valid_err):
        if self.weight_path is not None and valid_err < self.min_valid_err:
            self.min_valid_loss = valid_loss
            self.min_valid_err = valid_err
            try:
                dummy_input = torch.randn(128, 3, 32, 32).to(DEVICE)
                torch.onnx.export(self.model, dummy_input, self.weight_path + ".onnx")
                torch.save(self.model.state_dict(), self.weight_path + ".pth")
            except IOError as e:
                print(f"saving model failed: {e}")

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

def display_images(images, output_dir, max_columns=10, max_rows=10):
    num_images = len(images)
    num_plots = math.ceil(num_images / (max_columns * max_rows))

    os.system(f"rm -rf {output_dir}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for plot_idx in range(num_plots):
        start_idx = plot_idx * max_columns * max_rows
        end_idx = min((plot_idx + 1) * max_columns * max_rows, num_images)
        plot_images = images[start_idx:end_idx]

        num_plot_images = len(plot_images)
        num_rows = (num_plot_images + max_columns - 1) // max_columns
        num_columns = min(num_plot_images, max_columns)

        _, axes = plt.subplots(num_rows, num_columns, figsize=(3*num_columns, 3*num_rows))

        for i, (img, label) in enumerate(plot_images):
            row = i // num_columns
            col = i % num_columns
            ax = axes[row, col] if num_rows > 1 else axes[col]

            # PIL Imageを直接表示
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(label)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'overview_{plot_idx}.png'))

        print(f"Saved images to {output_dir}/overview_{plot_idx}.png")

        plt.close()

    print(f"Saved {num_images} images to {output_dir}")
