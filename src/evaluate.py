from typing import Tuple
import torch
import matplotlib.pyplot as plt

class Evaluator:
    def __init__(self, model, train_dataloader, valid_dataloader, criterion, num_epochs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.num_epochs = num_epochs
        
        self.train_loss_history = []
        self.train_acc_history = []
        self.valid_loss_history = []
        self.valid_acc_history = []
        self.min_valid_loss = float('inf')

    def eval(self, train_result:Tuple[float, float, float] = None):
        self.model.eval()

        train_loss = 0.0
        train_acc = 0.0

        if train_result is not None:
            train_time = train_result[0]
            train_loss = train_result[1] / len(self.train_dataloader)
            train_acc = 100 * train_result[2]
            print(f"    Train Time: {train_time:.2f}s")
            print(f"    Train Loss: {train_loss:.4f}")
            print(f"    Train Accuracy: {train_acc:.2f}%")

        valid_loss:float = 0.0
        valid_acc:float = 0.0
        total:int = 0

        for inputs, labels in self.valid_dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(inputs)
            valid_loss += self.criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            valid_acc += (predicted == labels).sum().item()
            total += labels.size(0)
        valid_loss = valid_loss / len(self.valid_dataloader)
        valid_acc = 100 * valid_acc / total
        print(f"    Valid Loss: {valid_loss:.4f}")
        print(f"    Valid Accuracy: {valid_acc:.2f}%")

        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            try:
                torch.save(self.model.state_dict(), "../weights/model.pth")
            except IOError as e:
                print(f"saving model failed: {e}")

        self._save_graph(float(train_loss) , train_acc, float(valid_loss), valid_acc)

    def _save_graph(self, train_loss, train_acc, valid_loss, valid_acc):
        self.train_loss_history.append(train_loss)
        self.train_acc_history.append(train_acc)
        self.valid_loss_history.append(valid_loss)
        self.valid_acc_history.append(valid_acc)

        fig, ax1 = plt.subplots(figsize=(8, 6))
        ax2 = ax1.twinx()

        ax1.plot(self.train_loss_history, linewidth=2, color='#FDB813', label='Train Loss')
        ax1.plot(self.valid_loss_history, linewidth=2, color='#B2D732', label='Valid Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.set_yscale('log')
        ax1.set_xlim(1, self.num_epochs)
        ax1.set_ylim(1e-3, 1)
        ax1.grid(True)
        ax1.legend(loc='upper left')

        ax2.plot(self.train_acc_history, linewidth=1, color='#FD7F20', label='Train Accuracy')
        ax2.plot(self.valid_acc_history, linewidth=1, color='#87CB16', label='Valid Accuracy')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlim(1, self.num_epochs)
        ax2.set_ylim(0, 100)
        ax2.legend(loc='upper right')

        plt.title('Loss and Accuracy')
        plt.savefig('../data/loss_accuracy_graph.png')
        plt.close()
