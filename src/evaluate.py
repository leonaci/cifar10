import torch

def evaluate(model, train_dataloader, valid_dataloader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    train_loss = 0.0
    train_acc = 0.0
    total = 0

    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        train_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        train_acc += (predicted == labels).sum().item()
        total += labels.size(0)
    train_loss = train_loss / len(train_dataloader)
    train_acc = 100 * train_acc / total  
    print(f"    Train Loss: {train_loss:.4f}")
    print(f"    Train Accuracy: {train_acc:.2f}%")

    valid_loss = 0.0
    valid_acc = 0.0
    total = 0
    
    for inputs, labels in valid_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        valid_loss += criterion(outputs, labels).item()
        _, predicted = torch.max(outputs.data, 1)
        valid_acc += (predicted == labels).sum().item()
        total += labels.size(0)
    valid_loss = valid_loss / len(valid_dataloader)
    valid_acc = 100 * valid_acc / total
    print(f"    Valid Loss: {valid_loss:.4f}")
    print(f"    Valid Accuracy: {valid_acc:.2f}%")

    return float(train_loss) , train_acc, float(valid_loss), valid_acc
