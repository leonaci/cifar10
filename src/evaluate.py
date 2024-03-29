import torch
from load_dataset import get_dataloader
from model import ImageClassifier, evaluate

test_dataloader = get_dataloader("test", batch_size=32)

model = ImageClassifier()

model.load_state_dict(torch.load("../weights/model.pth"))

accuracy = evaluate(model, test_dataloader)

print(f"Accuracy: {accuracy:.2f}%")
