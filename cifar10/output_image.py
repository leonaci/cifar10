import os
from . import DEVICE
import torch
import torchvision.transforms as T
from .evaluate import display_images
from .config import load_config
from .model import ImageClassifier
from .load_dataset import get_dataloader

config = load_config("baseline.yaml")

weight_path = [config.weight_path]

if config.suffix != "":
    weight_path.append(config.suffix)

weight_path = os.path.join(config.weight_dir, "-".join(weight_path) + ".pth")

valid_dataloader = get_dataloader("valid", batch_size=config.batch_size)

model = ImageClassifier(config)
model.to(DEVICE)

def output_misclassified_images(weights_path:str = None):
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))

    model.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
    ])

    misclassified_images = []
    predicted_labels = []
    true_labels = []
    for inputs, labels in valid_dataloader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted

        misclassified_mask = predicted != labels

        # extract misclassified images
        misclassified_images.extend([transform(image) for idx, image in enumerate(inputs) if misclassified_mask[idx]])
        predicted_labels.extend([predict.item() for idx, predict in enumerate(predicted) if misclassified_mask[idx]])
        true_labels.extend([label.item() for idx, label in enumerate(labels) if misclassified_mask[idx]])

    assert len(misclassified_images) == len(predicted_labels) == len(true_labels)

    images = []
    for (image, true_label, pred_label) in zip(misclassified_images, true_labels, predicted_labels):
        label = 'True: {true_label}, Pred: {pred_label}'.format(
            true_label=valid_dataloader.dataset.classes[true_label],
            pred_label=valid_dataloader.dataset.classes[pred_label]
        )
        images.append((image, label))

    display_images(images, "data/misclassified")

def output_valid_images():
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((64, 64)),
    ])

    images = []
    for inputs, labels in valid_dataloader:
        for image, label in zip(inputs, labels):
            label = 'Label: {label}'.format(label=valid_dataloader.dataset.classes[label])
            images.append((transform(image), label))

    display_images(images, "data/valid")
