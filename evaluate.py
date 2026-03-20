import torch
from torch.utils.data import DataLoader
import numpy as np
import os

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from dataset import FundusDataset
from model import get_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FundusDataset(
        csv_file="data/test.csv",
        image_dir="data/images",
        train=False   # 🔥 VERY IMPORTANT
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = get_model()
    model.load_state_dict(torch.load("models/multiclass_model.pth"))
    model = model.to(device)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds)

    print(f"\nAccuracy: {acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.txt", "w") as f:
        f.write(f"Accuracy: {acc*100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(report)


if __name__ == "__main__":
    main()