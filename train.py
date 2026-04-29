import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from dataset import FundusDataset
from model import get_model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = FundusDataset(
        csv_file="data/train.csv",
        image_dir="data/images",
        train=True
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )

    model = get_model().to(device)

    # Class weights 
    class_counts = [3104, 95, 312, 446, 310]
    total = sum(class_counts)
    weights = [total / c for c in class_counts]
    weights = torch.tensor([1.0, 8.0, 5.0, 2.0, 2.0]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epochs = 25   

    epoch_logs = []

    for epoch in range(epochs):

        model.train()
        total_loss = 0

        for images, labels in loader:

            images = images.to(device)
            labels = labels.long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        log_line = f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}"
        print(log_line)
        epoch_logs.append(log_line)

    # Save model
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/multiclass_model.pth")

    # Save logs
    os.makedirs("results", exist_ok=True)
    with open("results/train_log.txt", "w") as f:
        for line in epoch_logs:
            f.write(line + "\n")

    print("Training complete!")


if __name__ == "__main__":
    main()
