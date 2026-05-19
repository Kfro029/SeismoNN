from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from seismonn.data.dataset import SeismoDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


metadata_path = "data/metadata.csv"
data_root = "."

train_dataset = SeismoDataset(
    metadata_path=metadata_path,
    split="train",
    data_root=data_root,
    normalize=True,
)

val_dataset = SeismoDataset(
    metadata_path=metadata_path,
    split="val",
    data_root=data_root,
    normalize=True,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)


class SeismoCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


model = SeismoCNN()
# изменить первый слой (3 → 2 канала)
"""
model.features[0][0] = nn.Conv2d(
    2,
    32,
    kernel_size=3,
    stride=2,
    padding=1,
    bias=False
)

# изменить классификатор (3 класса)

model.classifier[1] = nn.Linear(1280,3)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def evaluate(model, loader):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)

            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, cm


train_losses = []
train_accuracies = []
val_accuracies = []

num_epochs = 2

for epoch in range(num_epochs):
    model.train()

    total_loss = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        out = model(x)

        loss = criterion(out, y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    train_acc, _ = evaluate(model, train_loader)
    val_acc, cm = evaluate(model, val_loader)

    avg_loss = total_loss / len(train_loader)

    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print("\nEpoch {0}".format(epoch + 1))
    print("Loss:", total_loss)
    print("Train acc:", train_acc)
    print("Val acc:", val_acc)

print("\nConfusion matrix")
print(cm)


sample, _ = train_dataset[0]


plt.figure()
plt.plot(train_accuracies, label="Train")
plt.plot(val_accuracies, label="Val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(OUTPUT_DIR / "accuracy.png")
plt.close()


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": total_loss,
    },
    OUTPUT_DIR / "checkpoint.pth",
)
