import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config import Config
from model import SwinTinyBinary
from data_loader import get_dataloaders
from utils import plot_metrics

def train_model():
    train_loader, val_loader = get_dataloaders()
    model = SwinTinyBinary().to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.learning_rate)

    best_val_acc = 0.0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(Config.num_epochs):
        model.train()
        train_loss, correct, total = 0, 0, 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.num_epochs}")

        for images, labels in progress_bar:
            images, labels = images.to(Config.device), labels.to(Config.device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(Config.device), labels.to(Config.device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{Config.num_epochs}], Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.checkpoint_path)
            print("Checkpoint Saved!")

    plot_metrics(train_losses, val_losses, train_accs, val_accs)
