# train.py
import torch, time
import torch.nn as nn
import torch.optim as optim
from Utils import plot_accuracy

def train_model(model, train_loader, val_loader, class_weights, device, epochs=50, patience=10):
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_acc, trigger = 0, 0
    tacc, vacc = [], []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        correct, total = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total
        tacc.append(train_acc)

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                correct += (out.argmax(1) == yb).sum().item()
                total += yb.size(0)
        val_acc = correct / total
        vacc.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs} | Train {train_acc:.4f} | Val {val_acc:.4f} | Time {(time.time()-start)/60:.2f}m")

        if val_acc > best_val_acc:
            best_val_acc, trigger = val_acc, 0
            #torch.save(model.state_dict(), "best_model.pth")
        else:
            trigger += 1
            if trigger >= patience:
                print("Early stopping.")
                break

    plot_accuracy(tacc, vacc)
    torch.save(model, "model.pth")

    return model
