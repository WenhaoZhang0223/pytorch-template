import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

####################################
# 0. Reproducibility
####################################
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

####################################
# 1. Device
####################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################################
# 2. Dataset
####################################
class MyDataset(Dataset):
    def __init__(self, split="train"):
        ...
        self.x = ...
        self.y = ...

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# 建议三分：train / val / test
# train 用来训练，val 用来每个 epoch 监控 + 选最优模型，test 留到最后跑一次
train_dataset = MyDataset(split="train")
val_dataset = MyDataset(split="val")
test_dataset = MyDataset(split="test")

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

####################################
# 3. Model
####################################
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        ...
        self.layers = ...

    def forward(self, x):
        ...
        return x


model = MyModel().to(device)

####################################
# 4. Loss
####################################
loss_fn = nn.CrossEntropyLoss()

####################################
# 5. Optimizer + Scheduler
####################################
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.5,
)

####################################
# 6. Train
####################################
def train(loader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(loader)

####################################
# 7. Evaluate
####################################
def evaluate(loader, model):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return acc, precision, recall, f1

####################################
# 8. Training Loop
####################################
epochs = 10
best_f1 = 0.0

for epoch in range(epochs):
    train_loss = train(train_loader, model, loss_fn, optimizer)
    acc, precision, recall, f1 = evaluate(val_loader, model)
    scheduler.step()

    print(
        f"Epoch {epoch+1} "
        f"Loss:{train_loss:.4f} "
        f"Acc:{acc:.4f} "
        f"P:{precision:.4f} "
        f"R:{recall:.4f} "
        f"F1:{f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        torch.save(model.state_dict(), "best_model.pth")

####################################
# 9. Final Test (只跑一次)
####################################
model.load_state_dict(torch.load("best_model.pth"))
test_acc, test_precision, test_recall, test_f1 = evaluate(test_loader, model)
print(
    f"[Test] Acc:{test_acc:.4f} "
    f"P:{test_precision:.4f} "
    f"R:{test_recall:.4f} "
    f"F1:{test_f1:.4f}"
)