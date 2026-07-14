import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

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
        self.split = split

        if split == "train":
            self.x = ...
            self.y = ...

        elif split == "val":
            self.x = ...
            self.y = ...

        elif split == "test":
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
    shuffle=True, # shuffle=True 表示在每个 epoch 开始时打乱数据顺序，有助于模型更好地泛化。
    num_workers=4, # num_workers 指的是 DataLoader 在加载数据时使用的子进程数量，设置为 0 表示不使用子进程，直接在主进程中加载数据。设置为大于 0 的值可以加快数据加载速度，但也会增加内存占用。默认是 0。
    pin_memory=True, # pin_memory=True 表示将数据加载到固定内存中，这样可以加快数据传输到 GPU 的速度。对于使用 GPU 训练的模型，建议设置为 True。
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False, # 这里不需要打乱验证集数据顺序，因为验证集主要用于评估模型性能，保持数据顺序有助于结果的可重复性。
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

    def forward(self, x): # 前向传播
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
scheduler = torch.optim.lr_scheduler.StepLR( # shcheduler 是学习率调度器，用于在训练过程中动态调整学习率。StepLR 是一种常用的调度策略，它会在每隔一定的 epoch 后将学习率按指定的比例衰减。
    optimizer,
    step_size=5, # step_size=5 表示每隔 5 个 epoch 就会调整一次学习率。
    gamma=0.5, # gamma=0.5 表示每次调整学习率时，将当前学习率乘以 0.5，从而实现学习率的衰减。
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

        optimizer.zero_grad() # 清空梯度
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

            pred = model(x).argmax(dim=1) # argmax(dim=1) 表示在每一行中找到最大值的索引，这里是为了得到模型预测的类别标签。
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
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
    scheduler.step() # scheduler.step() # 调用 scheduler.step() 用于更新学习率，根据 StepLR 的设置，每隔 step_size 个 epoch 会调整一次学习率。

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