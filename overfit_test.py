import torch, torch.nn as nn
from train_perception import PerceptionNet, NUM_CLASSES
from train_perception import PerceptionDataset
import random
from torchvision import transforms

ds = PerceptionDataset(data_dir='perception_data', transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])]))
if len(ds) < 4:
    print("Dataset too small for overfit test. Generate data first.")
    raise SystemExit

xs = []
ys = []
for i in range(4):
    x, y = ds[i]
    xs.append(x.unsqueeze(0))
    ys.append(torch.tensor([int(y)], dtype=torch.long))

x_batch = torch.cat(xs, dim=0)
y_batch = torch.cat(ys, dim=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = PerceptionNet().to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

x_batch = x_batch.to(device)
y_batch = y_batch.to(device)

for i in range(1000):
    model.train()
    opt.zero_grad()
    logits = model(x_batch)
    loss = loss_fn(logits, y_batch)
    loss.backward()
    opt.step()
    if i % 100 == 0:
        preds = logits.argmax(dim=-1)
        acc = (preds == y_batch).float().mean().item()
        print(f"iter {i} loss={loss.item():.4f} acc={acc:.3f}")

print("Final loss", loss.item())
