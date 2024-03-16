import argparse
import os
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import CNNModel, LinearModel

main_start_time = time.time()


def load_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST("data", train=True, download=True, transform=transform)
    test = datasets.MNIST("data", train=False, transform=transform)
    return train, test


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--batch_size", type=int)
arg_parser.add_argument("--epochs", type=int)
arg_parser.add_argument("--lr", type=float)
arg_parser.add_argument("--model_path", type=str)
arg_parser.add_argument("--linear", action="store_true")
arg_parser.add_argument("--no_bench", action="store_true")
arg_parser.add_argument("--no_cudnn", action="store_true")
args = arg_parser.parse_args()

linear = bool(args.linear)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_name = "linear" if linear else "cnn"
batch_size = int(args.batch_size) if args.batch_size else (3000 if linear else 200)
model_path = (
    str(args.model_path) if args.model_path else f"model/{model_name}_model.safetensors"
)
lr = float(args.lr) if args.lr else (1e-2 if linear else 2e-4)
epochs = int(args.epochs) if args.epochs else (50 if linear else 20)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
if args.no_bench:
    torch.backends.cudnn.benchmark = False
if args.no_cudnn:
    torch.backends.cudnn.enabled = False

train_set, test_set = load_dataset()


model = CNNModel() if not linear else LinearModel()
model = model.to(device)

train_loader = DataLoader(train_set, batch_size=batch_size)
test_loader = DataLoader(test_set, batch_size=batch_size)

train_size = len(train_set)
test_size = len(test_set)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()
model.train()

for epoch in tqdm(range(epochs)):
    start_time = time.time()
    tqdm.write(f"Epoch {epoch + 1} training")
    loss_sum = 0.0
    for input, target in tqdm(train_loader):
        input = input.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.int64)
        optimizer.zero_grad()
        output: Tensor = model(input)
        loss: Tensor = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * input.size(0)

    avg_loss = loss_sum / train_size
    tqdm.write(f"\tavg_loss: {avg_loss:.4f} (Cost {time.time() - start_time:.2f}s)")

    start_time = time.time()
    tqdm.write(f"Epoch {epoch + 1} testing")
    correct_sum = 0
    for input, target in tqdm(test_loader):
        input = input.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.int64)
        output: Tensor = model(input)
        pred = output.argmax(dim=1, keepdim=True)
        correct_sum += pred.eq(target.view_as(pred)).sum().item()

    accuracy = correct_sum / test_size
    tqdm.write(f"\taccuracy: {accuracy:.2%} (Cost {time.time() - start_time:.2f}s)")

os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
print(f"Total cost {time.time() - main_start_time:.2f}s")
