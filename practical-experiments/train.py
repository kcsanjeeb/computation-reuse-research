# train.py
import argparse, os, time, json, io
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from datetime import datetime

# -----------------------
# Utils: Katib + MinIO
# -----------------------
def katib_log(**metrics):
    # Katib default file collector regex matches "name=value"
    for k, v in metrics.items():
        try:
            vv = float(v)
        except Exception:
            vv = v
        print(f"{k}={vv}", flush=True)

def make_minio_client():
    endpoint = os.getenv("MINIO_ENDPOINT", "")
    bucket = os.getenv("MINIO_BUCKET", "")
    ak = os.getenv("MINIO_ACCESS_KEY", "")
    sk = os.getenv("MINIO_SECRET_KEY", "")
    if not endpoint or not bucket or not ak or not sk:
        return None, None
    try:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=("http://" + endpoint) if not endpoint.startswith("http") else endpoint,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
        )
        return s3, bucket
    except Exception as e:
        print(f"minio_client_error={e}", flush=True)
        return None, None

def s3_upload_bytes(s3, bucket, key, data_bytes, content_type="application/octet-stream"):
    if s3 is None:
        return
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=data_bytes, ContentType=content_type)
        print(f"s3_upload=ok key={key}", flush=True)
    except Exception as e:
        print(f"s3_upload_error key={key} err={e}", flush=True)

# -----------------------
# Activation capture
# -----------------------
class HookKeeper:
    def __init__(self, device, probe_loader, layers_to_hook):
        self.device = device
        self.probe_loader = probe_loader
        self.layers_to_hook = layers_to_hook
        self.handles = []
        self.cache = {name: [] for name in layers_to_hook}

    def _mk_hook(self, name):
        def _hook(module, inp, out):
            with torch.no_grad():
                # Flatten per-sample activations
                act = out.detach()
                if isinstance(act, (list, tuple)):
                    act = act[0]
                act = act.reshape(act.size(0), -1).cpu().numpy()
                self.cache[name].append(act)
        return _hook

    def register(self, model):
        # Map dotted names to modules
        module_map = dict(model.named_modules())
        for name in self.layers_to_hook:
            if name not in module_map:
                raise ValueError(f"Layer '{name}' not found in model.named_modules()")
            self.handles.append(module_map[name].register_forward_hook(self._mk_hook(name)))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def run_probe(self, model):
        model.eval()
        self.cache = {k: [] for k in self.cache}
        with torch.no_grad():
            for x, _ in self.probe_loader:
                x = x.to(self.device, non_blocking=True)
                _ = model(x)
        # Concatenate across probe batches
        out = {}
        for k, parts in self.cache.items():
            if len(parts) == 0:
                continue
            arr = np.concatenate(parts, axis=0)
            out[k] = arr
        return out

# -----------------------
# Data
# -----------------------
def build_datasets(use_fake, data_dir):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([transforms.ToTensor()])

    if use_fake:
        train = datasets.FakeData(size=4096, image_size=(3, 32, 32), num_classes=10, transform=tf_train)
        test  = datasets.FakeData(size=2048, image_size=(3, 32, 32), num_classes=10, transform=tf_eval)
        return train, test

    # Try CIFAR-10 with offline fallback
    try:
        train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf_train)
        test  = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf_eval)
        return train, test
    except Exception as e:
        print(f"cifar_download_error={e} -> falling back to FakeData", flush=True)
        return build_datasets(True, data_dir)

# -----------------------
# Train/Eval
# -----------------------
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        running_loss += loss.item() * x.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            out = model(x)
            loss = loss_fn(out, y)
            total_loss += loss.item() * x.size(0)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(total, 1), total_loss / max(total, 1)

# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cifar10")
    p.add_argument("--model", default="resnet18")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adam","sgd"], default="adam")
    p.add_argument("--data-dir", default="/workspace/data")
    p.add_argument("--probe-size", type=int, default=512)
    p.add_argument("--log-activations-epochs", default="1,2,4,8,16,20")
    p.add_argument("--trial-id", default=os.getenv("KATIB_TRIAL_NAME", "local-trial"))
    p.add_argument("--minio-prefix", default="exp1")
    p.add_argument("--use-fakedata", action="store_true", help="Force FakeData to avoid downloads")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.data_dir, exist_ok=True)

    train_ds, test_ds = build_datasets(args.use_fakedata, args.data_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    # Fixed probe subset (first N of test) for reproducibility
    probe_indices = list(range(min(args.probe_size, len(test_ds))))
    probe_loader  = DataLoader(Subset(test_ds, probe_indices), batch_size=64, shuffle=False, num_workers=2)

    # Model
    if args.model.lower() == "resnet18":
        net = models.resnet18(num_classes=10)
        if args.dropout > 0:
            # Replace fc with dropout+fc
            in_f = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(p=args.dropout), nn.Linear(in_f, 10))
    else:
        raise ValueError("Only resnet18 demo implemented")

    net = net.to(device)

    # Optimizer
    if args.optimizer == "adam":
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)

    loss_fn = nn.CrossEntropyLoss()

    # Layers to hook (resnet18 canonical names)
    layers_to_hook = [
        "conv1",
        "layer1.1.relu",   # last block in stage1
        "layer2.1.relu",
        "layer3.1.relu",
        "layer4.1.relu",
        "fc",
    ]
    hk = HookKeeper(device, probe_loader, layers_to_hook)
    hk.register(net)

    # MinIO
    s3, bucket = make_minio_client()
    prefix = f"{args.minio_prefix}/{args.trial_id}"
    # act_epochs = set([int(x) for x in args.log_activations-epochs.split(",")]) if isinstance(args.log_activations-epochs,str) else set(args.log_activations_epochs)
    # before (buggy): args.log_activations-epochs
    # after (fixed):
    if isinstance(args.log_activations_epochs, str):
        act_epochs = set(int(x) for x in args.log_activations_epochs.split(","))
    else:
        act_epochs = set(args.log_activations_epochs)

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(net, train_loader, opt, loss_fn, device)
        val_acc, val_loss = evaluate(net, test_loader, device)

        # Metrics for Katib
        wall = time.time() - t0
        katib_log(train_loss=tr_loss, val_accuracy=val_acc, wall_time_seconds=wall, epoch=epoch)

        # Save checkpoint
        ckpt_bytes = io.BytesIO()
        torch.save({"model": net.state_dict(), "epoch": epoch}, ckpt_bytes)
        ckpt_key = f"{prefix}/checkpoints/epoch_{epoch}.pt"
        s3_upload_bytes(s3, bucket, ckpt_key, ckpt_bytes.getvalue(), content_type="application/octet-stream")

        # Log activations on probe at selected epochs
        if epoch in act_epochs:
            acts = hk.run_probe(net)
            for lname, arr in acts.items():
                npz_bytes = io.BytesIO()
                # Save as compressed npz with consistent key
                np.savez_compressed(npz_bytes, activations=arr)
                npz_key = f"{prefix}/activations/epoch_{epoch}/{lname}.npz"
                s3_upload_bytes(s3, bucket, npz_key, npz_bytes.getvalue(), content_type="application/octet-stream")

        # Track best
        if val_acc > best_acc:
            best_acc = val_acc
            # Also upload a "best.pt"
            best_bytes = io.BytesIO()
            torch.save({"model": net.state_dict(), "epoch": epoch, "val_acc": float(val_acc)}, best_bytes)
            best_key = f"{prefix}/checkpoints/best.pt"
            s3_upload_bytes(s3, bucket, best_key, best_bytes.getvalue(), content_type="application/octet-stream")

    # Final summary
    katib_log(final_val_accuracy=best_acc)
    print("training_done=1", flush=True)

if __name__ == "__main__":
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    main()
