import os
import random
import yaml
import numpy as np
import torch


def load_config(path: str):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_checkpoint(model, optimizer, epoch, best_metric, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
    }, path)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint
