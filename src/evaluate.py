import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LiverTumorDataset
from metrics import binarize, compute_metrics
from model import ProposedLiverSegNet
from utils import load_config, get_device, load_checkpoint, ensure_dir
from visualize import save_prediction_panel


def main(config_path, checkpoint_path=None):
    cfg = load_config(config_path)
    device = get_device()
    ensure_dir(cfg['output_dir'])

    split_path = os.path.join(cfg['output_dir'], 'test_split.csv')
    if not os.path.exists(split_path):
        raise FileNotFoundError('test_split.csv not found. Run train.py first to create splits.')

    test_pairs = pd.read_csv(split_path)[['image', 'mask']].values.tolist()
    test_ds = LiverTumorDataset(test_pairs, cfg['image_size'], cfg['window_min'], cfg['window_max'], augment=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    model = ProposedLiverSegNet(
        in_channels=cfg['in_channels'],
        base_channels=cfg['base_channels'],
        attention_heads=cfg['attention_heads'],
        dropout=0.3
    ).to(device)

    checkpoint_path = checkpoint_path or os.path.join(cfg['output_dir'], 'best_model.pth')
    model, _ = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    rows = []
    pred_dir = os.path.join(cfg['output_dir'], 'prediction_panels')
    ensure_dir(pred_dir)

    with torch.no_grad():
        for images, masks, names in tqdm(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
            target = masks.cpu().numpy()[0, 0]
            pred = binarize(probs, cfg['threshold'])
            metrics = compute_metrics(pred, (target > 0.5).astype('uint8'))
            metrics['case'] = names[0]
            rows.append(metrics)
            save_prediction_panel(
                images.cpu().numpy()[0, 0],
                target,
                pred,
                os.path.join(pred_dir, names[0] + '_panel.png')
            )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg['output_dir'], 'test_metrics.csv'), index=False)
    print(df.mean(numeric_only=True))


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else './configs/config.yaml'
    checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
    main(config_path, checkpoint_path)
