import os
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import pair_image_masks, make_splits, LiverTumorDataset
from losses import CompositeSegLoss
from metrics import binarize, compute_metrics
from model import ProposedLiverSegNet
from utils import load_config, set_seed, ensure_dir, get_device, save_checkpoint
from visualize import plot_training_curves


def run_epoch(model, loader, criterion, optimizer, device, train=True, threshold=0.5):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_metrics = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, masks, _ in tqdm(loader, leave=False):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets = masks.detach().cpu().numpy()

            for p, t in zip(probs, targets):
                pred = binarize(p.squeeze(), threshold)
                target = (t.squeeze() > 0.5).astype('uint8')
                all_metrics.append(compute_metrics(pred, target))

    avg_loss = total_loss / len(loader.dataset)
    df = pd.DataFrame(all_metrics)
    return avg_loss, df.mean(numeric_only=True).to_dict()


def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg['seed'])
    device = get_device()
    ensure_dir(cfg['output_dir'])

    pairs = pair_image_masks(cfg['image_dir'], cfg['mask_dir'])
    train_pairs, val_pairs, test_pairs = make_splits(
        pairs,
        cfg['train_split'],
        cfg['val_split'],
        cfg['test_split'],
        cfg['seed']
    )

    pd.DataFrame(train_pairs, columns=['image', 'mask']).to_csv(os.path.join(cfg['output_dir'], 'train_split.csv'), index=False)
    pd.DataFrame(val_pairs, columns=['image', 'mask']).to_csv(os.path.join(cfg['output_dir'], 'val_split.csv'), index=False)
    pd.DataFrame(test_pairs, columns=['image', 'mask']).to_csv(os.path.join(cfg['output_dir'], 'test_split.csv'), index=False)

    train_ds = LiverTumorDataset(train_pairs, cfg['image_size'], cfg['window_min'], cfg['window_max'], augment=True)
    val_ds = LiverTumorDataset(val_pairs, cfg['image_size'], cfg['window_min'], cfg['window_max'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_ds, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])

    model = ProposedLiverSegNet(
        in_channels=cfg['in_channels'],
        base_channels=cfg['base_channels'],
        attention_heads=cfg['attention_heads'],
        dropout=0.3
    ).to(device)

    criterion = CompositeSegLoss(cfg['loss_dice_weight'], cfg['loss_bce_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    best_dice = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    log_rows = []

    for epoch in range(1, cfg['num_epochs'] + 1):
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, True, cfg['threshold'])
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, False, cfg['threshold'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_metrics['dice'])
        history['val_iou'].append(val_metrics['iou'])

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **{f'train_{k}': v for k, v in train_metrics.items()},
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        log_rows.append(row)
        pd.DataFrame(log_rows).to_csv(os.path.join(cfg['output_dir'], 'training_log.csv'), index=False)

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            save_checkpoint(model, optimizer, epoch, best_dice, os.path.join(cfg['output_dir'], 'best_model.pth'))

        print(f"Epoch {epoch:03d}/{cfg['num_epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_metrics['dice']:.4f} | Val IoU: {val_metrics['iou']:.4f}")

    plot_training_curves(history, os.path.join(cfg['output_dir'], 'training_curves.png'))
    print('Training complete. Best validation Dice:', best_dice)


if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else './configs/config.yaml'
    main(config_path)
