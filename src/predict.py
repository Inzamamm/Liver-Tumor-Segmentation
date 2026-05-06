import os
import sys
import cv2
import torch
import numpy as np

from preprocessing import preprocess_ct
from model import ProposedLiverSegNet
from utils import load_config, get_device, load_checkpoint, ensure_dir


def main(config_path, image_path, checkpoint_path=None):
    cfg = load_config(config_path)
    device = get_device()
    checkpoint_path = checkpoint_path or os.path.join(cfg['output_dir'], 'best_model.pth')
    ensure_dir(cfg['output_dir'])

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f'Could not read image: {image_path}')
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    processed = preprocess_ct(image, cfg['image_size'], cfg['window_min'], cfg['window_max'])
    tensor = torch.from_numpy(processed).float().unsqueeze(0).unsqueeze(0).to(device)

    model = ProposedLiverSegNet(
        in_channels=cfg['in_channels'],
        base_channels=cfg['base_channels'],
        attention_heads=cfg['attention_heads'],
        dropout=0.3
    ).to(device)
    model, _ = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).cpu().numpy()[0, 0]
        pred = (prob >= cfg['threshold']).astype(np.uint8) * 255

    out_mask = os.path.join(cfg['output_dir'], 'single_prediction_mask.png')
    out_prob = os.path.join(cfg['output_dir'], 'single_prediction_probability.png')
    cv2.imwrite(out_mask, pred)
    cv2.imwrite(out_prob, (prob * 255).astype(np.uint8))
    print('Saved:', out_mask)
    print('Saved:', out_prob)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python src/predict.py configs/config.yaml path/to/image.png [path/to/best_model.pth]')
        sys.exit(1)
    config_path = sys.argv[1]
    image_path = sys.argv[2]
    checkpoint_path = sys.argv[3] if len(sys.argv) > 3 else None
    main(config_path, image_path, checkpoint_path)
