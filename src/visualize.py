import os
import numpy as np
import matplotlib.pyplot as plt


def save_prediction_panel(image, mask, pred, save_path):
    image = image.squeeze()
    mask = mask.squeeze()
    pred = pred.squeeze()
    error = np.abs(pred.astype(float) - mask.astype(float))

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('CT Image')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Prediction')
    axes[3].imshow(image, cmap='gray')
    axes[3].imshow(pred, alpha=0.45, cmap='Greens')
    axes[3].set_title('Overlay')
    axes[4].imshow(error, cmap='hot')
    axes[4].set_title('Error Map')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_training_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history['train_loss'], label='Training Loss')
    axes[0].plot(epochs, history['val_loss'], label='Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(epochs, history['val_dice'], label='Validation Dice')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].legend()

    axes[2].plot(epochs, history['val_iou'], label='Validation IoU')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
