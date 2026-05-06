import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric_bars(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    means = df.mean(numeric_only=True)
    metrics = ['dice', 'iou', 'precision', 'recall', 'hd95', 'asd']
    values = [means[m] for m in metrics if m in means]
    labels = [m.upper() for m in metrics if m in means]

    plt.figure(figsize=(8, 4))
    plt.bar(labels, values)
    plt.ylabel('Score / Distance')
    plt.title('Test Performance Summary')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'test_metric_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    plot_metric_bars('./outputs/test_metrics.csv', './outputs')
