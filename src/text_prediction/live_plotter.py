import json
import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output
import os

def live_plot(metadata_file, iterations=10, interval=5):
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Loss')
    train_line, = ax.plot([], [], label='Train Loss')
    val_line, = ax.plot([], [], label='Val Loss')
    gap_line, = ax.plot([], [], label='Gap (Val - Train)')
    grad_norm_line, = ax.plot([], [], label='Gradient Norm')
    ax.legend()

    for _ in range(iterations):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        train_losses = metadata["train_losses"]
        val_losses = metadata["val_losses"]
        grad_norms = metadata.get("grad_norms", [])
        gap = [v - t for v, t in zip(val_losses, train_losses)]

        train_line.set_xdata(range(len(train_losses)))
        train_line.set_ydata(train_losses)
        val_line.set_xdata(range(len(val_losses)))
        val_line.set_ydata(val_losses)
        grad_norm_line.set_xdata(range(len(grad_norms)))
        grad_norm_line.set_ydata(grad_norms)

        ax.relim()
        ax.autoscale_view()
        ax.legend([f'Train Loss: {train_losses[-1]:.4f}', f'Val Loss: {val_losses[-1]:.4f}', f'Gap: {gap[-1]:.4f}', f'Grad Norm: {grad_norms[-1]:.4f}' if grad_norms else 'Grad Norm: N/A'])
        clear_output(wait=True)
        display(fig)
        time.sleep(interval)  # Update every interval seconds

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Live Plotter")
    parser.add_argument("--metadata_file", type=str, required=True, help="Path to the metadata file")
    args = parser.parse_args()

    live_plot(args.metadata_file)
