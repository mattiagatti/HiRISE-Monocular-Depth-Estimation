from pathlib import Path


def save_metrics(save_dir, metrics, epoch=None, n_epochs=None):
    epoch_line = f"Epoch: {epoch:03} - {n_epochs:03}" if epoch != None else "Test"
    keys_line = "".join([f"{key:>10}" for key in list(metrics.keys())])
    values_line = "".join([f"{value:10.4f}" for value in list(metrics.values())])
    separator_line = "=" * (len(values_line) + 4)

    lines = [epoch_line, separator_line, keys_line, values_line, separator_line, "\n"]
    summary = "\n".join(lines)

    with open(Path(save_dir) / "metrics.txt", "a") as f:
        f.write(summary)
