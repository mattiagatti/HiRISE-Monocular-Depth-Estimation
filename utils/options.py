import argparse
from pathlib import Path


def initialize():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-cp", "--ckpt_path", type=Path, default=None, help="checkpoint or pretrained path")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--data_dir", type=Path, default=Path.cwd())
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu"], help="cpu or gpu")
    parser.add_argument("--devices", type=str, default=1, help="number of accelerators to use")
    parser.add_argument('--test', action=argparse.BooleanOptionalAction)
    return parser

