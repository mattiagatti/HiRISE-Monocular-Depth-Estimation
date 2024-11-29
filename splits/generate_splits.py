import argparse
import numpy as np
import pandas as pd

from pathlib import Path


def write_split(split_name, split): 
    data_dir = args.data_dir
    tile_list = []
    for folder_name in split:
        folder_path = data_dir / folder_name / "tiles"
        tiles = list(Path(folder_path).iterdir())
        tile_list += [f"{folder_name}/tiles/{x.name}" for x in tiles]
    
    with open(Path(f"splits/{split_name}.txt"), "w") as f:
        f.writelines(x + '\n' for x in tile_list)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    
    df = pd.read_csv("dataset/data.csv")["name"]
    train, val, test = np.split(df.sample(frac=1, random_state=42), [int(.8*len(df)), int(.9*len(df))])
    write_split("train", train)
    write_split("val", val)
    write_split("test", test)
