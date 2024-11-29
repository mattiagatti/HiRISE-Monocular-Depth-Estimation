import argparse
import numpy as np
import pandas as pd
import rasterio as rio
import requests
import time

from bs4 import BeautifulSoup
from patchify import patchify
from pathlib import Path
from rasterio.enums import Resampling
from tqdm import tqdm


data_path = Path("dataset/data.csv")
dataset_dir = Path(f"/home/super/datasets-nas/HiRISE/")
orbits_url = "https://www.uahirise.org/PDS/DTM/ESP/"


def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            tqdm_params = {
                'desc': url,
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)
                    

def split(image_path, model_path, output_dir):    
    with rio.open(model_path, "r") as ds:
        model_np, model_profile = ds.read(1), ds.profile
        
    out_shape = (1, model_np.shape[0], model_np.shape[1])
    resampling = Resampling.bilinear
    
    with rio.open(image_path, "r") as ds:
        image_np, image_profile = ds.read(1, out_shape=out_shape, resampling=resampling), ds.profile

    resource_np = np.stack([image_np, model_np])
    patches = patchify(resource_np, (2, patch_size, patch_size), step=step) # 1, rows, cols, 3, height, width
    patches = patches.reshape(-1, 2, patch_size, patch_size)
    
    image_nodata = image_profile["nodata"]
    model_nodata = model_profile["nodata"]

    patches = [x for x in patches if image_nodata not in x[0] and model_nodata not in x[1]]

    for i, patch in enumerate(patches):
        with rio.open(
            output_dir / f"{i}.tiff",
            "w",
            driver="GTiff",
            count=2,
            height=patch_size,
            width=patch_size,
            dtype=rio.float32
        ) as dst:
            dst.write(patch.astype(rio.float32))
    
    print(f"{len(patches)} patches have been generated...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--save_dir", type=int, default=Path.cwd())
    parser.add_argument("--patch_size", type=int, default=512, choices=[256, 512])
    parser.add_argument("--step", type=int, default=384)
    args = parser.parse_args()
    
    patch_size = args.patch_size
    step = args.step
    
    images = []
    models = []
    if not data_path.exists():
        print(f"Scraping data from the website...")
        
        orbits_soup = BeautifulSoup(requests.get(orbits_url).content, "html.parser")
        orbits_href = [orbits_url + a["href"] for a in orbits_soup.find_all("a")[1:]]
        orbits_soup = [BeautifulSoup(requests.get(url).content, "html.parser") for url in orbits_href]
        orbits_href = [[href + x["href"] for x in soup.find_all("a")[1:]] for href, soup in zip(orbits_href, orbits_soup)]
        orbits_href = [x for xs in orbits_href for x in xs]
        
        for i, orbit_href in enumerate(orbits_href):
            print(f"{i+1}/{len(orbits_href)} {orbit_href}")
            orbit_soup = BeautifulSoup(requests.get(orbit_href).content, "html.parser")
            resources_href = [orbit_href + a["href"] for a in orbit_soup.find_all("a")]
            images_href = [x for x in resources_href if x.endswith("RED_C_01_ORTHO.JP2")]
            models_href = [x for x in resources_href if x.endswith("A01.IMG")]
            if len(images_href) > 0 and len(models_href) > 0:
                images.append(images_href[0])
                models.append(models_href[0])
        
        df = pd.DataFrame({"name": [x.split("/")[-2] for x in images], "image": images, "model": models}).sort_values(by="image")
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)
        observations = df["name"].tolist()
        images = df["image"].tolist()
        models = df["model"].tolist()
    
    for observation, image, model in zip(observations, images, models):
        image_name = image.split("/")[-1]
        model_name = model.split("/")[-1]
        file_stem = image_name.split(".")[0]

        output_dir = dataset_dir / observation
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / image_name
        model_path = output_dir / model_name
            
        if not image_path.exists():
            while(True):
                try:
                    print(f"Downloading {image_name}...")
                    download(image, image_path)
                    break
                except:
                    time.sleep(600)
                    continue   
        
        if not model_path.exists():
            while(True):
                try:
                    print(f"Downloading {model_name}...")
                    download(model, model_path)
                    break
                except:
                    time.sleep(600)
                    continue
        
        output_dir = image_path.parent / "tiles"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Splitting {image_name} and {model_name} into tiles...")
            split(image_path, model_path, output_dir)
            
            