import rasterio as rio
import warnings

from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms


class HiRISEDataset(Dataset):
    def __init__(self, root, stage, transform=None):
        self.root = Path(root)
        self.stage = stage
        self.transform = transform
        self.filenames = self._read_split()
        warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        raster_path = self.root / filename

        with rio.open(raster_path, "r") as ds:
            image = ds.read(1).astype("uint8")
            dtm = ds.read(2)

        dtm = abs(dtm - dtm.max())
        to_tensor = transforms.ToTensor()

        if self.transform is not None:
            transformed = self.transform(image=image, mask=dtm)
            image = transformed["image"]
            dtm = transformed["mask"]

        image=to_tensor(image)
        dtm=to_tensor(dtm)

        return image, dtm

    def _read_split(self):
        split_filename = f"{self.stage}.txt"
        split_filepath = Path.cwd() / "splits" / split_filename
        filenames = split_filepath.read_text().splitlines()
        return filenames