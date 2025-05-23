from torch.utils.data import Dataset
import os
import PIL.Image


class DFDataset(Dataset):
    def __init__(self, root, df, transform=None):
        super().__init__()
        self.images = [
            os.path.join(root, str(row[1]["image_name"])) for row in df.iterrows()
        ]

        if "target" in df.columns:
            self.target = df["target"].values
        else:
            self.target = None

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.images[idx])
        if self.transform is not None:
            img = self.transform(img)

        if self.target is None:
            return img

        return img, self.target[idx]
