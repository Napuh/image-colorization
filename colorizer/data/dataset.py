import os

import kornia
import torch
import torchvision.io as io
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode


class NormalizeLAB(object):
    """Normalizes LAB image values to expected ranges.

    L channel is scaled to [0,1]
    ab channels are scaled to [0,1] from [-128,128]
    """

    def __call__(self, img_lab: torch.Tensor) -> torch.Tensor:
        img_lab[:1, ...] /= 100.0  # L en [0, 1]
        img_lab[1:, ...] = (img_lab[1:, ...] + 128.0) / 256.0  # a,b en [0, 1]

        return img_lab


class ToLABTensor(object):
    """Converts RGB tensor to LAB tensor using kornia."""

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        # Normalize to [0,1] for kornia
        img_tensor = img_tensor.float() / 255.0
        # Add batch dimension required by kornia
        img_tensor = img_tensor.unsqueeze(0)
        # Convert to LAB
        lab_tensor = kornia.color.rgb_to_lab(img_tensor)
        # Remove batch dimension
        lab_tensor = lab_tensor.squeeze(0)

        return lab_tensor


class MITPlacesDataset(Dataset):
    """Custom dataset for loading and pre-processing images."""

    def __init__(self, data_path: str, is_train: bool = True):
        """
        Expects a dataset with the following structure:
        data_path/
            class1/
                image1.jpg
                image2.jpg
                ...
            class2/
                image1.jpg
                image2.jpg
                ...
            ...

        Args:
            data_path: Path to the dataset directory
            is_train: If True, apply random augmentations. If False, use deterministic transforms.

        __getitem__ will return L, ab and label.
        """

        self.data_path = data_path
        self.is_train = is_train
        images = []

        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    full_path = os.path.join(root, file)
                    images.append(full_path)

        classes = sorted(
            list(set(os.path.basename(os.path.dirname(image)) for image in images))
        )
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.images = images

        # Different transforms for training and validation
        if is_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    ToLABTensor(),
                    NormalizeLAB(),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(224),
                    ToLABTensor(),
                    NormalizeLAB(),
                ]
            )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets an image in LAB color space.

        Returns:
            Returns a tuple (L, ab, label), where:
                L: stands for lightness - it's the net input (1,224,224)
                ab: is chrominance - something that the net learns (2,224,224)
                label: one-hot encoded class label
        """
        image_path = self.images[idx]

        image = io.read_image(image_path, mode=ImageReadMode.RGB)  # CxHxW, uint8

        img_lab_tensor = self.transform(image)

        # Extract L and ab channels
        L = img_lab_tensor[:1]
        ab = img_lab_tensor[1:]

        # Get label number directly from path
        label_number = self.class_to_idx[os.path.basename(os.path.dirname(image_path))]

        # Return class index as long tensor for CrossEntropyLoss
        label = torch.tensor(label_number, dtype=torch.long)

        return L, ab, label

    def __len__(self) -> int:
        return len(self.images)
