import torch
import numpy as np
from PIL import Image

# Mean and std for normalization
MEAN_GRAY = torch.tensor([0.47744867]).reshape(1, 1, 1)
STD_GRAY = torch.tensor([0.24919957]).reshape(1, 1, 1)
MEAN_RGB = torch.tensor([0.49828297, 0.47790086, 0.42099345]).reshape(3, 1, 1)
STD_RGB = torch.tensor([0.2627286,  0.2536955,  0.27374935]).reshape(3, 1, 1)

# --- Pixel proportion for mask classes ---"""
# pixels class 0 (background): 0.1026
# pixels class 1 (vessel): 0.8974

def normalize_img(img, gray_scale=True):
    """
    Normalizes an input image tensor to have zero mean and unit variance.

    Args:
        img (torch.Tensor): The input image tensor. Expected to have pixel values in the range [0, 255].
        gray_scale (bool, optional): If True, normalizes using grayscale mean and standard deviation (MEAN_GRAY, STD_GRAY).
                                     If False, normalizes using RGB mean and standard deviation (MEAN_RGB, STD_RGB).
                                     Defaults to True.

    Returns:
        torch.Tensor: The normalized image tensor.

    Note:
        The constants MEAN_GRAY, STD_GRAY, MEAN_RGB, and STD_RGB must be defined in the scope.
    """

    img = img.float() / 255.0
    if gray_scale:
        img = (img - MEAN_GRAY) / STD_GRAY
    else:
        img = (img - MEAN_RGB) / STD_RGB
    return img


class VesselShapeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for generating vessel shape images and corresponding masks.

    This dataset utilizes a vessel shape generator to produce synthetic vessel images, binary masks, and optional metadata.
    It supports grayscale or RGB images, normalization, and can optionally return metadata for each sample.

    Args:
        vess_shape_generator: An object with a `generate_vess_shape()` method that returns (img_blend, mask_bin, img_metadata).
        n_samples (int, optional): Number of samples in the dataset. Default is 10.
        gray_scale (bool, optional): If True, images are converted to grayscale. If False, images are RGB. Default is True.
        normalize (bool, optional): If True, images are normalized using `normalize_img`. Default is True.
        metadata (bool, optional): If True, returns metadata along with image and mask. Default is False.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Generates and returns the image, mask, and optionally metadata for the given index.

    Returns:
        If metadata is False:
            img (Tensor): Image tensor of shape [1, H, W] (grayscale) or [3, H, W] (RGB), normalized if specified.
            mask (Tensor): Binary mask tensor of shape [H, W].
        If metadata is True:
            img (Tensor): As above.
            mask (Tensor): As above.
            img_metadata: Metadata associated with the generated image.
    """


    def __init__(self, vess_shape_generator, n_samples=10, gray_scale=True, normalize=True, metadata=False):
        self.vess_shape_generator = vess_shape_generator
        self.n_samples = n_samples
        self.gray_scale = gray_scale
        self.normalize = normalize
        self.metadata = metadata

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img_blend, mask_bin, img_metadata = self.vess_shape_generator.generate_vess_shape()
        if self.gray_scale:
            img_pil = Image.fromarray(img_blend).convert("L")
            img = torch.from_numpy(np.array(img_pil)).unsqueeze(0)  # [1,H,W]
            if self.normalize:
                img = normalize_img(img, gray_scale=True)
            else:
                img = img.float() / 255.0
        else:
            img = torch.from_numpy(np.array(img_blend)).permute(2,0,1)  # [3,H,W]
            if self.normalize:
                img = normalize_img(img, gray_scale=False)
            else:
                img = img.float() / 255.0
        mask = torch.from_numpy(mask_bin).long()  # [H,W]
        # mask = mask.unsqueeze(0)
        if self.metadata:
            return img, mask, img_metadata
        else:
            return img, mask
