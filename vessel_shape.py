import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import gaussian_filter
from vessel_geometry import VesselGeometry
import torch

# Set seed for reproducibility
np.random.seed(42)


class VesselShape:
    def __init__(
        self,
        image_size=256,
        n_control_points=3,
        max_vd=30,
        radius=3,
        num_curves=1,
        extra_space=32,
        sigma=1.0,
        texture_dir=None,
        annotation_csv=None,
        crop_size=(256, 256)
    ):
        """
        Initialize the VesselShape dataset generator.

        Args:
            image_size (int): Size of the generated image (height and width).
            n_control_points (int): Number of control points for the vessel geometry.
            max_vd (int): Maximum vessel diameter.
            radius (int): Vessel radius.
            num_curves (int): Number of vessel curves per image.
            extra_space (int): Extra space around the vessel.
            texture_dir (str): Directory containing texture images.
            annotation_csv (str): Path to the CSV file with texture annotations.
            crop_size (tuple): Size of the crop to be applied to textures.
            n_samples (int): Number of samples in the dataset.
            sigma (float): Sigma value for Gaussian smoothing of the mask.
        """
        self.n_control_points = n_control_points
        self.max_vd = max_vd
        self.radius = radius
        self.num_curves = num_curves
        self.image_size = image_size
        self.extra_space = extra_space
        self.crop_size = crop_size
        self.texture_dir = texture_dir
        self.annotation_csv = annotation_csv
        self.sigma = sigma
        self.texture_metadata = None
        
        
        
        if annotation_csv is not None:
            self.texture_metadata = pd.read_csv(annotation_csv)
        self.texture_files = [
            f for f in os.listdir(texture_dir) if f.lower().endswith(".jpeg")
        ]
        
    def random_crop(self, img, crop_size=(256, 256)):
        """
        Perform a random crop on the input image.

        Args:
            img (np.ndarray): Input image to be cropped.
            crop_size (tuple): Desired crop size (height, width).

        Returns:
            np.ndarray: Cropped image.

        Raises:
            ValueError: If the image is smaller than the crop size.
        """
        img_height, img_width = img.shape[:2]
        if img_height < crop_size[0] or img_width < crop_size[1]:
            raise ValueError("Image is too small for the desired crop size.")
        y = np.random.randint(0, img_height - crop_size[0] + 1)
        x = np.random.randint(0, img_width - crop_size[1] + 1)
        return img[y : y + crop_size[0], x : x + crop_size[1]]

    def validate_image_dimensions(self, img, target_channels=3):
        """
        Ensure the image has the required number of channels.

        Args:
            img (np.ndarray): Input image.
            target_channels (int): Number of channels required.

        Returns:
            np.ndarray: Image with the correct number of channels.

        Raises:
            ValueError: If the image cannot be converted to the required number of channels.
        """
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, target_channels, axis=-1)
        elif img.shape[2] != target_channels:
            raise ValueError(
                f"Image has {img.shape[2]} channels, but {target_channels} channels are required."
            )
        return img

    def validate_image_size(self, foreground_texture, background_texture, crop_size):
        """
        Check if both foreground and background textures are large enough for cropping.

        Args:
            foreground_texture (np.ndarray): Foreground texture image.
            background_texture (np.ndarray): Background texture image.
            crop_size (tuple): Desired crop size (height, width).

        Returns:
            bool: True if both textures are large enough, False otherwise.
        """
        if (
            foreground_texture.shape[0] >= crop_size[0]
            and foreground_texture.shape[1] >= crop_size[1]
            and background_texture.shape[0] >= crop_size[0]
            and background_texture.shape[1] >= crop_size[1]
        ):
            return True
        else:
            return False

    def select_textures(self):
        """
        Randomly select two different texture images (foreground and background).
        Ensures they are from different classes if annotation is available and
        that both are large enough for cropping.

        Returns:
            tuple: (foreground_image, background_image, foreground_filename, background_filename)
        """
        while True:
            fg_file = random.choice(self.texture_files)
            bg_file = random.choice(self.texture_files)
            if fg_file == bg_file:
                continue
            if self.texture_metadata is not None:
                fg_label = self.texture_metadata[
                    self.texture_metadata["image_id"] == os.path.splitext(fg_file)[0]
                ]["label_id"].values
                bg_label = self.texture_metadata[
                    self.texture_metadata["image_id"] == os.path.splitext(bg_file)[0]
                ]["label_id"].values
                if len(fg_label) == 0 or len(bg_label) == 0:
                    continue
                if fg_label[0] == bg_label[0]:
                    continue
            fg_img = np.array(Image.open(os.path.join(self.texture_dir, fg_file)))
            bg_img = np.array(Image.open(os.path.join(self.texture_dir, bg_file)))
            if self.validate_image_size(fg_img, bg_img, self.crop_size):
                return fg_img, bg_img, fg_file, bg_file

    def blend(self, foreground, background, mask, sigma=1):
        """
        Blend the foreground and background images using the provided mask.

        Args:
            foreground (np.ndarray): Foreground image.
            background (np.ndarray): Background image.
            mask (np.ndarray): Binary mask for blending.
            sigma (float): Sigma value for Gaussian smoothing of the mask.

        Returns:
            np.ndarray: Blended image.
        """
        alpha_fore = gaussian_filter(mask.astype(float), sigma=sigma)
        if alpha_fore.max() > 0:
            alpha_fore = alpha_fore / alpha_fore.max()
        alpha_fore = np.expand_dims(alpha_fore, 2)
        img_blend = foreground * alpha_fore + background * (1 - alpha_fore)
        img_blend = img_blend.astype(np.uint8)
        return img_blend
    
    def _sample_param(self, param, is_int=True):
        """
        Sample a parameter value based on its type.
        If the parameter is a tuple, it samples from a range defined by the tuple.
        If the parameter is a single value, it returns that value.
        
        Args:
            param (int, float, tuple): Parameter to sample from.
            is_int (bool): If True, samples an integer; otherwise, samples a float.
        
        Returns:
            int, float: Sampled parameter value.
        """
        
        if isinstance(param, tuple):
            return np.random.randint(*param) if is_int else np.random.uniform(*param)

    
    # Generation pipeline
    def generate_vess_shape(self, build_grid=False):
        """
        Generate a synthetic image with vessel shapes, baground and foreground textures and its respective segmentation mask. In addition, it returns a dictionary with the parameters used to generate the image and metadata.
        
        Args:
            build_grid (bool): If True, returns a grid with the blended image and the mask.
        
        Returns:
            img_blend (np.ndarray): Blended image with vessel shapes.
            mask_bin (np.ndarray): Binary mask of the vessel shapes.
            metadata (dict): Dictionary with metadata about the generated image.
            grid (np.ndarray): Grid with the blended image and the mask (if build_grid is True).
        
        """
        
        # Ramdomly generate parameters for the current sample
        n_control_points = self._sample_param(self.n_control_points, is_int=True)
        max_vd = self._sample_param(self.max_vd, is_int=False)
        radius = self._sample_param(self.radius, is_int=True)
        num_curves = self._sample_param(self.num_curves, is_int=True)
        random_sigma = self._sample_param(self.sigma, is_int=False) # Random sigma for Gaussian smoothing
            
        self.vessel_geometry = VesselGeometry(
            self.image_size, n_control_points, max_vd, radius, num_curves, self.extra_space
        )
        
        mask = self.vessel_geometry.create_curves() # TODO: Passar os parâmetros ao invés de usar os padrões
        fg_img, bg_img, fg_file, bg_file = self.select_textures()
        fg_img = self.validate_image_dimensions(fg_img)
        bg_img = self.validate_image_dimensions(bg_img)
        fg_crop = self.random_crop(fg_img, self.crop_size)
        bg_crop = self.random_crop(bg_img, self.crop_size)
        mask_bin = (mask > 0).astype(np.uint8)
        
        img_blend = self.blend(fg_crop, bg_crop, mask_bin, sigma=random_sigma)
        
        metadata = {
            "foreground_texture": fg_file,
            "background_texture": bg_file,
            "mask_shape": mask.shape,
            "vessel_geometry_params": {
                "n_control_points": n_control_points,
                "max_vd": max_vd,
                "radius": radius,
                "num_curves": num_curves,
            },
        }
        
        if build_grid:
            # Create a grid for visualization
            grid = np.hstack(
                (
                    fg_crop,
                    bg_crop,
                    np.repeat(mask_bin[..., None] * 255, 3, axis=2),
                    img_blend,
                )
            )
            return img_blend, mask_bin, metadata, grid
        else:
            return img_blend, mask_bin, metadata
        
        
class VesselShapeDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for generating vessel shape images and corresponding binary masks.
    Args:
        vess_shape_generator: An object with a `generate_vess_shape()` method that returns (img_blend, mask_bin, metadata).
        n_samples (int, optional): Number of samples in the dataset. Default is 10.
        gray_scale (bool, optional): If True, images are converted to grayscale. Default is False.
    Methods:
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Generates a vessel shape image and its binary mask.
            Args:
                idx (int): Index of the sample (not used, as samples are generated on the fly).
            Returns:
                img_blend (Tensor): The vessel image as a torch.FloatTensor, shape [1, H, W] if gray_scale else [3, H, W], normalized to [0, 1].
                mask_bin (Tensor): The binary mask as a torch.FloatTensor, shape [1, H, W], values in {0, 1}.
                metadata (Any): Additional metadata returned by the generator.
    """

    def __init__(self, vess_shape_generator, n_samples=10, gray_scale=False):
        self.vess_shape_generator = vess_shape_generator
        self.n_samples = n_samples
        self.gray_scale = gray_scale
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        img_blend, mask_bin, metadata = self.vess_shape_generator.generate_vess_shape()

        if self.gray_scale:
            # Convert to grayscale and normalize
            img_pil = Image.fromarray(img_blend)
            img_pil = img_pil.convert("L")
            img_blend = np.array(img_pil)
            img_blend = torch.from_numpy(img_blend).unsqueeze(0).float() / 255.0  # [1, H, W]
        else:
            img_blend = torch.from_numpy(np.array(img_blend)).permute(2, 0, 1).float() / 255.0  # [3, H, W]

        mask_bin = torch.from_numpy(mask_bin).unsqueeze(0).float() / 255.0  # [1, H, W]
        mask_bin = (mask_bin > 0).float() # Convert to binary mask [0, 1]


        return img_blend, mask_bin, metadata
