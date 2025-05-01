import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import gaussian_filter
from vessel_geometry import VesselGeometry
import threading


class VesselShape:
    def __init__(
        self,
        image_size=256,
        n_control_points=3,
        max_vd=30,
        radius=3,
        num_curves=1,
        extra_space=32,
        texture_dir=None,
        annotation_csv=None,
        crop_size=(256, 256),
        n_samples=10, 
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
        """
        self.image_size = image_size
        self.crop_size = crop_size
        self.texture_dir = texture_dir
        self.annotation_csv = annotation_csv
        self.texture_metadata = None
        if annotation_csv is not None:
            self.texture_metadata = pd.read_csv(annotation_csv)
        self.texture_files = [
            f for f in os.listdir(texture_dir) if f.lower().endswith(".jpeg")
        ]
        self.vessel_geometry = VesselGeometry(
            image_size, n_control_points, max_vd, radius, num_curves, extra_space
        )
        self.n_samples = n_samples
        self._lock = threading.Lock()
        self._reset_epoch_state()

    def _reset_epoch_state(self):
        self._epoch_sample_count = 0
        self._epoch_indices = np.arange(self.n_samples)
        np.random.shuffle(self._epoch_indices)

    def reset_epoch(self):
        """Reseta o estado interno para uma nova época."""
        with self._lock:
            self._reset_epoch_state()

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
    
    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        Generate a synthetic sample for the given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (img_blend, mask_bin), where img_blend is the synthetic image and mask_bin is the binary mask label.
        """
        with self._lock:
            if self._epoch_sample_count >= self.n_samples:
                self._epoch_sample_count = 0
                raise IndexError("Todas as amostras da época já foram geradas.")
                
            self._epoch_sample_count += 1
        
        # TODO: Sortear os valores dos parâmetros das curvas
        mask = self.vessel_geometry.create_curves() # TODO: Passar os parâmetros ao invés de usar os padrões
        fg_img, bg_img, fg_file, bg_file = self.select_textures()
        fg_img = self.validate_image_dimensions(fg_img)
        bg_img = self.validate_image_dimensions(bg_img)
        fg_crop = self.random_crop(fg_img, self.crop_size)
        bg_crop = self.random_crop(bg_img, self.crop_size)
        mask_bin = (mask > 0).astype(np.uint8)
        random_sigma = np.random.uniform(1, 2)
        img_blend = self.blend(fg_crop, bg_crop, mask_bin, sigma=random_sigma)

        return img_blend, mask_bin

    def generate(self, output_dir=None, grid_dir=None, n_samples=None):
        """
        Generate synthetic images in memory without saving to disk.

        Args:
            output_dir (str, optional): Directory to save outputs (not used).
            grid_dir (str, optional): If provided, includes a visualization grid in the results.
            n_samples (int, optional): Number of samples to generate. Defaults to self.n_samples.

        Returns:
            list: List of dictionaries with generated images and associated information.
        """
        if n_samples is None:
            n_samples = self.n_samples
        results = []
        for i in tqdm(range(n_samples), desc="Gerando imagens"):
            img_blend, mask_bin = self[i]  # reutiliza __getitem__
            result = {
                "img_blend": img_blend,
                "mask_bin": mask_bin,
            }
            if grid_dir:
                # Para visualização, pode-se criar um grid simples
                grid = np.hstack(
                    (
                        img_blend,
                        np.repeat(mask_bin[..., None] * 255, 3, axis=2),
                    )
                )
                result["grid"] = grid
            results.append(result)
            
        return results
