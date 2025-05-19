import os
import random
import numpy as np
from PIL import Image
import pandas as pd
from scipy.ndimage import gaussian_filter
from vessel_geometry import VesselGeometry



class VesselShape:
    """
    VesselShape is a class for generating synthetic vessel-shaped images by blending foreground and background textures using procedurally generated vessel masks.

    This class provides methods for:
        - Randomly sampling vessel geometry parameters (number of control points, vessel diameter, radius, number of curves, etc.).
        - Selecting and validating texture images for foreground and background, optionally using metadata to ensure label diversity.
        - Randomly cropping texture images to a specified size.
        - Validating image dimensions and sizes to ensure compatibility.
        - Blending textures using a binary vessel mask and Gaussian smoothing.
        - Generating vessel masks using a VesselGeometry object.
        - Producing metadata about the textures and vessel geometry used for each generated image.

    Attributes:
        n_control_points (tuple or int): Range or fixed value for the number of control points in the vessel geometry.
        max_vd (tuple or float): Range or fixed value for the maximum vessel diameter.
        radius (tuple or int): Range or fixed value for the vessel radius.
        num_curves (tuple or int): Range or fixed value for the number of vessel curves.
        image_size (int): Size of the generated vessel mask image.
        extra_space (int): Additional space around the vessel mask.
        crop_size (tuple): Size of the cropped region from texture images.
        texture_dir (str): Directory containing texture images.
        annotation_csv (str): Path to CSV file with texture metadata (image_id, label_id).
        sigma (tuple or float): Range or fixed value for Gaussian smoothing sigma.
        texture_metadata (pd.DataFrame or None): Loaded metadata from annotation_csv.
        texture_files (list): List of available texture image filenames.

    Methods:
        random_crop(img, crop_size): Randomly crops a region from the input image.
        validate_image_dimensions(img, target_channels): Ensures the image has the specified number of channels.
        validate_image_size(foreground_texture, background_texture, crop_size): Checks if both textures are large enough for cropping.
        select_textures(): Randomly selects two distinct texture images, ensuring label diversity if metadata is provided.
        blend(foreground, background, mask, sigma): Blends foreground and background images using a mask and Gaussian smoothing.
        _sample_param(param, is_int): Samples a parameter value from a tuple or returns the fixed value.
        generate_vess_shape(build_grid): Generates a synthetic vessel image, mask, metadata, and optionally a visualization grid.

    Example:
        vessel_shape = VesselShape(texture_dir='textures', annotation_csv='metadata.csv')
        img, mask, metadata = vessel_shape.generate_vess_shape()
    """
    def __init__(
        self,
        image_size=256,
        n_control_points=(2, 15),
        max_vd=(50.0, 150.0),
        radius=(1, 4),
        num_curves=(1, 15),
        extra_space=32,
        sigma=(1, 2),
        texture_dir=None,
        annotation_csv=None,
        crop_size=(256, 256)
    ):
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
        Randomly crops a region from the input image of the specified size.

        Parameters:
            img (numpy.ndarray): Input image to be cropped.
            crop_size (tuple of int, optional): Size of the crop (height, width). Defaults to (256, 256).

        Returns:
            numpy.ndarray: Cropped image region of the specified size.

        Raises:
            ValueError: If the input image is smaller than the desired crop size.
        """
        img_height, img_width = img.shape[:2]
        if img_height < crop_size[0] or img_width < crop_size[1]:
            raise ValueError("Image is too small for the desired crop size.")
        y = np.random.randint(0, img_height - crop_size[0] + 1)
        x = np.random.randint(0, img_width - crop_size[1] + 1)
        return img[y : y + crop_size[0], x : x + crop_size[1]]

    def validate_image_dimensions(self, img, target_channels=3):
        """
        Ensures that the input image has the specified number of channels.

        If the input image is grayscale (2D), it expands the dimensions and repeats the channel to match the target number of channels.
        If the image already has channels but does not match the target, raises a ValueError.

        Args:
            img (np.ndarray): Input image array.
            target_channels (int, optional): Desired number of channels. Defaults to 3.

        Returns:
            np.ndarray: Image array with the correct number of channels.

        Raises:
            ValueError: If the image has a different number of channels than target_channels and is not grayscale.
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
        Checks if both the foreground and background textures are at least as large as the specified crop size.

        Args:
            foreground_texture (np.ndarray): The foreground image as a NumPy array.
            background_texture (np.ndarray): The background image as a NumPy array.
            crop_size (tuple): The desired crop size as a tuple (height, width).

        Returns:
            bool: True if both textures are at least as large as crop_size in both dimensions, False otherwise.
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
        Randomly selects two distinct texture images (foreground and background) from the available texture files,
        ensuring that they have different labels if texture metadata is provided. The method continues sampling
        until it finds a valid pair of images that meet the following criteria:
            - The images are not the same file.
            - If texture metadata is available, the images have different label IDs.
            - Both images exist in the metadata (if provided).
            - The images pass the size validation check via `self.validate_image_size`.

        Returns:
            tuple: A tuple containing:
                - fg_img (np.ndarray): The selected foreground image as a NumPy array.
                - bg_img (np.ndarray): The selected background image as a NumPy array.
                - fg_file (str): The filename of the foreground image.
                - bg_file (str): The filename of the background image.
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
        Blends a foreground image onto a background image using a mask and Gaussian smoothing.

        Args:
            foreground (np.ndarray): The foreground image array.
            background (np.ndarray): The background image array.
            mask (np.ndarray): A binary or grayscale mask indicating the blending region.
            sigma (float, optional): Standard deviation for Gaussian kernel used to smooth the mask. Default is 1.

        Returns:
            np.ndarray: The blended image as an unsigned 8-bit integer array.
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
        Samples a parameter value, either as an integer or a float, depending on the input.

        If `param` is a tuple, returns a random value within the range specified by the tuple:
            - Uses `np.random.randint` if `is_int` is True (default), returning an integer.
            - Uses `np.random.uniform` if `is_int` is False, returning a float.
        If `param` is not a tuple, returns `param` as is.

        Args:
            param (Any or tuple): The parameter to sample from. If a tuple, should specify the range (min, max).
            is_int (bool, optional): Whether to sample as an integer (True) or float (False). Defaults to True.

        Returns:
            int, float, or Any: The sampled value or the original parameter if not a tuple.
        """
        if isinstance(param, tuple):
            return np.random.randint(*param) if is_int else np.random.uniform(*param)
        return param

    def generate_vess_shape(self, build_grid=False):
        """
        Generates a synthetic vessel shape image by blending foreground and background textures using a procedurally generated vessel mask.

        Args:
            build_grid (bool, optional): If True, returns a visualization grid containing the foreground crop, background crop, mask, and blended image. Defaults to False.

        Returns:
            tuple:
                - img_blend (np.ndarray): The blended image with vessel shape.
                - mask_bin (np.ndarray): Binary mask of the vessel shape.
                - metadata (dict): Dictionary containing metadata about the textures and vessel geometry parameters.
                - grid (np.ndarray, optional): Visualization grid (only if build_grid is True).
        """
        n_control_points = self._sample_param(self.n_control_points, is_int=True)
        max_vd = self._sample_param(self.max_vd, is_int=False)
        radius = self._sample_param(self.radius, is_int=True)
        num_curves = self._sample_param(self.num_curves, is_int=True)
        random_sigma = self._sample_param(self.sigma, is_int=False)
        self.vessel_geometry = VesselGeometry(
            self.image_size, n_control_points, max_vd, radius, num_curves, self.extra_space
        )
        mask = self.vessel_geometry.create_curves()
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


