import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import gaussian_filter
from vessel_geometry import VesselGeometry


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
    ):
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

    def random_crop(self, img, crop_size=(256, 256)):
        img_height, img_width = img.shape[:2]
        if img_height < crop_size[0] or img_width < crop_size[1]:
            raise ValueError("Image is too small for the desired crop size.")
        y = np.random.randint(0, img_height - crop_size[0] + 1)
        x = np.random.randint(0, img_width - crop_size[1] + 1)
        return img[y : y + crop_size[0], x : x + crop_size[1]]

    def validate_image_dimensions(self, img, target_channels=3):
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, target_channels, axis=-1)
        elif img.shape[2] != target_channels:
            raise ValueError(
                f"Image has {img.shape[2]} channels, but {target_channels} channels are required."
            )
        return img

    def validate_image_size(self, foreground_texture, background_texture, crop_size):
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
        alpha_fore = gaussian_filter(mask.astype(float), sigma=sigma)
        if alpha_fore.max() > 0:
            alpha_fore = alpha_fore / alpha_fore.max()
        alpha_fore = np.expand_dims(alpha_fore, 2)
        img_blend = foreground * alpha_fore + background * (1 - alpha_fore)
        img_blend = img_blend.astype(np.uint8)
        return img_blend

    def generate(self, output_dir=None, grid_dir=None, n_samples=10):
        """
        Gera imagens sintéticas em memória, sem salvar em disco.
        Retorna uma lista de dicionários com as imagens e informações associadas.
        """
        results = []
        for i in tqdm(range(n_samples), desc="Gerando imagens"):
            mask = self.vessel_geometry.create_curves()
            fg_img, bg_img, fg_file, bg_file = self.select_textures()
            fg_img = self.validate_image_dimensions(fg_img)
            bg_img = self.validate_image_dimensions(bg_img)
            fg_crop = self.random_crop(fg_img, self.crop_size)
            bg_crop = self.random_crop(bg_img, self.crop_size)
            mask_bin = (mask > 0).astype(np.uint8)
            random_sigma = np.random.uniform(1, 2)
            img_blend = self.blend(fg_crop, bg_crop, mask_bin, sigma=random_sigma)
            # Não salva mais as imagens em disco
            result = {
                "img_blend": img_blend,
                "fg_crop": fg_crop,
                "bg_crop": bg_crop,
                "mask_bin": mask_bin,
                "fg_file": fg_file,
                "bg_file": bg_file,
            }
            if grid_dir:
                grid = np.hstack(
                    (
                        fg_crop,
                        bg_crop,
                        np.repeat(mask_bin[..., None] * 255, 3, axis=2),
                        img_blend,
                    )
                )
                result["grid"] = grid
            results.append(result)
        return results
