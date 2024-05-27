"""Doc Strings automatically generated

pyenv local 3.10.6"""

import os
import json
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms

from typing import Tuple, Union


try:
    from ....utils.tensor_utils import TensorImgUtils
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.tensor_utils import TensorImgUtils


class LoadParallaxStartNode:
    def __init__(self):
        self.start_frame_keyword = "start"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parallax_config": ("parallax_config",),
            },
            "optional": {
                "start_image": ("IMAGE",),
            },
        }

    CATEGORY = "image"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("parallax_start_frame", "optional_mask")
    FUNCTION = "load_image"

    def load_image(
        self,
        parallax_config: str,  # json string
        start_image: Union[torch.Tensor, None],  # [Batch_n, H, W, 3-channel]
    ) -> Tuple[torch.Tensor, ...]:

        self.__set_config(parallax_config)

        # Use most recent frame if it exists (this isn't the first iteration of the project)
        if self.get_project_frame_ct() == 0:
            start_img_pil = transforms.ToPILImage()(
                TensorImgUtils.convert_to_type(start_image, "CHW")
            )
            # Create the project directory, and save the loaded image as the start frame
            output_path = self.__get_parallax_proj_dirpath()
            os.makedirs(output_path, exist_ok=True)
            start_img_pil.save(os.path.join(output_path, "original.png"))

            mask = torch.zeros(
                (start_img_pil.height, start_img_pil.width),
                dtype=torch.float32,
                device="cpu",
            )
            mask = mask.unsqueeze(0)
            start_image = TensorImgUtils.convert_to_type(start_image, "BHWC")
            return (start_image, mask)

        img = Image.open(self.try_get_start_img())

        # If the image has exif data, rotate it to the correct orientation and remove the exif data
        img_raw = ImageOps.exif_transpose(img)

        # If in 32-bit mode, normalize the image appropriately
        if img_raw.mode == "I":
            img_raw = img.point(lambda i: i * (1 / 255))

        # If image is rgba, create mask
        if "A" in img_raw.getbands():
            mask = np.array(img_raw.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros(
                (img_raw.height, img_raw.width), dtype=torch.float32, device="cpu"
            )
        mask = mask.unsqueeze(0)  # Add a batch dimension to mask

        # Convert the image to RGB, TODO: should be able to handle rgba throughout the pipeline easily
        rgb_image = img_raw.convert("RGB")
        # Normalize the image's rgb values to {x | x ∈ float32, 0 <= x <= 1}
        rgb_image = np.array(rgb_image).astype(np.float32) / 255.0
        # Convert the image to a tensor (torch.from_numpy gives a tensor with the format of [H, W, C])
        rgb_image = torch.from_numpy(rgb_image)[
            None,
        ]  # Add a batch dimension, new format is [B, H, W, C]

        return (rgb_image, mask)

    def get_project_frame_ct(self):
        if not self.__project_dir_exists():
            return 0
        return len(
            [f for f in os.listdir(self.__get_parallax_proj_dirpath()) if "start" in f]
        )

    def try_get_start_img(self):

        output_path = self.__get_parallax_proj_dirpath()
        cur_image_path = False
        if os.path.exists(output_path):
            start_images = [f for f in os.listdir(output_path) if "start" in f]
            if len(start_images) > 0:
                start_images.sort()
                cur_image_path = os.path.join(output_path, start_images[-1])
        return cur_image_path

    def __set_config(self, parallax_config: str) -> None:
        self.parallax_config = json.loads(parallax_config)

    def __get_proj_name(self):
        return self.parallax_config["unique_project_name"]

    def __project_dir_exists(self):
        return os.path.exists(self.__get_parallax_proj_dirpath())

    def __get_parallax_proj_dirpath(self):
        node_dir = os.path.dirname(os.path.abspath(__file__)).split(
            "elimination-nodes"
        )[0]
        node_dir = os.path.join(node_dir, "elimination-nodes", "nodes", "file_system")
        output_path = os.path.join(node_dir, self.__get_proj_name())
        return output_path

    @classmethod
    def IS_CHANGED(s, image):
        return LoadParallaxStartNode.get_project_frame_ct()
