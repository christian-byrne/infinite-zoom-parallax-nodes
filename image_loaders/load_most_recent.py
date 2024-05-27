"""Doc Strings automatically generated

pyenv local 3.10.6"""

import os
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple, Union

try:
    from ....utils.tensor_utils import TensorImgUtils
    from ....constants import (
        PICTURE_EXTENSION_LIST,
        VIDEO_EXTENSION_LIST,
        TEXT_EXTENSION_LIST,
    )
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.tensor_utils import TensorImgUtils
    from constants import (
        PICTURE_EXTENSION_LIST,
        VIDEO_EXTENSION_LIST,
        TEXT_EXTENSION_LIST,
    )


class LoadMostRecentInFolderNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_abs_path": (
                    "STRING",
                    {
                        "default": "/absolute/path/to/folder",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "start_image": ("IMAGE",),
                "file_type": (
                    PICTURE_EXTENSION_LIST + VIDEO_EXTENSION_LIST + TEXT_EXTENSION_LIST,
                    {"default": ".png"},
                ),
                "load": (
                    "BOOLEAN",
                    {"default": True, "label_on": "Most Recent", "label_off": "Oldest"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    FUNCTION = "load_image"
    RETURN_NAMES = ("IMAGE", "MASK", "folder_path_string")

    def load_image(
        self,
        folder_abs_path: str,  # absolute path to folder
        start_image: Union[torch.Tensor, None],  # [Batch_n, H, W, 3-channel]
        file_type: str,
        load: bool,
    ) -> Tuple[torch.Tensor, ...]:
        self.folder_abs_path = folder_abs_path
        self.file_type = file_type
        self.target_index = -1 if load else 0

        candidate_files = self.__get_candidate_files()

        print(f"Found {len(candidate_files)} files in {self.folder_abs_path}\n")
        for i, f in enumerate(candidate_files):
            print(f"File {i}: {f}")
        print("\n")

        # If no files are found, load the start_image instead
        if len(candidate_files) == 0:
            start_img_pil = transforms.ToPILImage()(
                TensorImgUtils.convert_to_type(start_image, "CHW")
            )
            # Create the project directory, and save the loaded image as the start frame
            output_path = self.folder_abs_path
            os.makedirs(output_path, exist_ok=True)
            start_img_pil.save(os.path.join(output_path, "z9original9.png"))

            mask = torch.zeros(
                (start_img_pil.height, start_img_pil.width),
                dtype=torch.float32,
                device="cpu",
            )
            mask = mask.unsqueeze(0)
            start_image = TensorImgUtils.convert_to_type(start_image, "BHWC")
            return (start_image, mask, self.folder_abs_path)

        img = Image.open(
            os.path.join(self.folder_abs_path, candidate_files[self.target_index])
        )

        # If the image has exif data, rotate it to the correct orientation and remove the exif data.
        img_raw = ImageOps.exif_transpose(img)

        # If in 32-bit mode, normalize the image appropriately.
        if img_raw.mode == "I":
            img_raw = img.point(lambda i: i * (1 / 255))

        # If image is rgba, create mask.
        if "A" in img_raw.getbands():
            mask = np.array(img_raw.getchannel("A")).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            # otherwise create a blank mask.
            mask = torch.zeros(
                (img_raw.height, img_raw.width), dtype=torch.float32, device="cpu"
            )
        mask = mask.unsqueeze(0)  # Add a batch dimension to mask

        # Convert the image to RGB.
        rgb_image = img_raw.convert("RGB")
        # Normalize the image's rgb values to {x | x ∈ float32, 0 <= x <= 1}
        rgb_image = np.array(rgb_image).astype(np.float32) / 255.0
        # Convert the image to a tensor (torch.from_numpy gives a tensor with the format of [H, W, C])
        rgb_image = torch.from_numpy(rgb_image)[
            None,
        ]  # Add a batch dimension, new format is [B, H, W, C]

        return (rgb_image, mask, self.folder_abs_path)

    def __get_candidate_files(self):
        if not os.path.exists(self.folder_abs_path):
            comfy_root = (
                os.path.dirname(os.path.abspath(__file__)).split("ComfyUI")[0]
                + "ComfyUI"
            )
            try_path = os.path.join(comfy_root, self.folder_abs_path)
            if os.path.exists(try_path):
                self.folder_abs_path = try_path
            else:
                try_path = os.path.join(comfy_root, "input", self.folder_abs_path)
                if os.path.exists(try_path):
                    self.folder_abs_path = try_path
                else:
                    os.makedirs(self.folder_abs_path, exist_ok=True)

        candidate_files = [
            f for f in os.listdir(self.folder_abs_path) if self.file_type in f
        ]
        try:
            for f in candidate_files:
                print(f"File: {f}, Modified: {os.path.getmtime(os.path.join(self.folder_abs_path, f))}")
            candidate_files = sorted(
                candidate_files,
                key=lambda f: os.path.getmtime(os.path.join(self.folder_abs_path, f)),
            )
        except OSError:
            print("Could not get the modified time of the files.")
            print(OSError)
            candidate_files = sorted(candidate_files)

        return candidate_files

    def get_candidate_files_ct(self):
        return len([f for f in os.listdir(self.folder_abs_path) if self.file_type in f])

    @classmethod
    def IS_CHANGED(s, image):
        return LoadMostRecentInFolderNode.get_candidate_files_ct()
