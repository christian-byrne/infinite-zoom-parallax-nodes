"""Doc Strings automatically generated

pyenv local 3.10.6"""

import os
from PIL import Image, ImageOps
import numpy as np
import torch
import re
from typing import Tuple
import random
import folder_paths
from termcolor import colored

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


class LoadRandomImgPosePairNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "select_method": (
                    [
                        "random",
                        "random no repeats",
                        "most recent",
                        "iterate forward",
                        "iterate backward",
                    ],
                ),
                "pose_file_selector_phrase": (
                    "STRING",
                    {
                        "default": "pose",
                        "multiline": False,
                    },
                ),
                "img_file_selector_phrase": (
                    "STRING",
                    {
                        "default": "preview",
                        "multiline": False,
                    },
                ),
            },
            "optional": {
                "folder_abs_path": (
                    "STRING",
                    {
                        "default": "/absolute/path/to/folder",
                        "multiline": False,
                    },
                ),
                "folder": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Custom Folder",
                        "label_off": "Default Input Folder",
                    },
                ),
                "file_type": (
                    PICTURE_EXTENSION_LIST + VIDEO_EXTENSION_LIST + TEXT_EXTENSION_LIST,
                    {"default": ".png"},
                ),
                "alternate_file_type": (
                    PICTURE_EXTENSION_LIST
                    + VIDEO_EXTENSION_LIST
                    + TEXT_EXTENSION_LIST
                    + [""],
                    {"default": ".jpg"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "STRING")
    FUNCTION = "main"
    RETURN_NAMES = (
        "image",
        "image_mask",
        "pose_image",
        "pose_image_mask",
        "index_string",
    )

    def main(
        self,
        select_method: str,  # random, random no repeats, most recent, iterate forward, iterate backward
        pose_file_selector_phrase: str,  # phrase to select pose file
        img_file_selector_phrase: str,  # phrase to select image file
        folder_abs_path: str,  # absolute path to folder
        folder: bool,  # Custom Folder, Default Input Folder
        file_type: str,
        alternate_file_type: str,
    ) -> Tuple[torch.Tensor, ...]:
        self.used_indices = []

        self.valid_match_extensions = set()
        for match_ext in [file_type, alternate_file_type]:
            if match_ext:
                self.valid_match_extensions.add(match_ext)
        if "jpg" in self.valid_match_extensions:
            self.valid_match_extensions.add("jpeg")
        if "jpeg" in self.valid_match_extensions:
            self.valid_match_extensions.add("jpg")

        if folder_abs_path == "/absolute/path/to/folder" or not folder:
            self.folder_abs_path = folder_paths.get_input_directory()
        else:
            self.folder_abs_path = folder_abs_path
        self.folder_abs_path = self.__try_get_dir(self.folder_abs_path)

        candidate_files = self.__get_candidate_files()
        pose_candidates = [f for f in candidate_files if pose_file_selector_phrase in f]
        img_candidates = [f for f in candidate_files if img_file_selector_phrase in f]

        if select_method == "random":
            target_index = random.randint(
                0, min(len(pose_candidates), len(img_candidates)) - 1
            )
            self.used_indices.append(target_index)
        elif select_method == "random no repeats":
            target_index = random.choice(
                [
                    i
                    for i in range(min(len(pose_candidates), len(img_candidates)))
                    if i not in self.used_indices
                ]
            )
            self.used_indices.append(target_index)
        elif select_method == "most recent":
            target_index = -1
            self.used_indices.append(target_index)
        elif select_method == "iterate forward":
            target_index = len(self.used_indices)
            self.used_indices.append(target_index)
        elif select_method == "iterate backward":
            target_index = len(self.used_indices)
            self.used_indices.append(target_index)
            target_index = -1 * target_index

        pose_path = os.path.join(self.folder_abs_path, pose_candidates[target_index])
        img_path = os.path.join(self.folder_abs_path, img_candidates[target_index])

        def extract_index(filename):
            pattern = r"(\d+)(?=\D*$)"
            matches = re.findall(pattern, filename)
            if matches:
                return int(matches[0])
            return False

        # Try to match any indices found in filenames if the indices in the folders lead to a disalignment.
        try:
            pose_image_fi_index = extract_index(pose_candidates[target_index])
            img_image_fi_index = extract_index(img_candidates[target_index])
            if pose_image_fi_index and img_image_fi_index:
                if pose_image_fi_index != img_image_fi_index:
                    attempt = img_path.replace(
                        str(img_image_fi_index), str(pose_image_fi_index)
                    )
                    if os.path.exists(attempt):
                        img_path = attempt
        except IndexError:
            pass

        image, image_mask = self.load_image(img_path)
        pose_image, pose_image_mask = self.load_image(pose_path)

        return (image, image_mask, pose_image, pose_image_mask, f"{target_index}")

    def __match_file(self, filename):
        return any([filename.endswith(ext) for ext in self.valid_match_extensions])

    def load_image(self, img):
        img = Image.open(img)

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

        return rgb_image, mask

    def __try_get_dir(self, path):
        if not os.path.exists(path):
            comfy_root = (
                os.path.dirname(os.path.abspath(__file__)).split("ComfyUI")[0]
                + "ComfyUI"
            )
            try_path = os.path.join(comfy_root, path)
            if os.path.exists(try_path):
                path = try_path
            else:
                try_path = os.path.join(comfy_root, "input", path)
                if os.path.exists(try_path):
                    path = try_path
                else:
                    os.makedirs(path, exist_ok=True)
        return path

    def __get_candidate_files(self):
        candidate_files = [
            f for f in os.listdir(self.folder_abs_path) if self.__match_file(f)
        ]
        try:
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

    def get_cur_indices_ct(self):
        return len(self.used_indices)

    @classmethod
    def IS_CHANGED(s, image):
        """
        Update if either:
            - Using an iteratoring selection method
            - Using random selection method
            - Using most recent selection method and a new image has been added to the folder
        """
        return (
            LoadRandomImgPosePairNode.get_candidate_files_ct()
            + LoadRandomImgPosePairNode.get_cur_indices_ct()
        )
