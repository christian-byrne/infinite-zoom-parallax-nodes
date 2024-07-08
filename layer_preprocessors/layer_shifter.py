import os
import torch
import json
from PIL import Image
from torchvision import transforms

from ..utils.tensor_utils import TensorImgUtils

import folder_paths

from rich import print
from typing import Tuple


class LayerShifterNode:
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("shifted_image", "shifted_mask", "inpaint_target")
    FUNCTION = "main"
    CATEGORY = "infinite/parallax"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "parallax_config": ("parallax_config",),
            },
            "optional": {
                "static_objects_mask": ("MASK",),
                "object_mask_1": ("MASK",),
                "object_mask_2": ("MASK",),
                "object_mask_3": ("MASK",),
            },
        }

    def main(
        self,
        input_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        parallax_config: str,  # json string
        static_objects_mask: torch.Tensor = None,
        object_mask_1: torch.Tensor = None,
        object_mask_2: torch.Tensor = None,
        object_mask_3: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, ...]:

        self.TRANSPARENT = 0
        self.OPAQUE = 1

        if static_objects_mask is not None:
            self.static_objects_mask = static_objects_mask.squeeze(0)
        if object_mask_1 is not None:
            self.object_mask_1 = object_mask_1
        if object_mask_2 is not None:
            self.object_mask_2 = object_mask_2
        if object_mask_3 is not None:
            self.object_mask_3 = object_mask_3

        self.parallax_config = json.loads(parallax_config)
        original_image = TensorImgUtils.convert_to_type(
            transforms.ToTensor()(self.get_original_image()), "BHWC"
        )
        self.input_height, self.input_width = TensorImgUtils.height_width(input_image)
        mask_tensor = torch.zeros(
            (self.input_height, self.input_width), dtype=torch.float32
        )

        for layer in self.parallax_config["layers"]:
            if layer["height"] == 0 or layer["velocity"] == 0:
                continue

            velocity = round(float(layer["velocity"]))

            input_image = self.shift_horizontal_slice(
                input_image,
                layer["top"],
                (
                    layer["bottom"]
                    if layer["bottom"] < self.input_height
                    else self.input_height
                ),
                velocity,
            )

            mask_tensor = self.add_mask_to_shifted_region(
                mask_tensor,
                layer["top"],
                (
                    layer["bottom"]
                    if layer["bottom"] < self.input_height
                    else self.input_height
                ),
                velocity,
            )

            # Shift the static objects mask
            if self.static_objects_mask is not None:
                self.static_objects_mask[layer["top"] : layer["bottom"], ...] = (
                    torch.roll(
                        self.static_objects_mask[layer["top"] : layer["bottom"], ...],
                        -velocity,
                        dims=1,
                    )
                )

            # Add the shifted static objects mask to the mask tensor.
            # For example, if there is a window panel in the static objects mask, the panel will be shifted
            # during layer shifting - to fix, we make any parts of the static object that were shifted transparent
            # so that inpainting takes care of them iteratively.
            if self.static_objects_mask is not None:
                layer_bot = (
                    layer["bottom"]
                    if layer["bottom"] < self.input_height
                    else self.input_height
                )
                mask_tensor[layer["top"] : layer_bot] = torch.max(
                    mask_tensor[layer["top"] : layer_bot],
                    1 - self.static_objects_mask[layer["top"] : layer_bot],
                )

            if layer["bottom"] > self.input_height:
                break

        print(f"input image shape: {input_image.shape}")
        print(f"original image shape: {original_image.shape}")
        print(f"mask tensor shape: {mask_tensor.shape}")
        print(f"static objects mask shape: {self.static_objects_mask.shape}")
        inpaint_target = torch.where(
            self.static_objects_mask.unsqueeze(-1) == self.OPAQUE, original_image, input_image
        )

        return (
            TensorImgUtils.test_unsqueeze_batch(input_image),
            mask_tensor,
            TensorImgUtils.test_unsqueeze_batch(inpaint_target),
        )

    def _set_config(self, parallax_config: str) -> None:
        self.parallax_config = json.loads(parallax_config)

    def __get_proj_name(self):
        return f"infinite_parallax-{self.parallax_config['unique_project_name']}"

    def __get_parallax_proj_dirpath(self):
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, self.__get_proj_name())
        return output_path

    def get_original_image(self):
        output_path = self.__get_parallax_proj_dirpath()
        return Image.open(os.path.join(output_path, "original.png"))

    def add_mask_to_shifted_region(self, mask_tensor, start_row, end_row, shift_pixels):
        mask_tensor[start_row:end_row, mask_tensor.shape[-1] - shift_pixels :] = (
            self.OPAQUE
        )
        return mask_tensor

    def shift_horizontal_slice(
        self,
        image_tensor: torch.Tensor,
        start_row: int,
        end_row: int,
        shift_pixels: float,
    ):
        width_dim = TensorImgUtils.infer_hw_axis(image_tensor)[1]
        image_tensor[:, start_row:end_row, ...] = torch.roll(
            image_tensor[:, start_row:end_row, ...], -shift_pixels, dims=width_dim
        )

        return image_tensor
