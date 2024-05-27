"""Doc Strings automatically generated

pyenv local 3.10.6"""

import torch
from torchvision import transforms
from termcolor import colored
from typing import Tuple


try:
    from ...utils.tensor_utils import TensorImgUtils
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.tensor_utils import TensorImgUtils


class ShrinkAndAlphaPadNode:
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
    )
    RETURN_NAMES = ("shrunk_padded_image", "mask_for_outpainting")
    FUNCTION = "main"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
            },
            "optional": {
                "velocity_left": ("INT",{
                    "default": 128,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                }),
                "velocity_top": ("INT",{
                    "default": 128,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                }),
                "velocity_right": ("INT",{
                    "default": 128,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                }),
                "velocity_bottom": ("INT",{
                    "default": 128,
                    "min": 0,
                    "max": 512,
                    "step": 1,
                }),
                "feather_px": ("INT",{
                    "default": 0,
                    "min": 0,
                    "max": 64,
                    "step": 1,
                    "display": "slider"
                }),
            },
        }

    def main(
        self,
        input_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        velocity_left: int,
        velocity_top: int,
        velocity_right: int,
        velocity_bottom: int,
        feather_px: int,
    ) -> Tuple[torch.Tensor, ...]:

        self.TRANSPARENT = 0
        self.OPAQUE = 1

        # squeeze batch dimension
        input_image = TensorImgUtils.test_squeeze_batch(input_image)

        # Create mask Tensor.
        mask_tensor = torch.ones(
            (input_image.shape[0], input_image.shape[1]), dtype=torch.uint8
        )

        if (velocity_left + velocity_right) % 2 != 0:
            velocity_left += 1
        if (velocity_top + velocity_bottom) % 2 != 0:
            velocity_top += 1
        # Total pixels removed horizontally
        remove_horizontal = velocity_left + velocity_right
        # Total pixels removed vertically
        remove_vertical = velocity_top + velocity_bottom

        # Shrink the image as close to the remove specifications as possible while maintaining aspect ratio
        target_width = input_image.shape[1] - remove_horizontal
        width_required_ratio = target_width / input_image.shape[1]
        target_height = input_image.shape[0] - remove_vertical
        height_required_ratio = target_height / input_image.shape[0]
        target_ratio = min(width_required_ratio, height_required_ratio)

        new_width = int(input_image.shape[1] * target_ratio)
        new_height = int(input_image.shape[0] * target_ratio)

        # Shrink the image
        shrunk_image = transforms.Resize((new_height, new_width))(
            TensorImgUtils.convert_to_type(input_image, "CHW")
        )
        shrunk_image = TensorImgUtils.convert_to_type(shrunk_image, "HWC")

        # Composite the shrunk image onto the previous image (to provide inpainting context - better than a black border in some setups)
        # Paste such that we re-incorporate uneven velocities on the same plane
        if velocity_left > velocity_right or remove_vertical > remove_horizontal:
            left_margin = velocity_left 
            right_margin = (input_image.shape[1] - shrunk_image.shape[1]) - velocity_left
        else:
            right_margin = velocity_right
            left_margin = (input_image.shape[1] - shrunk_image.shape[1]) - velocity_right
        print(colored(f"left_margin: {left_margin}, right_margin: {right_margin}", "light_green"))
        if velocity_top > velocity_bottom or remove_horizontal > remove_vertical:
            top_margin = velocity_top
            bottom_margin = (input_image.shape[0] - shrunk_image.shape[0]) - velocity_top
        else:
            bottom_margin = velocity_bottom
            top_margin = (input_image.shape[0] - shrunk_image.shape[0]) - velocity_bottom
        print(colored(f"top_margin: {top_margin}, bottom_margin: {bottom_margin}", "light_green"))
        
        # To simply center the image, use the following line instead
        # input_image[remove_top:-remove_bottom, remove_left:-remove_right, :] = shrunk_image

        # Make copy of input image
        output_image = input_image.clone()
        output_image[top_margin:-bottom_margin, left_margin:-right_margin, :] = shrunk_image
        
        # Apply feathering to the mask only
        top_margin = max(0, top_margin + feather_px) // 2
        bottom_margin = max(0, bottom_margin + feather_px // 2)
        left_margin = max(0, left_margin + feather_px // 2)
        right_margin = max(0, right_margin + feather_px // 2)
        mask_tensor[top_margin:-bottom_margin, left_margin:-right_margin] = self.TRANSPARENT

        output_image = TensorImgUtils.test_unsqueeze_batch(output_image)
        mask_tensor = TensorImgUtils.test_unsqueeze_batch(mask_tensor)

        return (output_image, mask_tensor)
