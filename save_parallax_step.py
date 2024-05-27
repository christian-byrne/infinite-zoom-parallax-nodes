"""Doc Strings automatically generated

pyenv local 3.10.6"""

import torch
from torchvision import transforms
import os
import json

from typing import Tuple


try:
    from ...utils.tensor_utils import TensorImgUtils
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.tensor_utils import TensorImgUtils


class SaveParallaxStepNode:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_path_str",)
    FUNCTION = "main"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "parallax_config": ("parallax_config",),
            },
        }

    def main(
        self,
        input_image: torch.Tensor,  # [Batch_n, H, W, 3-channel]
        parallax_config: str,  # json string
    ) -> Tuple[str, ...]:

        # parallax_config json string to dict
        parallax_config = json.loads(parallax_config)

        # get path of node dir
        this_path = os.path.dirname(os.path.abspath(__file__))
        # get and create path to the parallax project dir
        output_path = os.path.join(this_path, parallax_config["unique_project_name"])
        os.makedirs(output_path, exist_ok=True)

        def last_num(str):
            if len(str) == 0:
                return 0
            if str[-1].isdigit():
                return int(str[-1])
            return last_num(str[:-1])

        # get current index based on highest index of a file in the dir
        current_index = 0
        for file in os.listdir(output_path):
            if file.endswith(".png"):
                index = last_num(file)
                if index > current_index:
                    current_index = index
        current_index += 1

        to_pil = transforms.ToPILImage()

        # squeeze batch dimension
        input_image = TensorImgUtils.test_squeeze_batch(input_image)

        # start by saving the entire image - to serve as the start image of the next step
        next_step_start_image = to_pil(
            TensorImgUtils.convert_to_type(input_image, "CHW")
        )
        save_path = os.path.join(output_path, f"start_{current_index}.png")
        next_step_start_image.save(save_path)

        max_height = input_image.shape[0]
        file_paths = []
        for layer_index, layer in enumerate(parallax_config["layers"]):
            if layer["height"] == 0 or layer["velocity"] == 0:
                continue

            layer_image = input_image[layer["top"] : layer["bottom"], :, :]
            layer_image = TensorImgUtils.convert_to_type(layer_image, "CHW")

            pil_image = to_pil(layer_image)
            save_path = os.path.join(
                output_path, f"layer{layer_index}_{current_index}.png"
            )

            file_paths.append(save_path)
            pil_image.save(save_path)

            if layer["bottom"] > max_height:
                print(
                    f"LayerSaveNode: layer['bottom'] > max_height: {layer['bottom']} > {max_height}"
                )
                break

        return (json.dumps(file_paths),)
