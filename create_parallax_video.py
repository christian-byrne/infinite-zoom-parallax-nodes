import os
import json
import subprocess
import torch
import cv2
from PIL import Image
from moviepy.editor import VideoClip, ImageClip, CompositeVideoClip
from torchvision.transforms import ToTensor, ToPILImage

from .utils.tensor_utils import TensorImgUtils

from rich import print
from typing import Tuple

import folder_paths


class LayerFramesToParallaxVideoNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parallax_config": ("parallax_config",),
                "parallax_end_frame": ("IMAGE",),
            },
            "optional": {
                "static_objects_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("saved_video_path_str",)
    FUNCTION = "main"
    OUTPUT_NODE = True
    CATEGORY = "infinite/parallax"

    def main(
        self,
        parallax_config: str,  # json string
        parallax_end_frame: torch.Tensor,
        static_objects_mask: torch.Tensor = None,
    ) -> Tuple[str, ...]:
        self._set_config(parallax_config)

        # If frame count matches the num_iterations, create the video, otherwise do nothing
        cur_frame_ct = self.get_project_frame_ct()
        iterations_needed = self.parallax_config["num_iterations"]
        if cur_frame_ct < iterations_needed:
            msg = f"Frames in Project: {cur_frame_ct}/{iterations_needed}, ({iterations_needed - cur_frame_ct} more frames before video will be created)."
            print(msg)
            return (msg,)

        self.static_objects_mask = static_objects_mask
        self.set_layer_frame_ct()
        self.set_original_dimensions()

        return (self.composite_layer_videoclips(),)

    def set_original_dimensions(self):
        cur_image_path = self.try_get_start_img()
        if not cur_image_path:
            return
        img = Image.open(cur_image_path)
        self.original_width, self.original_height = img.size

    def set_layer_frame_ct(self):
        self.layer_frame_ct = len(
            [
                f
                for f in os.listdir(self.__get_parallax_proj_dirpath())
                if "layer0_" in f
            ]
        )

    def composite_layer_videoclips(self):
        layer_video_clips = []
        for i, layer in enumerate(self.parallax_config["layers"]):
            layer_videoclip = self.create_layer_videoclip(layer, i)
            if layer_videoclip:
                layer_videoclip = layer_videoclip.set_position((0, int(layer["top"])))
                layer_video_clips.append(layer_videoclip)

        video_composite = CompositeVideoClip(
            layer_video_clips,
            size=(self.original_width, self.original_height),
        )

        output_path = self.__get_parallax_proj_dirpath()
        video_ct = len([f for f in os.listdir(output_path) if "parallax_video" in f])
        video_path = os.path.join(output_path, f"parallax_video_{video_ct}.mp4")

        def write_video(path):
            video_composite.write_videofile(
                path,
                codec="libx264",
                fps=30,
                preset="slow",
                ffmpeg_params=(
                    [
                        "-crf",
                        "18",
                        "-b:v",
                        "2M",
                        "-pix_fmt",
                        "yuv420p",
                        "-profile:v",
                        "high",
                        "-vf",
                        "scale=1920:1080",
                    ]
                ),
                threads=12,
            )

        write_video(video_path)
        composte_path = os.path.join(
            output_path, f"parallax_video_{video_ct}_composite.mp4"
        )
        overlay_image_on_video(video_path, self.get_rgba_original_path(), composte_path)

        inputs_path = os.path.join(
            folder_paths.get_input_directory(), f"parallax_video_{video_ct}.mp4"
        )
        # write_video(inputs_path)
        if os.name == "nt":
            open_cmd = "start"
        else:
            open_cmd = "open"

        subprocess.Popen(
            [open_cmd, composte_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return inputs_path

    def save_rgba_version_original(self):
        og = self.get_original_image()
        og = torch.cat(
            [og, self.static_objects_mask.unsqueeze(0).unsqueeze(-1)], dim=-1
        )
        pil = ToPILImage()(TensorImgUtils.convert_to_type(og, "cHW"))
        path = os.path.join(self.__get_parallax_proj_dirpath(), "original_rgba.png")
        pil.save(path)
        self.rgba_original_path = path

    def get_rgba_original_path(self):
        if not hasattr(self, "rgba_original_path"):
            self.save_rgba_version_original()
        return self.rgba_original_path

    def get_original_image(self) -> torch.Tensor:
        output_path = self.__get_parallax_proj_dirpath()
        original_image = TensorImgUtils.convert_to_type(
            ToTensor()(Image.open(os.path.join(output_path, "original.png"))), "HWC"
        )
        return original_image

    def create_layer_videoclip(self, layer_config, layer_index):
        layer_velocity = layer_config["velocity"]
        max_height = self.original_height
        if layer_config["top"] >= max_height:
            return False
        if layer_config["top"] < 0:
            layer_config["top"] = 0
        if layer_config["bottom"] > max_height:
            layer_config["bottom"] = max_height
        layer_height = layer_config["bottom"] - layer_config["top"]

        output_path = self.__get_parallax_proj_dirpath()

        # Get the final width: number of steps * layer velocity
        added_width = int(self.layer_frame_ct * layer_velocity)
        final_width = added_width + self.original_width

        # Set the start frame slice by loading "original.png" and slicing by the layer's "top" and "bottom" values
        start_frame = Image.open(os.path.join(output_path, "original.png"))
        start_frame = start_frame.crop(
            (0, layer_config["top"], self.original_width, layer_config["bottom"])
        )

        stitched_image = Image.new("RGB", (final_width, layer_height))
        stitched_image.paste(start_frame, (0, 0))

        # Stitch each layer frame horizontally, with velocity offset
        # Go in reverse, because each frame is overlaid on the previous frame except for the velocity offset
        x_offset = final_width - self.original_width
        for i in range(self.layer_frame_ct - 1, -1, -1):
            layer_frame_path = os.path.join(
                output_path, f"layer{layer_index}_{i+1}.png"
            )
            layer_frame = Image.open(layer_frame_path)
            stitched_image.paste(layer_frame, (int(x_offset), 0))
            x_offset -= layer_velocity

        duration = float(self.layer_frame_ct) * (
            1.0 / float(self.parallax_config["fps"])
        )

        save_path = os.path.join(output_path, f"stitched_{layer_index}.png")
        stitched_image.save(save_path)
        image_clip = ImageClip(save_path)

        def make_frame(t):
            x = int(added_width * (t / duration))
            cur_frame = image_clip.get_frame(t)[:, x : x + self.original_width]
            return cur_frame

        return VideoClip(
            make_frame,
            duration=duration,
        )

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

    def _set_config(self, parallax_config: str) -> None:
        self.parallax_config = json.loads(parallax_config)

    def __get_proj_name(self):
        return f"infinite_parallax-{self.parallax_config['unique_project_name']}"

    def __project_dir_exists(self):
        return os.path.exists(self.__get_parallax_proj_dirpath())

    def __get_parallax_proj_dirpath(self):
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, self.__get_proj_name())
        return output_path

    @classmethod
    def IS_CHANGED(s, image):
        return LayerFramesToParallaxVideoNode.get_project_frame_ct()


def overlay_image_on_video(video_path, image_path, output_path):
    cap = cv2.VideoCapture(video_path)
    overlay = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load with alpha

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Choose your desired codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize overlay to match frame size if necessary
        overlay_resized = cv2.resize(overlay, (frame_width, frame_height))

        # Extract alpha channel
        alpha = overlay_resized[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha

        # Blend images using alpha channel
        for c in range(0, 3):
            frame[:, :, c] = (
                alpha * overlay_resized[:, :, c] + alpha_inv * frame[:, :, c]
            )

        out.write(frame)

    cap.release()
    out.release()
