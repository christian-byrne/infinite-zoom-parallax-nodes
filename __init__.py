from .parallax_config import ParallaxConfigDictNode
from .save_parallax_step import SaveParallaxStepNode
from .create_parallax_video import LayerFramesToParallaxVideoNode
from .layer_preprocessors.layer_shifter import LayerShifterNode
from .layer_preprocessors.shrink_and_alpha_pad import ShrinkAndAlphaPadNode
from .image_loaders.load_most_recent import LoadMostRecentInFolderNode
from .image_loaders.load_random_img_pose_pair import LoadRandomImgPosePairNode
from .image_loaders.load_parallax_start import LoadParallaxStartNode

NODE_CLASS_MAPPINGS = {
    "Layer Shifter for Parallax Outpainting": LayerShifterNode,
    "Parallax Config": ParallaxConfigDictNode,
    "Save Parallax Frame": SaveParallaxStepNode,
    "Load Parallax Frame": LoadParallaxStartNode,
    "Create Parallax Video": LayerFramesToParallaxVideoNode,
    "Shrink and Pad for Outpainting": ShrinkAndAlphaPadNode,
    "Load Most Recent Image in Folder": LoadMostRecentInFolderNode,
    "Load Random Image-Pose Pair": LoadRandomImgPosePairNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Layer Shifter for Parallax Outpainting": "Wrap-Shift & Mask Slices",
    "Parallax Config": "IP User Dict",
    "Save Parallax Frame": "Save IP Step Components",
    "Load Parallax Frame": "Load IP Step Start",
    "Create Parallax Video": "Create Composited Layer VClips",
    "Shrink and Pad for Outpainting": "Shrink Inplace & Alpha Pad",
    "Load Most Recent Image in Folder": "Load File by Mtime",
    "Load Random Image-Pose Pair": "Load Image Pair by Method",
}
# WEB_DIRECTORY = "./web"
