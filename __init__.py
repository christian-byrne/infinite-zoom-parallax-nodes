"""
pyenv local 3.10.6
"""

from .nodes.preprocessors.size_match_node import *
from .nodes.preprocessors.layer_shifter import *
from .nodes.config_dicts.parallax_config import *
from .nodes.file_system.save_parallax_step import *
from .nodes.loaders.image_loaders.load_parallax_start import *
from .nodes.animation.create_parallax_video import *
from .nodes.loaders.image_loaders.load_most_recent import *
from .nodes.preprocessors.shrink_and_alpha_pad import *
from .nodes.loaders.image_loaders.load_random_img_pose_pair import *

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
WEB_DIRECTORY = "./web"
    

def _assign_class_mappings():
    """
    Assign the class mappings using the below class_mappings dictionary, whose
    items follow this format:
        "node category name" : [
            (
                "name displayed in the web ui",
                "name of the node in the backend" (also displayed in the subtext in the ui),
                reference to the node class
            )
        ]

    The category name will replace each node class's CATEGORY attribute.
    The backend name will be appended with the nobe_library_name.
    """

    node_library_name = "🔹Elimination"
    class_mappings = {
        "🖼️✨🎬 Infinite Parallax": [
            (
                "Layer Shifter for Parallax Outpainting",
                "Wrap-Shift & Mask Slices",
                LayerShifterNode,
            ),
            (
                "Parallax Config",
                "IP User Dict",
                ParallaxConfigDictNode,
            ),
            (
                "Save Parallax Frame",
                "Save IP Step Components",
                SaveParallaxStepNode,
            ),
            (
                "Load Parallax Frame",
                "Load IP Step Start",
                LoadParallaxStartNode,
            ),
            (
                "Create Parallax Video",
                "Create Composited Layer VClips",
                LayerFramesToParallaxVideoNode,
            ),
        ],
        "Infinite Zoom": [
            (
                "Shrink and Pad for Outpainting",
                "Shrink Inplace & Alpha Pad",
                ShrinkAndAlphaPadNode,
            )
        ],
        "Composite": [
            (
                "Paste Cutout on Base Image",
                "Composite Alpha Layer",
                CompositeCutoutOnBaseNode,
            ),
            (
                "Mask from RGB Image by Method",
                "Infer Alpha",
                AutoAlphaMaskNode,
            ),
            (
                "Size Match Images/Masks",
                "Resize Images to Match Size",
                SizeMatchNode,
            ),
        ],
        "Utils": [
            (
                "Load Most Recent Image in Folder",
                "Load File by Mtime",
                LoadMostRecentInFolderNode,
            ),
            (
                "Load Random Image-Pose Pair",
                "Load Image Pair by Method",
                LoadRandomImgPosePairNode,
            ),
        ],
        # "img2txt": [
        #     (
        #         "Image to Text - Auto Caption",
        #         "img2txt BLIP SalesForce Large",
        #         Img2TxtBlipNode,
        #     )
        # ],
    }

    for node_category, node_mapping in class_mappings.items():
        for display_name, node_name, node_class in node_mapping:
            # Strip and add the suffix to the node name
            node_name = f"{node_name.strip()}{node_library_name.strip()}"
            # Assign node class
            NODE_CLASS_MAPPINGS[node_name] = node_class
            # Assign display name
            NODE_DISPLAY_NAME_MAPPINGS[node_name] = display_name.strip()
            # Update the node class's CATEGORY
            node_class.CATEGORY = node_category.strip()


_assign_class_mappings()
