from .parallax_config import ParallaxConfigDictNode
from .save_parallax_step import SaveParallaxStepNode
from .create_parallax_video import LayerFramesToParallaxVideoNode
from .layer_preprocessors.layer_shifter import LayerShifterNode
from .layer_preprocessors.shrink_and_alpha_pad import ShrinkAndAlphaPadNode
from .image_loaders.load_parallax_start import LoadParallaxStartNode

NODE_CLASS_MAPPINGS = {
    "Layer Shifter for Parallax Outpainting": LayerShifterNode,
    "Parallax Config": ParallaxConfigDictNode,
    "Save Parallax Frame": SaveParallaxStepNode,
    "Load Parallax Frame": LoadParallaxStartNode,
    "Create Parallax Video": LayerFramesToParallaxVideoNode,
    "Shrink and Pad for Outpainting": ShrinkAndAlphaPadNode,
}