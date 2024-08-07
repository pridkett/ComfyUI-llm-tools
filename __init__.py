from .nodes.nodes import *

NODE_CLASS_MAPPINGS = {
    "OpenAI Vision": OpenAIVision,
    "Load Environment Variable": LoadEnvironmentVariable,
    "Image Dimensions": ImageDimensions,
    "Side by Side Images": SideBySideImage,
}