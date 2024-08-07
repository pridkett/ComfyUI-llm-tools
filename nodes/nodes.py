import base64
import torch
import numpy as np
import requests
import os
from PIL import Image
from io import BytesIO
from typing import Tuple, List

DEFAULT_PROMPT = """Describe the image as a prompt that an AI image generator could use to generate a similar image. Be detailed. Note if the image is a photograph, diagram, illustration, painting, etc. Provide details about paintings including the style of art and possible artist. Describe attributes of the photograph, such as being award winning, nature photography, portrait, snapshot, etc. Make note of specific brands, logos, locations, scenes, and individuals if it is possible to identify them. Describe any text in the image - including the size, location, color, and style of the text. Do not include any superfluous text in the output, such as headers or statements like "Create an image" or "This image describes" - the generation model does not need those."""

def resize_image(image: Image, max_size: Tuple[int, int] = (512, 512)) -> Image:
    """
    Resize the given image while retaining the aspect ratio.
    Args:
        image (Image): The image to be resized.
        max_size (Tuple[int, int], optional): The maximum size for the image as a tuple of (max_width, max_height). Defaults to (512, 512).
    Returns:
        Image: The resized image.
    """
    # Rest of the code...
    # Get the current size of the image
    original_width, original_height = image.size
    
    # Check if the image is already within the desired size
    if original_width <= max_size[0] and original_height <= max_size[1]:
        return image
    
    # Calculate the new size while retaining the aspect ratio
    aspect_ratio = original_width / original_height
    if aspect_ratio > 1:
        # Landscape orientation
        new_width = max_size[0]
        new_height = int(max_size[0] / aspect_ratio)
    else:
        # Portrait orientation
        new_width = int(max_size[1] * aspect_ratio)
        new_height = max_size[1]
    
    # Resize the image
    resized_image = image.resize((new_width, new_height))
    
    return resized_image

def pil2tensor(image: Image) -> torch.Tensor:
    """
    Converts a PIL image to a PyTorch tensor.

    Args:
        image: A PIL image object.

    Returns:
        A PyTorch tensor representing the image.

    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0) 

def tensor2pil(image: torch.Tensor) -> Image:
    """
    Converts a tensor image to a PIL image.

    Parameters:
        image (torch.Tensor): The input tensor image.

    Returns:
        PIL.Image: The converted PIL image.
    """
    return Image.fromarray((image.squeeze().numpy() * 255).astype(np.uint8))

def tensor2base64(image: torch.Tensor, resize: bool = True) -> str:
    """
    Converts a tensor image to a base64 encoded string.
    Args:
        image (torch.Tensor): The input tensor image.
        resize (bool, optional): Whether to resize the image to a maximum size of 512x512. Defaults to True.
    Returns:
        A base64 encoded string representation of the image.
    """
    image = tensor2pil(image)
    
    if resize:
        image = resize_image(image)

    buffer = BytesIO()
    image.save(buffer, format="JPEG")

    image_bytes = buffer.getvalue()
    base64_string = base64.b64encode(image_bytes).decode('utf-8')

    return base64_string


class LoadEnvironmentVariable:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {       
                    "env_var": ("STRING", {"multiline": False, "default": "OPENAI_API_KEY"}),
                    }
                }
    
    RETURN_TYPES=("STRING",)
    FUNCTION = "get_env_var"
    CATEGORY = "LLMs"

    def get_env_var(self, env_var):
        return (os.environ[env_var],)
    
class OpenAIVision:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {   
                    "image": ("IMAGE",),
                    "api_key": ("STRING", {"forceInput": True}),  
                    "prompt": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT}),
                    "model": ("STRING", {"multiline": False, "default": "gpt-4o-mini"}),
                    }
                }
    
    RETURN_TYPES=("STRING",)
    FUNCTION = "invoke_gpt"
    CATEGORY = "LLMs"

    def invoke_gpt(self, image, api_key, prompt, model):
        # Getting the base64 string
        base64_image = tensor2base64(image)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        payload = {
            "model": model,
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": prompt
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 500
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        print(response.json())
        try:
            return (response.json()["choices"][0]["message"]["content"],)
        except:
            return ("An error occurred.",)

class ImageDimensions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {   
                    "image": ("IMAGE",),
                    }
                }

    RETURN_TYPES=("INT", "INT")
    FUNCTION = "get_image_dimensions"
    CATEGORY = "Image Processing"

    def get_image_dimensions(self, image):
        image = tensor2pil(image)
        return (image.width, image.height)
            
class SideBySideImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {   
                    "base_image": ("IMAGE",),
                    "images": ("IMAGE",),
                    }
                }
    
    RETURN_TYPES=("IMAGE",)
    FUNCTION = "side_by_side"
    CATEGORY = "Image Processing"
    OUTPUT_IS_LIST = (True,)

    def side_by_side(self, base_image, images) -> List[torch.Tensor]:
        results = []
        base_img = tensor2pil(base_image)

        for (batch_number, image) in enumerate(images):
            image2 = tensor2pil(image)
            
            # Resize images to the same height
            new_width = base_img.width + image2.width
            new_height = max(base_img.height, image2.height)
            new_image = Image.new("RGB", (new_width, new_height))
            
            new_image.paste(base_img, (0, 0))
            new_image.paste(image2, (base_img.width, 0))
            results.append(pil2tensor(new_image))

        return (results,)