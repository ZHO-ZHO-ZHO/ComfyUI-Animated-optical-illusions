import sys
import torch, os
from PIL import Image, ImageFont, ImageDraw
import numpy as np



def tensor2numpy(image):
    return np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


class AOI_Processing_Zho:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 1, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "aoi_processing"
    CATEGORY = "Zho模块组/AOI"

    def interleave(self, frames, width):
        splits = range(width, frames[0].shape[1], width)
        return np.hstack(list(np.hstack(s) for s in zip(*(np.hsplit(f, splits)[k::len(frames)] for k, f in enumerate(frames)))))

    def aoi_processing(self, images, width):
        images = [tensor2numpy(img) for img in images]
        processed_images = Image.fromarray(self.interleave(images, width))
        output_img = pil2tensor(processed_images)

        white = (255 * np.ones(images[0].shape)).astype(np.uint8)
        black = np.zeros(images[0].shape).astype(np.uint8)
        mask = Image.fromarray(self.interleave([white, ] + [black for _ in range(len(images) - 1)], width))

        rgba_mask = mask.convert("RGBA")
        datas = rgba_mask.getdata()
        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        rgba_mask.putdata(newData)

        output_mask = pil2tensor(rgba_mask)

        return (output_img, output_mask)


NODE_CLASS_MAPPINGS = {
    "AOI_Processing_Zho": AOI_Processing_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AOI_Processing_Zho": "AOI_Processing_Zho"
}