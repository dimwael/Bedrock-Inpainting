import json
import os
import boto3
from PIL import Image
import io
import base64
import random


modelId = "stability.stable-diffusion-xl-v1"
negative_prompts = [
    "bad quality",
    "wierd details",
    "poorly rendered image",
    "not clear details",
]

session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", None))
client = session.client("bedrock-runtime")


def image_to_base64(img) -> str:
    if isinstance(img, str):
        if os.path.isfile(img):
            print(f"Reading image from file: {img}")
            with open(img, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        else:
            raise FileNotFoundError(f"File {img} does not exist")
    elif isinstance(img, Image.Image):
        print("Converting PIL Image to base64 string")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    else:
        raise ValueError(f"Expected str (filename) or PIL Image. Got {type(img)}")


def inpaint_image(change_prompt, image, mask, sampler="K_DPMPP_2S_ANCESTRAL") -> Image :
    request = json.dumps(
        {
            "text_prompts": (
                [{"text": change_prompt, "weight": 1.0}]
                + [
                    {"text": negprompt, "weight": -1.0}
                    for negprompt in negative_prompts
                ]
            ),
            "cfg_scale": 20,
            "init_image": image_to_base64(image),
            "mask_source": "MASK_IMAGE_WHITE",
            "script": "Outpainting MK2",
            "masked_content": "fill",
            "mask_image": image_to_base64(mask),
            "seed": random.randint(0, 10000),
            "steps": 30,
            "sampler": sampler,
        }
    )

    response = client.invoke_model(body=request, modelId=modelId)

    response_body = json.loads(response.get("body").read())
    image_b64_str = response_body["artifacts"][0].get("base64")
    image_result = Image.open(
        io.BytesIO(base64.decodebytes(bytes(image_b64_str, "utf-8")))
    )
    return image_result
