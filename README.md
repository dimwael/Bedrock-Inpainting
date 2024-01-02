# Bedrock-Inpainting
Using Bedrock to inpaint images. Using Stable Diffusion model through a GUI to specify location of changes in images. By default, this project uses Stable Diffusion v1.0 -> `stability.stable-diffusion-xl-v1`


# How to use
## Installation

```sh
pip install streamlit-drawable-canvas==0.9.3
```
```sh
pip install pip install boto3==1.34.9
```



## Running the server : 
```sh
streamlit run main.py
```



:exclamation: You may also need an AWS Account configured before running this
> This repo uses [botocore](https://github.com/boto/botocore) internally for authentication; you can read more about the default providers [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).


# Advanced
## Usage
You can tweak the API call to Stable Diffusion in `bedrock.py` : 

```py
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
```
You can also change the negative prompts in `bedrock.py` : 
```py
negative_prompts = [
    "bad quality",
    "wierd details",
    "poorly rendered image",
    "not clear details",
]

```

## Sampler 
You can refer to this documentation for furhter reading about Sampler [here](https://stable-diffusion-art.com/samplers/)
