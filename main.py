import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import bedrock

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:",
    ("rect", "transform"),
)

stroke_width = 3
debug_rectangle = True
sampler = st.sidebar.selectbox ("Sampler" , ['K_DPMPP_2S_ANCESTRAL', 'DDIM', 'DDPM', 'K_DPMPP_SDE', 'K_DPMPP_2M', 'K_DPM_2', 'K_DPM_2_ANCESTRAL', 'K_EULER', 'K_EULER_ANCESTRAL', 'K_HEUN', 'K_LMS'])
bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
realtime_update = st.sidebar.checkbox("Update in realtime", True)
image_size = (512, 512)
bg_color = "#000"
stroke_color = "#000"

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.1)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=Image.open(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=512,
    width=512,
    drawing_mode=drawing_mode,
    key="canvas",
)

if canvas_result.image_data is not None:
    mask = Image.fromarray(canvas_result.image_data)
    mask = mask.convert("RGB")

inpaint_text = st.text_input("Put input here")
inpaint_button = st.button("Inpaint", type="primary")


if inpaint_button:
    image = Image.open(bg_image)
    mask = mask.resize(image_size)
    image = image.resize(image_size)
    result_image = bedrock.inpaint_image(inpaint_text, image, mask , sampler)
    st.image(result_image)

if debug_rectangle and canvas_result.json_data is not None:
    objects = pd.json_normalize(
        canvas_result.json_data["objects"]
    )  # need to convert obj to str because PyArrow
    for col in objects.select_dtypes(include=["object"]).columns:
        objects[col] = objects[col].astype("str")
        objects["fill"] = "rgba(255, 255, 255, 1)"
        objects["visible"] = False
    st.dataframe(objects)
