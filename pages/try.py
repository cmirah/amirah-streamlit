import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io

# Create a blank image with white background
width, height = 800, 400
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# Define fonts
title_font = ImageFont.load_default()
section_font = ImageFont.load_default()
text_font = ImageFont.load_default()

# Function to calculate text size
def get_text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]

# Title
title_text = "Machine Learning Workflow"
title_width, title_height = get_text_size(draw, title_text, title_font)
draw.text(((width - title_width) / 2, 10), title_text, fill="black", font=title_font)

# Section Titles
sections = ["Data Preparation", "Model Building", "Deployment", "Production"]
section_positions = [(50, 60), (250, 60), (450, 60), (650, 60)]

for section, position in zip(sections, section_positions):
    draw.rectangle([position, (position[0] + 150, position[1] + 50)], outline="black", width=2)
    text_width, text_height = get_text_size(draw, section, section_font)
    draw.text((position[0] + (150 - text_width) / 2, position[1] + (50 - text_height) / 2), section, fill="black", font=section_font)

# Arrows
arrow_positions = [(200, 85), (400, 85), (600, 85)]
arrow_texts = ["Train Model", "Deploy Model", "Run in Production"]

for position, text in zip(arrow_positions, arrow_texts):
    draw.line([position, (position[0] + 50, position[1])], fill="black", width=2)
    draw.polygon([position, (position[0] + 50, position[1] - 5), (position[0] + 50, position[1] + 5), (position[0] + 60, position[1])], fill="black")
    text_width, text_height = get_text_size(draw, text, text_font)
    draw.text((position[0] + 25 - text_width / 2, position[1] + 10), text, fill="black", font=text_font)

# Descriptions
descriptions = [
    ["Training Code", "Labeled Data"],
    ["Model Building"],
    ["Deployment"],
    ["Production"]
]

description_positions = [(50, 120), (250, 120), (450, 120), (650, 120)]

for description, position in zip(descriptions, description_positions):
    for i, line in enumerate(description):
        text_width, text_height = get_text_size(draw, line, text_font)
        draw.text((position[0] + (150 - text_width) / 2, position[1] + i * 20), line, fill="black", font=text_font)

# Convert the image to a byte array
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format='PNG')
img_byte_arr = img_byte_arr.getvalue()

# Display the image in Streamlit
st.image(img_byte_arr, caption="Machine Learning Workflow")
