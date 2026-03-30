"""
Harmful dataset 1: https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors

Papers:
[Dataset 1] JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models
https://arxiv.org/pdf/2404.01318
"""

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
from tqdm import tqdm

import argparse

# Calculate the four corners of the rotated rectangle
def rotate_point(x, y, center_x, center_y, angle):
    """Rotate a point around a center point by given angle"""
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    
    # Translate to origin
    x_translated = x - center_x
    y_translated = y - center_y
    
    # Rotate
    x_rotated = x_translated * cos_angle - y_translated * sin_angle
    y_rotated = x_translated * sin_angle + y_translated * cos_angle
    
    # Translate back
    x_final = x_rotated + center_x
    y_final = y_rotated + center_y
    
    return (x_final, y_final)

def create_image(placement_data, image, text, filepath):
    draw = ImageDraw.Draw(image)

    # Original bounding box (x1, y1, x2, y2) and rotation angle in degrees
    x1, y1, x2, y2 = placement_data["coord"]
    rotation_angle = placement_data["rotation"]

    # Calculate center of the bounding box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Convert angle to radians
    angle_rad = math.radians(rotation_angle)

    # Calculate the four corners
    corners = [
        rotate_point(x1, y1, center_x, center_y, angle_rad),  # top-left
        rotate_point(x2, y1, center_x, center_y, angle_rad),  # top-right
        rotate_point(x2, y2, center_x, center_y, angle_rad),  # bottom-right
        rotate_point(x1, y2, center_x, center_y, angle_rad),  # bottom-left
    ]

    # Create a temporary image for the bounding box with text
    box_width = x2 - x1
    box_height = y2 - y1

    # Create temp image large enough to handle rotation without clipping
    temp_size = int(max(box_width, box_height) * 1.5)
    temp_img = Image.new('RGBA', (temp_size, temp_size), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)

    # Calculate center position in temp image
    temp_center = temp_size // 2
    temp_x1 = temp_center - box_width // 2
    temp_y1 = temp_center - box_height // 2
    temp_x2 = temp_x1 + box_width
    temp_y2 = temp_y1 + box_height

    # # Draw the red bounding box on temp image before rotation
    # temp_draw.rectangle([temp_x1, temp_y1, temp_x2, temp_y2], outline="red", width=5)

    # Try different font sizes to fit within the bounding box (minimum font size 20)
    font_size = 60
    font = None
    text_lines = []
    text_fits = False

    while font_size >= 30 and not text_fits:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Calculate available width and height (with padding)
        available_width = int(box_width * 0.95)
        available_height = int(box_height * 0.95)
        
        # Get line height for this font
        line_bbox = temp_draw.textbbox((0, 0), "Ag", font=font)
        line_height = line_bbox[3] - line_bbox[1]
        
        # Calculate maximum number of lines that fit
        max_lines = available_height // line_height
        
        # Try to wrap text to fit within the available width
        words = text.split()
        text_lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            test_width = test_bbox[2] - test_bbox[0]
            
            if test_width <= available_width:
                current_line = test_line
            else:
                if current_line:
                    text_lines.append(current_line)
                    current_line = word
                else:
                    # Single word is too long, force it (this means font is too big)
                    text_lines.append(word)
                    current_line = ""
        
        if current_line:
            text_lines.append(current_line)
        
        # Check if text fits within bounding box
        total_text_height = len(text_lines) * line_height
        
        # Calculate actual max width of all lines
        actual_max_width = 0
        for line in text_lines:
            line_bbox = temp_draw.textbbox((0, 0), line, font=font)
            line_width = line_bbox[2] - line_bbox[0]
            actual_max_width = max(actual_max_width, line_width)
        
        # Check if everything fits
        width_fits = actual_max_width <= available_width
        height_fits = total_text_height <= available_height
        lines_fit = len(text_lines) <= max_lines
        
        if width_fits and height_fits and lines_fit:
            text_fits = True
        else:
            font_size -= 5

    # If we couldn't fit with minimum font size, use it anyway with truncation
    if font_size < 30:
        font_size = 30
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # Recalculate with minimum font
        available_width = int(box_width * 0.9)
        available_height = int(box_height * 0.9)
        line_bbox = temp_draw.textbbox((0, 0), "Ag", font=font)
        line_height = line_bbox[3] - line_bbox[1]
        max_lines = available_height // line_height
        
        # Re-wrap text with minimum font
        words = text.split()
        text_lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + " " + word if current_line else word
            test_bbox = temp_draw.textbbox((0, 0), test_line, font=font)
            test_width = test_bbox[2] - test_bbox[0]
            
            if test_width <= available_width:
                current_line = test_line
            else:
                if current_line:
                    text_lines.append(current_line)
                    current_line = word
                else:
                    text_lines.append(word)
                    current_line = ""
        
        if current_line:
            text_lines.append(current_line)
        
        # Truncate lines if necessary
        if len(text_lines) > max_lines:
            text_lines = text_lines[:max_lines]
            print(f"Text truncated to {max_lines} lines")

    # Calculate total text dimensions
    total_text_height = len(text_lines) * line_height
    max_text_width = 0
    for line in text_lines:
        line_bbox = temp_draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        max_text_width = max(max_text_width, line_width)

    # Center the text block within the bounding box in temp image
    text_x = temp_x1 + (box_width - max_text_width) // 2
    text_y = temp_y1 + (box_height - total_text_height) // 2

    # Draw each line of text
    for i, line in enumerate(text_lines):
        line_bbox = temp_draw.textbbox((0, 0), line, font=font)
        line_width = line_bbox[2] - line_bbox[0]
        # Center each line horizontally
        line_x = temp_x1 + (box_width - line_width) // 2
        line_y = text_y + i * line_height
        temp_draw.text((line_x, line_y), line, font=font, fill="black")

    # Rotate the entire temp image
    rotated_temp_img = temp_img.rotate(rotation_angle, expand=True)

    # Calculate position to paste the rotated temp image at the original bounding box center
    rotated_width, rotated_height = rotated_temp_img.size
    paste_x = int(center_x - rotated_width // 2)
    paste_y = int(center_y - rotated_height // 2)

    # Paste the rotated temp image onto the main image
    image.paste(rotated_temp_img, (paste_x, paste_y), rotated_temp_img)

    # Save and display
    image.save(f"{filepath}.png")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--type", default="all", choices=["article", "speech", "instruction", "message", "code", "all"], help="Type of the comic")

    argparser.add_argument("--start", type=int, default=0, help="Start index of the dataset to process")
    argparser.add_argument("--end", type=int, default=300, help="End index of the dataset to process")

    args = argparser.parse_args()

    assert args.start >= 0 and args.end > args.start, "Invalid start and end indices"

    # Coordinates and rotation for each type of comic and rotation angle in degrees
    TYPES = {
        "instruction": {"coord": (1200, 130, 1420, 500), "rotation": 10},
        "speech": {"coord": (1050, 40, 1500, 240), "rotation": 0},
        "article": {"coord": (1080, 70, 1480, 680), "rotation": 0},
        "message": {"coord": (1160, 120, 1400, 580), "rotation": 6},
        "code": {"coord": (1130, 210, 1490, 510), "rotation": 0}
    }

    TYPE = ["article", "speech", "instruction", "message", "code"] if args.type == "all" else [args.type]

    for template in TYPE:
        folder_dir = f"dataset/{template}"

        os.makedirs(folder_dir, exist_ok=True)

        # Load dataset and process each row
        df = pd.read_csv("dataset.csv")[args.start:args.end]

        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            image = Image.open(f"template/{template}.png")
            filepath = f"{folder_dir}/{idx}"
            text = row[template.capitalize()]
            if text is None or text is np.nan:
                continue
            create_image(TYPES[template], image, text, filepath)