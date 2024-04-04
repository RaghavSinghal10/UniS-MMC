import pathlib
import textwrap
import os
import re
import PIL.Image
import requests
import time

import google.generativeai as genai

GOOGLE_API_KEY = 'AIzaSyC0_2UMJO3JNA9b3h6sIt0jC6QY-1Gmqu4'
genai.configure(api_key=GOOGLE_API_KEY)

prompt = """
Perform analysis of the skin lesion in the provided image, integrating dermatoscopic terminologies. Conduct a color analysis, patterns, and pigment network detailing. Analyze structural features such as globules, clods, cobblestone pattern, and dots evaluation. Assess symmetry and border characteristics. Detail vascular structures, categorizing vessel types and morphology. Do not provide the differential diagnosis and provide description in a concise paragraph and be less verbose.

"""

model = genai.GenerativeModel('models/gemini-1.0-pro-vision-latest')

# Path to the folder containing images in Google Drive
image_folder_path = '/home/raghav/multimodal-classification/UniS-MMC/datasets/Hamm100000/ISIC2018_Task3_Training_Input'

# Get a list of image files in the folder
image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_NONE"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_NONE"
  }
]

# Loop through each image file
for image_file_name in image_files[:]:
    try:
        # Extract the file name without extension
        file_name_without_extension = os.path.splitext(os.path.basename(image_file_name))[0]
        # Check if the corresponding text file exists
        if not os.path.exists(os.path.join(image_folder_path, f"{file_name_without_extension}.txt")):
            print(image_file_name)
            # Open the image file
            img = PIL.Image.open(os.path.join(image_folder_path, image_file_name))

            response = model.generate_content([prompt, img], stream=False, safety_settings=safety_settings)
            # print(response.prompt_feedback)
            # Check if prompt feedback is equal to "SUCCESS"
            # Use your code to generate response text
            response_text = response.text

            # Get the initials
            initials = file_name_without_extension

            # Define a function to save response text to a .txt file
            def save_response_to_txt(response_text, initials):
                with open(os.path.join(image_folder_path, f"{initials}.txt"), "w") as file:
                    file.write(response_text)

            # Save response text to .txt file
            save_response_to_txt(response_text, initials)
    except Exception as e:
        # Print the error message
        print(f"Error processing image '{image_file_name}': {e}")
        # Skip to the next image
        continue
