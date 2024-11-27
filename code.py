import os
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import re
import dateparser
from datetime import datetime

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

def process_images_from_folder(folder_path):
    current_date = datetime.now().strftime('%d/%m/%Y')

    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        
        if not image_filename.lower().endswith(('.png', '.jpg', '.jpeg')): 
            continue
            
        image = Image.open(image_path)
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"""
EXTRACT FOLLOWING INFORMATION FROM THE IMAGE:

- MRP (IN INDIAN RUPEES, e.g., ₹123.45)
- MANUFACTURE DATE (e.g., 12/2022, 12-2022, 12 2022)
- EXPIRY DATE (e.g., 12/2022, 12-2022, 12 2022)
- NET WEIGHT (e.g., 100g, 100G, 100 GM)
- BRAND NAME

CURRENT DATE: {current_date}

PROVIDE OUTPUT IN THE FOLLOWING FORMAT:

MRP: 
Manufacture Date: 
Expiry Date: 
Net Weight: 
Brand Name: 
expired or not (considering current date {current_date}): 
"""
                    }
                ]
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    
        inputs = processor(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device) 
        
        output = model.generate(**inputs, max_new_tokens=200)

        generated_text = processor.decode(output[0], skip_special_tokens=True).strip()

        prompt_length = len(processor.decode(inputs["input_ids"][0], skip_special_tokens=True).strip())
        output_text = generated_text[prompt_length:].strip()

        mrp = re.search(r"MRP: (\d+(?:\.\d+)?)", output_text)
        mfg_date = re.search(r"Manufacture Date: (\d{1,2}/\d{1,2}/\d{2,4})", output_text)
        exp_date = re.search(r"Expiry Date: (\d{1,2}/\d{1,2}/\d{2,4})", output_text)
        net_weight = re.search(r"Net Weight: (\d+(?:\.\d+)?[gGmM])", output_text)
        brand_name = re.search(r"Brand Name: ([\w\s]+)", output_text)
        expiry_status = re.search(r"expired or not: (.+)", output_text)

        if mrp and mfg_date and exp_date and net_weight and brand_name and expiry_status:
    
            mfg_date = dateparser.parse(mfg_date.group(1))
            exp_date = dateparser.parse(exp_date.group(1))

            print(f"Output for {image_filename}:")
            print(f"Brand Name: {brand_name.group(1)}")
            print(f"MRP: ₹{mrp.group(1)}")
            print(f"Manufacture Date: {mfg_date.strftime('%d/%m/%Y')}")
            print(f"Expiry Date: {exp_date.strftime('%d/%m/%Y')}")
            print(f"Net Weight: {net_weight.group(1)}")
            print(f"Expiry Status: {expiry_status.group(1)}")
        else:
            print(f"Error parsing output for {image_filename}: {output_text}")

# Specify the folder path containing the images
folder_path = "folder path"

process_images_from_folder(folder_path)
