import os
import json
import re
import base64
import time
import yaml
from openai import OpenAI

# Load configuration
def load_config(config_path="./ISG_eval/config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def modify_content(dir):
    json_file = [f for f in os.listdir(dir) if f.endswith('.json')][0]
    with open(os.path.join(dir, json_file), 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    text = data.get('text', '')
    
    parts = re.split(r'(#image\d+#)', text)
    
    result = []
    for part in parts:
        if part.startswith('#image') and part.endswith('#'):
            result.append({
                "type": "image",
                "image_path": part.strip('#')
            })
        elif part.strip():
            result.append({
                "type": "text",
                "content": part.strip()
            })
    
    return result


def get_detailed_caption(image_path, config_path="./ISG_eval/config.yaml"):
    """
    Generate a detailed caption for an image using OpenRouter API
    
    Args:
        image_path: Path to the image file
        config_path: Path to the configuration file
    
    Returns:
        Detailed caption string
    """
    config = load_config(config_path)
    
    content = [
        {
            "type": "text",
            "text": """Task: Generate a detailed caption for an image.
Input: An image.

Output: A detailed caption describe what is in this image. Focus on all important entities, their attributes and their relationships. Do not include any other information. Make sure the caption is clear, accurate and easy to understand.

Here is the images:"""
        },
        {
            "type": "image_url",
            "image_url": {"url": local_image_to_data_url(image_path)}
        }
    ]
    
    max_retries = config.get("max_retries", 3)
    retry_delay = config.get("retry_delay", 2)
    
    retries = 0
    while retries < max_retries:
        try:
            client = OpenAI(
                base_url=config["base_url"],
                api_key=config["api_key"]
            )
            
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": config["site_url"],
                    "X-Title": config["site_name"],
                },
                model=config["model_name"],
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant. Please output in JSON format. Do not write an introduction or summary. Do not output other irrelevant information. Do not output the example in the prompt. Focus on the input prompt and image."
                    },
                    {"role": "user", "content": content}
                ],
                temperature=config.get("temperature", 0.7),
            )
            
            output = response.choices[0].message.content.strip()
            print(f"Generated detailed caption: {output}")
            return output
            
        except Exception as e:
            retries += 1
            print(f"Error generating detailed caption (attempt {retries}/{max_retries}): {str(e)}")
            if retries >= max_retries:
                print(f"Failed after {max_retries} attempts")
                return None
            time.sleep(retry_delay)
    
    return None

def get_caption(image_path, config_path="./ISG_eval/config.yaml"):
    """
    Generate a short caption for an image using OpenRouter API
    
    Args:
        image_path: Path to the image file
        config_path: Path to the configuration file
    
    Returns:
        Short caption string
    """
    config = load_config(config_path)
    
    content = [
        {
            "type": "text",
            "text": """Task: Generate a caption for an image.
Input: An image.

Output: A short and accurate caption describe what is in this image. Focus on the main entities, their attributes and their relationships. Do not include any other information. Make sure the caption is clear, accurate and easy to understand.

Here is the images:"""
        },
        {
            "type": "image_url",
            "image_url": {"url": local_image_to_data_url(image_path)}
        }
    ]

    max_retries = config.get("max_retries", 3)
    retry_delay = config.get("retry_delay", 2)
    
    retries = 0
    while retries < max_retries:
        try:
            client = OpenAI(
                base_url=config["base_url"],
                api_key=config["api_key"]
            )
            
            response = client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": config["site_url"],
                    "X-Title": config["site_name"],
                },
                model=config["model_name"],
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant. Please output in JSON format. Do not write an introduction or summary. Do not output other irrelevant information. Do not output the example in the prompt. Focus on the input prompt and image."
                    },
                    {"role": "user", "content": content}
                ],
                temperature=config.get("temperature", 0.7),
            )
            
            output = response.choices[0].message.content.strip()
            print(f"Generated caption: {output}")
            return output
            
        except Exception as e:
            retries += 1
            print(f"Error generating caption (attempt {retries}/{max_retries}): {str(e)}")
            if retries >= max_retries:
                print(f"Failed after {max_retries} attempts")
                return None
            time.sleep(retry_delay)
    
    return None

def local_image_to_data_url(image_path):
    """
    Convert a local image file to a data URL
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Data URL string
    """
    # Get MIME type based on file extension
    file_ext = os.path.splitext(image_path)[1].lower()
    
    # Map file extensions to MIME types
    mime_type_map = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
        '.svg': 'image/svg+xml'
    }
    
    mime_type = mime_type_map.get(file_ext, 'image/jpeg')  # Default to jpeg
    
    # Read and encode the image file
    if not os.path.exists(image_path):
        image_path = "./ISG_eval/" + image_path
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

