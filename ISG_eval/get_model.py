from openai import OpenAI
import time
import torch
from diffusers import (
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    FluxPipeline
)
import re
import os
import json
import base64
import io
from PIL import Image
from .utils import local_image_to_data_url, modify_content
import gc
import yaml


class LLM_SD:
    def __init__(self, text_generator=None, image_generator=None, config_path="./ISG_eval/config.yaml"):
        """
        Unified LLM + Image Generation model class that supports calling various text models through OpenRouter
        
        Args:
            text_generator: Name of text generation model, if None uses default model from config
            image_generator: Name of image generation model (sd3, sd2.1, flux)
            config_path: Path to config file
        """
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Set text generation model
        self.text_generator = text_generator if text_generator is not None else self.config["model_name"]
        self.image_generator = image_generator
        
        print(f"Initialized LLM_SD with text model: {self.text_generator}, image model: {image_generator}")
        
        # Initialize image generation pipeline if specified
        if image_generator == "sd3":
            self.pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch.float16
            ).to("cuda")
            
        elif image_generator == "sd2.1":
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16
            ).to("cuda")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            
        elif image_generator == "flux":
            self.pipe = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16
            ).to("cuda")
        else:
            self.pipe = None

    def generate_image(self, prompt):
        """
        Method for generating images
        
        Args:
            prompt: Image generation prompt
            
        Returns:
            Generated PIL Image object, or None if no image generator
        """
        if self.pipe is None:
            print(f"No image generator specified, cannot generate image for: {prompt}")
            return None
            
        if self.image_generator == "sd3":
            image = self.pipe(
                prompt,
                negative_prompt="",
                num_inference_steps=28,
                guidance_scale=7.0,
            ).images[0]
            
        elif self.image_generator == "sd2.1":
            image = self.pipe(prompt).images[0]
            
        elif self.image_generator == "flux":
            image = self.pipe(
                prompt,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=torch.Generator("cuda").manual_seed(0)
            ).images[0]
        else:
            print(f"Unknown image generator: {self.image_generator}")
            return None
        
        gc.collect()
        torch.cuda.empty_cache()
        return image

    def get_res(self, content, image: bool = True, max_retries=None, retry_delay=None):
        """
        Unified text generation method using OpenRouter API to call various models
        
        Args:
            content: Input content list
            image: Whether to include images
            max_retries: Maximum number of retries
            retry_delay: Delay between retries
        """
        print(f"Processing content with model: {self.text_generator}")
        
        # Process content format, convert to OpenAI format
        processed_content = self._process_content_for_openrouter(content, image)
        
        max_retries = max_retries or self.config["max_retries"]
        retry_delay = retry_delay or self.config["retry_delay"]
        
        retries = 0
        while retries < max_retries:
            try:
                # Create OpenRouter client
                client = OpenAI(
                    base_url=self.config["base_url"],
                    api_key=self.config["api_key"]
                )
                
                # Build messages
                messages = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that can generate images based on the user's instructions. If the task requirement need you to provide the caption of the generated image, please provide it out of <image> and </image>. Notice: please use <image> and </image> to wrap the image caption for the images you want to generate. For example, if you want to generate an image of a cat, you should write <image>a cat</image> in your output."
                    },
                    {"role": "user", "content": processed_content}
                ]
                
                # Call API
                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": self.config["site_url"],
                        "X-Title": self.config["site_name"],
                    },
                    model=self.text_generator,
                    messages=messages,
                    temperature=self.config["temperature"],
                    max_tokens=4096
                )
                
                output = response.choices[0].message.content.strip()
                print(f"Model response: {output}")
                return output
                
            except Exception as e:
                retries += 1
                print(f"Connection error (attempt {retries}/{max_retries}): {str(e)}")
                if retries >= max_retries:
                    print(f"Failed after {max_retries} attempts")
                    return None
                time.sleep(retry_delay)
        
        return None

    def _process_content_for_openrouter(self, content, image: bool = True):
        """
        Convert input content to OpenRouter API compatible format
        
        Args:
            content: Original content list
            image: Whether to include images
        
        Returns:
            Processed content list suitable for OpenRouter API
        """
        processed_content = []
        
        for item in content:
            if item["type"] == "image":
                if image:
                    # Convert image to data URL format
                    image_url = local_image_to_data_url(item["content"])
                    processed_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                else:
                    # Replace with text if images not included
                    processed_content.append({
                        "type": "text",
                        "text": "```Here is an image block, while I can't provide you the real image content. You can assume here is an image here.```"
                    })
            elif item["type"] == "text":
                processed_content.append({
                    "type": "text",
                    "text": item["content"]
                })
        
        return processed_content

    def get_mm_output(self, content, save_dir, id):
        text_output = self.get_res(content)
        return self._process_output(text_output, save_dir, id)

    def get_mm_output_wo_image(self, content, save_dir, id):
        text_output = self.get_res(content, image=False)
        return self._process_output(text_output, save_dir, id)
        
    def _process_output(self, text_output, save_dir, id):
        """
        Process output, extract text and image captions, generate corresponding images
        
        Args:
            text_output: Model output text
            save_dir: Image save directory
            id: Sample ID
            
        Returns:
            Processed result list containing text and image parts
        """
        if text_output is None:
            return []
            
        print(f"Processing output: {text_output}")
        image_captions = re.findall(r'<image>(.*?)</image>', text_output)
        result = []
        text_parts = re.split(r'<image>.*?</image>', text_output)
        
        for i, text in enumerate(text_parts):
            if text.strip():
                result.append({"type": "text", "content": text.strip()})
                
            if i < len(image_captions):
                image_dict = {
                    "type": "image",
                    "caption": image_captions[i]
                }
                
                # Try to generate image
                image = self.generate_image(image_captions[i])
                if image is not None:
                    # Ensure save directory exists
                    os.makedirs(save_dir, exist_ok=True)
                    image_filename = f"{id}_g{i+1}.png"
                    image_path = os.path.join(save_dir, image_filename)
                    image.save(image_path)
                    print(f"Image saved to {image_path}")
                    image_dict["content"] = image_path
                else:
                    # Record but continue processing if image generation fails
                    print(f"Failed to generate image for caption: {image_captions[i]}")
                    image_dict["content"] = None
                    
                result.append(image_dict)
                
        return result