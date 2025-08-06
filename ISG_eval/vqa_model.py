from openai import OpenAI
import time
import base64
import json
import os
import yaml

class VQA_Model:
    def __init__(self, model_name=None, config_path="./ISG_eval/config.yaml"):
        """
        Unified VQA model class that supports calling various models through OpenRouter
        
        Args:
            model_name: Model name, if None use default model from config
            config_path: Path to config file
        """
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Set model name
        self.model_name = model_name if model_name is not None else self.config["model_name"]
        
        # Set default prompt
        self.prompt = """You are a helpful and impartial visual assistant. Please follow user's instructions strictly."""
        
        print(f"Initialized VQA_Model with model: {self.model_name}")

    @staticmethod
    def encode_image(image_path):
        """Encode image file as base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    def generate_answer(self, content_list, max_retries=None, retry_delay=None, response_format=None):
        """
        Unified answer generation method using OpenRouter API
        
        Args:
            content_list: List of input contents
            max_retries: Maximum number of retries
            retry_delay: Delay between retries
            response_format: Response format, defaults to json_object
        """
        max_retries = max_retries or self.config["max_retries"]
        retry_delay = retry_delay or self.config["retry_delay"]
        response_format = response_format or {"type": "json_object"}
        
        retries = 0
        while retries < max_retries:
            try:
                # Create OpenRouter client
                client = OpenAI(
                    base_url=self.config["base_url"],
                    api_key=self.config["api_key"]
                )
                
                # Call API
                response = client.chat.completions.create(
                    extra_headers={
                        "HTTP-Referer": self.config["site_url"],
                        "X-Title": self.config["site_name"],
                    },
                    model=self.model_name,
                    messages=[{"role": "user", "content": content_list}],
                    temperature=self.config["temperature"],
                    response_format=response_format
                )
                
                output = response.choices[0].message.content.strip()
                
                # Try to parse JSON response
                if output is not None and response_format.get("type") == "json_object":
                    try:
                        output = json.loads(output)
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON response: {output}")
                        output = {"error": "Invalid JSON response", "raw_output": output}
                
                print(f"Model response: {output}")
                return output
                
            except Exception as e:
                retries += 1
                print(f"Connection error (attempt {retries}/{max_retries}): {str(e)}")
                if retries >= max_retries:
                    print(f"Failed after {max_retries} attempts")
                    return None
                time.sleep(retry_delay)
