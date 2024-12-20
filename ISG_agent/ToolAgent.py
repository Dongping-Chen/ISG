import os
from openai import OpenAI
from anthropic import  Anthropic
import google.generativeai as genai
from argparse import ArgumentParser
import json
import base64
from typing import List, Dict, Any
from retry import retry

"""
Unverify: 
The input of each function
StoryTelling part
"""

from api_interface import (
    generate_video_agent,
    generate_image_agent,
    edit_image_agent,
    generate_3d_video_agent,
    morph_images_agent
)
ClaudeClient = Anthropic()
OpenAIClient = OpenAI()
TOOL_DIR = "Tools"
TOOL_AGENT_FILE = os.path.join(TOOL_DIR, "Tools_For_Sota.json")

def load_tools(tool_names):
    tools = []
    for tool_name in tool_names:
        tool_file = os.path.join('Tools', f'{tool_name}.json')
        try:
            with open(tool_file, 'r') as f:
                tool_definition = json.load(f)
                tools.append(tool_definition)
        except Exception as e:
            print(f'Error loading tool {tool_name}: {e}')
    return tools

def process_agent_tool_use(agent_response,data):
    # Assume `agent_response` contains tool_use info
    # print(agent_response)
    if agent_response.stop_reason == "tool_use":
        tool_use = agent_response.content[-1]  # Extract the tool use information
        tool_name = tool_use.name
        tool_input = tool_use.input

        try:
            # Depending on the tool name, call the appropriate API function
            if tool_name == "VideoGeneration":
                print("Agent wants to generate a video")
                prompt_list = []
                if len(data['Input_images']) == 1:
                    prompt_list=tool_input["prompt_list"]
                    num_images = tool_input.get("num_screenshot", 4)
                    seconds_per_screenshot = float(4) / num_images
                    print(seconds_per_screenshot)
                    prompt_list.append({"type": "image", "content": data['Input_images'][0]})
                else:
                    raise ValueError("Invalid number of input images")
                
                screenshots = generate_video_agent(prompt_list,seconds_per_screenshot=seconds_per_screenshot)
                return {"text":"","images": screenshots}
            
            elif tool_name == "ImageGeneration":
                print("Agent wants to generate an image")
                prompt = tool_input["prompt"]
                if len(data['Input_images']) != 0:
                    print(data['Input_images'])
                    raise ValueError("ImageGeneration tool does not accept input images")
                image = generate_image_agent(prompt)
                return {"text":"","images": [image]}
            
            elif tool_name == "ImageEdit":
                print("Agent wants to edit an image")
                prompt = tool_input["prompt"]
                try:
                    image_input = data['Input_images'][0]
                except IndexError:
                    raise ValueError("Edit tool requires an input image")
                edited_image = edit_image_agent(prompt, image_input)
                return {"text":"","images": [edited_image]}
            
            elif tool_name == "Fixed3DGeneration":
                print("Agent wants to generate a 3D video")
                input_list = []
                input_list.append({"type": "image", "content": data['Input_images'][0]})
                screenshots_per_second = tool_input.get("screenshots_per_second", 1)
                screenshots = generate_3d_video_agent(input_list, screenshots_per_second)
                
                return {"text":"","images": screenshots}
            elif tool_name == "Free3DGeneration":
                print("Agent wants to generate a 3D video with free form")
                input_list = []
                input_list.append({"type": "image", "content": data['Input_images'][0]})
                screenshots_per_second = tool_input.get("screenshots_per_second", 1)
                proportions = tool_input.get("time_stamps",[])
                screenshots = generate_3d_video_agent(input_list, screenshots_per_second=screenshots_per_second,proportions=proportions)
                return {"text":"","images": screenshots}
            elif tool_name == "ImageMorph":
                print("Agent wants to morph images")
                if len(data['Input_images']) != 2:
                    raise ValueError("ImageMorph tool requires two input images")
                img1 = data['Input_images'][0]
                img2 = data['Input_images'][1]
                prompt = tool_input["prompt"]
                morphed_images = morph_images_agent(img1, img2, prompt)
                return {"text":"","images": morphed_images}
            else:
                raise ValueError(f"Unknown tool name: {tool_name}")

        except Exception as e:
            raise ValueError(f"Error using tool '{tool_name}': {str(e)}")
    elif agent_response.stop_reason == "end_turn":
        print("No tool used")
        return {"text": agent_response.content[0].text, "images": []}
    else:
        raise ValueError(f"Unsupported stop reason: {agent_response.stop_reason}")

def get_image_media_type(image_data):
    """
    Detects the image media type based on the magic number in the binary data.
    """
    if image_data[:4] == b'\x89PNG':
        return "image/png"
    elif image_data[:2] == b'\xFF\xD8':
        return "image/jpeg"
    else:
        raise ValueError("Unsupported image format")

def tool_agent(json_input:str,task: str) -> Dict[str, Any]:
    """
    Agent Tool
    """
    print(f"Task: {task}")

    try:
        data = json.loads(json_input)
        input_text = data.get('Input_text', '')
        input_images = data.get('Input_images', [])
        print(f"Input text: {input_text}")
        print(f"Input images: {input_images}")
    except json.JSONDecodeError as e:
        return {'error': f'Invalid JSON input: {str(e)}'}
    
    # input_text = Simplify(client,input_text)
    # print(f"Extracted for better generation: {input_text}")
    
    try:
        with open(TOOL_AGENT_FILE, 'r') as f:
            tools = json.load(f)
            # print(tools)
    except Exception as e:
        return {'error': f'Error reading tools.json: {str(e)}'}
    
        # Prepare the messages
    messages = []

    # Prepare the content for the message
    content = []
    if task == "Call_tool":
        # Data Health Check: If <GEN_{ID}> in input_images, raise error
        if any('<GEN_' in img for img in input_images):
            raise ValueError("Invalid input_images: contains unresolved placeholders like <GEN_{ID}>.")
        content.append({"type": "text", "text": "Input text:" + input_text})
        content.append({"type": "text", "text": "Input images number:" + str(len(input_images))})
        print(content[-2]['text'])
        print(content[-1]['text'])
        messages.append({
            "role": "user",
            "content": content,
        })
        tool_choice = {"type":"any"}
        
        # Call the API
        try:
            response = ClaudeClient.messages.create(
                model="claude-3-5-sonnet-20240620",  # Replace with your model name
                max_tokens=512,
                tools=tools,
                tool_choice=tool_choice,
                messages=messages,
                system="Decide which tool to use with regard to the instruction and the input image number. Since you have no access to the images themselves, you should make decision base on the text instruction and image number only. Instruction will explicitly tell which tool to use, input image number will restrict which tool cannot use. ImageGeneration requires no image input, ImageEdit,VideoGeneration,3DGeneration requires one image input.Make the input text argument **descriptive** for image, video generation or image edit tools to understand. Make the input text argument concise, refine the important information to avoid truncation raised by visual generation model",
            )
        except Exception as e:
            raise ValueError(f"Error calling the API: {e}")
    else:
        if input_images:
            for img in input_images:
                if os.path.isfile(img):
                    try: 
                        with open(img, "rb") as f:
                            image_data = f.read()
                            # Encode the image to Base64
                            img_base64 = base64.b64encode(image_data).decode("utf-8")
                            # Dynamically determine the media type
                            image_media_type = get_image_media_type(image_data)
                            
                            # Append the image data to the content list
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_media_type,  # Adjusted dynamically
                                    "data": img_base64,
                                },
                            })
                    except Exception as e:
                        raise ValueError(f"Error reading image file: {e}")
                else:
                    raise ValueError(f"Invalid image file: {img}")

        content.append({"type": "text", "text": input_text})
        # print(content)
        print(content[-1]['text'])
        messages.append({
            "role": "user",
            "content": content,
        })
        try:
            response = ClaudeClient.messages.create(
                model="claude-3-5-sonnet-20240620",  # Replace with your model name
                max_tokens=8192,
                messages=messages,
                system="In this task, generate a **brief** `text-only` response with or without image input. You should simulate as if the image is generated by yourself to maintain the continuity. You should **Never** apologizes or response any form of negative acknowledgment like \"I apologize\", \"The image doesn't show\" of the image regarding the presence or absence of certain elements in the image. You should use \"This image\" instead of \"The image\" Never compare the differences and discrepancies between the text instruction and the image, only focus on the similar part. The response should be harmonious with both the image and the instruction, ensuring that any contradictions or irrelevant details are ignored. If you cannot extract any aligning information, you should focus more on the instruction. Do not output too many unimportant things if the instruction didn't ask even if the instruction wants you to output in detail.",
            )
        except Exception as e:
            raise ValueError(f"Error calling the API: {e}")
    return process_agent_tool_use(response,data)