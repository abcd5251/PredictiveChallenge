import os
import re
import base64
from utils.helper_functions import num_tokens_from_string
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# Function to encode an image to Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def create_image_message(image_paths):
    messages = []
    for image_path in image_paths:
        base64_image = encode_image(image_path)
        messages.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                },
            }
        )
    return messages

class OpenAIModel:
    def __init__(self, system_prompt, temperature):
        self.temperature = temperature
        self.system_prompt = system_prompt

        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = os.getenv("OPENAI_MODEL")
            
    def generate_json(self, prompt):
        try:
            input_tokens_length = num_tokens_from_string(self.system_prompt + prompt)
            print("input tokens length", input_tokens_length)
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=10000,
                model=self.model, 
                response_format={ "type": "json_object" }
            )
            
            response = chat_completion.choices[0].message.content
            output_tokens_length = num_tokens_from_string(response)
            print("output tokens length", output_tokens_length)
            return response, input_tokens_length, output_tokens_length
        
        except Exception as e:
            response = {"error": f"Error in invoking model! {str(e)}"}
            print(response)
            return response
        
    def generate_string(self, prompt):
        try:
            input_tokens_length = num_tokens_from_string(self.system_prompt + prompt)
            print("input tokens length", input_tokens_length)
            
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                model=self.model, 
                max_tokens=11000,
                response_format={ "type": "json_object" }
            )
            
            response = chat_completion.choices[0].message.content
            output_tokens_length = num_tokens_from_string(response)
            print("output tokens length", output_tokens_length)
            return response, input_tokens_length, output_tokens_length
        
        except Exception as e:
            response = {"error": f"Error in invoking model! {str(e)}"}
            print(response)
            return response
    def process_image(self, image_paths):
    
        image_messages = create_image_message(image_paths)
        
        request_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"{self.system_prompt}",
                        },
                        *image_messages,
                    ],
                }
            ]
        input_tokens = num_tokens_from_string(self.system_prompt)
        print(f"Input tokens: {input_tokens}")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,
            )
            output_text = response.choices[0].message.content
            output_tokens = num_tokens_from_string(output_text)
            print(f"Output tokens: {output_tokens}")
    
        except Exception as e:
            print(f"error: {e}")
        
        return output_text, input_tokens, output_tokens
        
        