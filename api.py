# -*- coding: utf-8 -*-
from openai import OpenAI
import json
import requests
import base64
import requests
from PIL import Image
import io
import json
import os
import re
api_url = "xxx"
api_key = "yyy"

class MultiModalChatSession:
    def __init__(self, system_prompt=""):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.api_url = api_url
        self.api_key = api_key
    
    def _encode_image(self, image_path):
        image = Image.open(image_path)
        
        # Convert to RGB mode (avoid RGBA problems in PNG)
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
            
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def chat(self, user_input, image_paths=None, model="gpt-4o", max_retries=10):
        # Construct message content
        content = [{"type": "text", "text": user_input}]
        
        # Add images (if any)
        if image_paths:
            for img_path in image_paths:
                base64_image = self._encode_image(img_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
        
        # Add to message history
        self.messages.append({"role": "user", "content": content})
        
        for attempt in range(max_retries):
            try:
                payload = {
                    "model": model,
                    "messages": self.messages,
                    "temperature": 1e-5,
                    "max_tokens": 2000
                }
                
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    ai_reply = response_data['choices'][0]['message']['content']
                    
                    # Add to history
                    self.messages.append({"role": "assistant", "content": ai_reply})
                    return ai_reply
                else:
                    print(f"Attempt {attempt+1} failed. Status: {response.status_code}")
                    print(response.text)
            
            except Exception as e:
                print(f"Attempt {attempt+1} error: {str(e)}")
                if attempt == max_retries - 1:
                    # raise Exception(f"Request failed after {max_retries} attempts")
                    assert 0
        
        return "Sorry, I encountered an error."
