from pydantic import BaseModel, Field, ValidationError, validator, ConfigDict
from typing import Literal, Union, Optional
import time
import numpy as np
import pyautogui
import json
import re
device = 'cuda'
from PIL import ImageGrab
BOX_TRESHOLD = 0.03
import requests 

def simple_extract_json_from_text(text):
    json_pattern = re.compile(r'\{.*?\}')
    json_match = json_pattern.search(text)
    
    if json_match:
        json_str = json_match.group()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Invalid JSON format")
            return None
    else:
        print("No JSON object found in the text")
        return None

def extract_json_from_text(response: str) -> dict:
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    json_str = response.replace("'", '"').replace("True", "true").replace("False", "false")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    json_candidates = re.findall(r'\{[^{}]*\}', response)
    for candidate in reversed(json_candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    
    print(f"No valid JSON found in response: {response}")
    return None

def ollama_generate(ollama_url, model_name, prompt, temp=0.6):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temp,
            'num_ctx': 8192
        }
    }

    try:
        response = requests.post(
            f"{ollama_url}/api/generate",
            data=json.dumps(data),
            headers=headers
        )
        response.raise_for_status()
        print("Raw Response", response.json())
        return response.json()["response"]
    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama API: {e}")
        return None

draw_bbox_config = {
    'text_scale': 0.8,
    'text_thickness': 2,
    'text_padding': 3,
    'thickness': 3,
}

# Updated action models - removed id field
class MouseClick(BaseModel):
    action: Literal["click", "right_click", "double_click"]
    coordinates: list = Field(..., description="List of [x, y] coordinates to interact with")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('coordinates', pre=True)
    def validate_coordinates(cls, value):
        if len(value) != 2:
            raise ValueError("Coordinates must be a list of exactly two values [x, y]")
        return [float(value[0]), float(value[1])]

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

class ScrollAction(BaseModel):
    action: Literal["scroll"]
    direction: Literal["up", "down"]
    amount: int = Field(1, description="Number of scroll steps")

class FillAction(BaseModel):
    action: Literal["fill"]
    coordinates: list = Field(..., description="List of [x, y] coordinates for text field")
    data: str = Field(..., min_length=1, description="Text to input")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('coordinates', pre=True)
    def validate_coordinates(cls, value):
        if len(value) != 2:
            raise ValueError("Coordinates must be a list of exactly two values [x, y]")
        return [float(value[0]), float(value[1])]

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

class DoneAction(BaseModel):
    action: Literal["done"]

ActionType = Union[MouseClick, ScrollAction, FillAction, DoneAction]

class ComputerAgent:
    def __init__(self):
        self.max_steps = 10
        self.current_step = 0
        self.last_coordinates = None
        pyautogui.PAUSE = 0.5
        pyautogui.FAILSAFE = True

    def execute_action(self, action: ActionType):
        """Execute validated action using pyautogui"""
        if isinstance(action, MouseClick):
            print(f"Clicking at ({action.x}, {action.y})")
            
            if action.action == "click":
                pyautogui.click(action.x, action.y)
            elif action.action == "right_click":
                pyautogui.rightClick(action.x, action.y)
            elif action.action == "double_click":
                pyautogui.doubleClick(action.x, action.y)
                
            self.last_coordinates = (action.x, action.y)
            return True

        elif isinstance(action, ScrollAction):
            scroll_steps = min(max(action.amount, 1), 20)
            pixel_amount = scroll_steps * (120 if action.direction == "up" else -120)
            print(f"Scrolling {action.direction} {scroll_steps} steps")
            pyautogui.scroll(pixel_amount)
            return True

        elif isinstance(action, FillAction):
            print(f"Typing '{action.data}' at ({action.x}, {action.y})")
            pyautogui.click(action.x, action.y)
            pyautogui.write(action.data)
            return True

        elif isinstance(action, DoneAction):
            print("Task completed successfully!")
            return True

        return False

    def parse_action(self, response: str) -> ActionType:
        """Parse and validate LLM response using Pydantic"""
        try:
            json_data = extract_json_from_text(response)
            print("Extracted JSON", json_data)
            if not json_data:
                raise ValueError("No valid JSON found in response")
                
            # Try all possible action types
            for model in [MouseClick, ScrollAction, FillAction, DoneAction]:
                try:
                    return model(**json_data)
                except ValidationError:
                    continue
                    
            raise ValueError("No matching action type found")
        except Exception as e:
            print(f"Action parsing error: {str(e)}")
            return None

    def run_task(self, task: str):
        """Main agent loop to execute tasks"""
        print(f"Starting task: {task}")
        self.current_step = 0
        
        while self.current_step < self.max_steps:
            self.current_step += 1
            print(f"\n--- Step {self.current_step}/{self.max_steps} ---")
            
            # Capture screenshot
            screenshot = ImageGrab.grab()
            screenshot.save('screenshot.png')
            
            # Generate prompt for LLM
            prompt = f"""
            You are an AI assistant that interacts with computer interfaces. 
            Current task: {task}
            
            Output ONLY a JSON object with one of these structures:
            - Click: {{"action": "click", "coordinates": [x, y]}}
            - Right click: {{"action": "right_click", "coordinates": [x, y]}}
            - Double click: {{"action": "double_click", "coordinates": [x, y]}}
            - Scroll: {{"action": "scroll", "direction": "up|down", "amount": <int>}}
            - Fill text: {{"action": "fill", "coordinates": [x, y], "data": "<text>"}}
            - Task complete: {{"action": "done"}}
            
            Important: 
            - Coordinates should be absolute screen coordinates (e.g., [100, 200])
            - Coordinates must be a list of two numbers [x, y]
            """
            
            print("Prompt:", prompt)
            
            # Get LLM response
            # no mahjong :(, but outputs correctly the coords
            model = "gpt-oss:20b"  # Or any other model

            # some failing coords
            model = "hf.co/mradermacher/UI-TARS-1.5-7B-GGUF:Q8_0"

            # great coordinates, no mahjong :(
            #model = "hf.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF:IQ1_S"

            #model = "qwen2.5vl:7b"

            response = ollama_generate("http://localhost:11434", model, prompt, temp=0.6)
            print("LLM Response:", response)

            # Process response
            response = response.replace("```", "").replace("json", "")
            
            action = self.parse_action(response)
            if not action:
                print("Invalid action format. Retrying...")
                continue
                
            if isinstance(action, DoneAction):
                print("Task marked as complete by agent")
                return True

            success = self.execute_action(action)
            if not success:
                print("Action execution failed. Retrying...")
                
            time.sleep(1)  # Allow UI to update
        
        print("Maximum steps reached. Task incomplete.")
        return False


# Initialize agent and models
if __name__ == "__main__":
    agent = ComputerAgent()
    user_task = input("Enter the task you want me to perform: ")
    agent.run_task(user_task)
