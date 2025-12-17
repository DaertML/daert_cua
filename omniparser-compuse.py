from pydantic import BaseModel, Field, ValidationError,validator,ConfigDict
from typing import Literal, Union, Optional
import time
import numpy as np
import pyautogui
import json
import re
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
device = 'cuda'
from PIL import ImageGrab
BOX_TRESHOLD = 0.03
import requests 
def simple_extract_json_from_text(text):
    # Use a regular expression to extract JSON object from the text
    json_pattern = re.compile(r'\{.*?\}')
    json_match = json_pattern.search(text)
    
    if json_match:
        json_str = json_match.group()
        try:
            # Convert JSON string to Python dictionary
            json_dict = json.loads(json_str)
            return json_dict
        except json.JSONDecodeError:
            print("Invalid JSON format")
            return None
    else:
        print("No JSON object found in the text")
        return None

def extract_json_from_text(response: str) -> dict:
    """Robust JSON extraction from model response"""
    # First try: Extract JSON using regex
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Second try: Look for JSON-like structure
    json_str = response.replace("'", '"').replace("True", "true").replace("False", "false")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # Final try: Extract the last JSON-like block
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
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        print("Raw Response",response.json())
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

# Define action models using Pydantic

class MouseClick(BaseModel):
    action: Literal["click", "right_click", "double_click"]
    id: int = Field(..., description="Element ID to interact with")
    coordinates: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('coordinates', pre=True)
    def validate_coordinates_shape(cls, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        
        if value.ndim != 1 or value.shape[0] < 2:
            raise ValueError("coordinates must be a 1-dimensional array with at least two values (for x and y).")
            
        return value

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

class ScrollAction(BaseModel):
    action: Literal["scroll"]
    direction: Literal["up", "down"]
    #amount: int = Field(1, ge=1, le=10, description="Number of scroll steps")
    amount: int = Field(1, description="Number of scroll steps")

class FillAction(BaseModel):

    action: Literal["fill"]
    id: int = Field(..., description="Text field element ID")
    data: str = Field(..., min_length=1, description="Text to input")
    coordinates: np.ndarray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('coordinates', pre=True)
    def validate_coordinates_shape(cls, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        
        if value.ndim != 1 or value.shape[0] < 2:
            raise ValueError("coordinates must be a 1-dimensional array with at least two values (for x and y).")
            
        return value

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
        pyautogui.PAUSE = 0.5  # Add delay between actions
        
        pyautogui.FAILSAFE = True  # Enable failsafe

    def execute_action(self, action: ActionType, label_coordinates: dict):
        """Execute validated action using pyautogui"""
        if isinstance(action, MouseClick):
            # Coordinates are already on the object, no need to look up
            print(f"Clicking at ({action.x}, {action.y}) - ID: {action.id}")
            
            element_id = str(action.id)
            if element_id not in label_coordinates:
                print(f"Element ID {element_id} not found")
                return False
                    

            if action.action == "click":
                pyautogui.click(action.x, action.y)
            elif action.action == "right_click":
                pyautogui.rightClick(action.x, action.y)
            elif action.action == "double_click":
                pyautogui.doubleClick(action.x, action.y)
                
            self.last_coordinates = (action.x, action.y)
            return True

        elif isinstance(action, ScrollAction):
            # Normalize scroll amount to reasonable range
            scroll_steps = min(max(action.amount, 1), 20)  # Cap between 1-20 steps
            pixel_amount = scroll_steps * (120 if action.direction == "up" else -120)
            print(f"Scrolling {action.direction} {scroll_steps} steps")
            pyautogui.scroll(pixel_amount)
            return True

        elif isinstance(action, FillAction):
            element_id = str(action.id)
            if element_id not in label_coordinates:
                print(f"Text field ID {element_id} not found")
                return False
            
            #print("Label coord:",label_coordinates[element_id])
            #x, y = label_coordinates[element_id]
            
            print(f"Typing '{action.data}' at ({action.x}, {action.y})")
            pyautogui.click(action.x, action.y)
            pyautogui.write(action.data)
            return True

        elif isinstance(action, DoneAction):
            print("Task completed successfully!")
            return True

        return False

    def parse_action(self, response: str, label_coordinates: dict) -> ActionType:
        """Parse and validate LLM response using Pydantic"""
        try:
            json_data = extract_json_from_text(response)
            print("Extracted JSON",json_data)
            if not json_data:
                raise ValueError("No valid JSON found in response")
                
            # Manually handle MouseClick and add coordinates
            action_type = json_data.get("action")
            if action_type in ["fill", "click", "right_click", "double_click"]:
                element_id = str(json_data.get("id"))
                if element_id in label_coordinates:
                    json_data["coordinates"] = label_coordinates[element_id]
                else:
                    raise ValueError(f"ID {element_id} not found in available coordinates.")

            
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
            
            # Capture and process screenshot
            screenshot = ImageGrab.grab()
            screenshot.save('screenshot.png')
            
            # Get UI elements from screenshot
            ocr_bbox_rslt, _ = check_ocr_box('screenshot.png', False, 'xyxy')
            text, ocr_bbox = ocr_bbox_rslt
            
            # FIX: Use named parameters and correct order for get_som_labeled_img
            _, label_coordinates, parsed_content_list = get_som_labeled_img(
                img_path='screenshot.png',
                model=som_model,
                BOX_TRESHOLD=BOX_TRESHOLD,
                output_coord_in_ratio=False,
                ocr_bbox=ocr_bbox,
                draw_bbox_config=draw_bbox_config,
                caption_model_processor=caption_model_processor,
                ocr_text=text,
                use_local_semantics=True,
                iou_threshold=0.1
            )
            
            # Generate prompt for LLM
            prompt = f"""
            You are an AI assistant that interacts with computer interfaces. 
            Current task: {task}
            Available UI elements: {", ".join(parsed_content_list)}
            
            Output ONLY a JSON object with one of these structures:
            - Click: {{"action": "click", "id": <element_id>}}
            - Right click: {{"action": "right_click", "id": <element_id>}}
            - Double click: {{"action": "double_click", "id": <element_id>}}
            - Scroll: {{"action": "scroll", "direction": "up|down", "amount": <int>}}
            - Fill text: {{"action": "fill", "id": <element_id>, "data": "<text>"}}
            - Task complete: {{"action": "done"}}
            """
            
            print("Prompt:",prompt)
            # Get LLM response

            # so good following instructions
            model = "gpt-oss:20b"

            # clicking here and there
            model = "hf.co/mradermacher/UI-TARS-1.5-7B-GGUF:Q8_0"
            
            # tons of clicking, seems steerable
            #model = "llama3.2"
            
            # finishes task, a bit too much thinking
            #model = "danielsheep/Qwen3-30B-A3B-Thinking-2507-Unsloth:latest"
            
            #model = "Qwen3:14b"
            #model = "hf.co/mradermacher/GUI-Owl-7B-GGUF"
            
            response = ollama_generate("http://localhost:11434", model, prompt, temp=0.6)

            print("LLM Response:", response)

            # Process response

            # llama3.2 post process
            response = response.replace("```","")
            response = response.replace("json","")
            response = response.replace("Here is the JSON object with one of the specified structures:","")

            # Pass label_coordinates to the parser
            action = self.parse_action(response, label_coordinates)
            if not action:
                print("Invalid action format. Retrying...")
                continue
                
            if isinstance(action, DoneAction):
                print("Task marked as complete by agent")
                return True

            # Now, the 'action' object is already validated and has coordinates
            success = self.execute_action(action, label_coordinates) # pass the full dict, not just the single coord
            if not success:
                print("Action execution failed. Retrying...")
                
            time.sleep(1)  # Allow UI to update
        
        print("Maximum steps reached. Task incomplete.")
        return False


# Initialize agent and models
if __name__ == "__main__":
    # Initialize models (from your existing code)
    som_model = get_yolo_model(model_path='weights/icon_detect/best.pt').to(device)
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="weights/icon_caption_florence", 
        device=device
    )
    
    # Create agent
    agent = ComputerAgent()
    
    # Get user task
    user_task = input("Enter the task you want me to perform: ")
    
    # Run task
    agent.run_task(user_task)
