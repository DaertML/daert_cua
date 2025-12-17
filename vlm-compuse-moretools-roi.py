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
import os # Added back for runcmd (though runcmd is not in ActionType, good practice)

# --- Existing Helper Functions (omitted for brevity) ---

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

# --- Pydantic Action Models ---

class MouseClick(BaseModel):
    action: Literal["click", "right_click", "double_click"]
    coordinates: list = Field(..., description="List of [x, y] coordinates (screen-absolute)") 

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('coordinates', pre=True)
    def validate_coordinates(cls, value):
        if len(value) != 2 or not all(isinstance(c, (int, float)) for c in value):
            raise ValueError("Coordinates must be a list of exactly two numbers [x, y]")
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
    amount: int = Field(1, description="Number of scroll steps")

class FillAction(BaseModel):
    action: Literal["fill"]
    coordinates: list = Field(..., description="List of [x, y] coordinates (screen-absolute) for text field")
    data: str = Field(..., min_length=1, description="Text to input")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('coordinates', pre=True)
    def validate_coordinates(cls, value):
        if len(value) != 2 or not all(isinstance(c, (int, float)) for c in value):
            raise ValueError("Coordinates must be a list of exactly two numbers [x, y]")
        return value

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

class DoneAction(BaseModel):
    action: Literal["done"]

class GiveResults(BaseModel):
    action: Literal["give_results"]
    results:  str = Field(..., min_length=1, description="Verbose description of the results obtained during the actions")

ActionType = Union[MouseClick, ScrollAction, FillAction, DoneAction, GiveResults]

class ComputerAgent:
    def __init__(self, screenshot_roi: Optional[tuple] = None):
        self.max_steps = 100
        self.current_step = 0
        self.last_coordinates = None
        pyautogui.PAUSE = 0.5
        
        pyautogui.FAILSAFE = False
        
        self.screenshot_roi = screenshot_roi 
        
        # Set up offsets
        self.offset_x = self.screenshot_roi[0] if self.screenshot_roi else 0
        self.offset_y = self.screenshot_roi[1] if self.screenshot_roi else 0
        
        # Store ROI boundaries for clamping (global coordinates)
        if self.screenshot_roi:
            self.roi_left, self.roi_top, self.roi_right, self.roi_bottom = self.screenshot_roi
        else:
            # If no ROI, use full screen size for bounds
            screen_width, screen_height = pyautogui.size()
            self.roi_left, self.roi_top = 0, 0
            self.roi_right, self.roi_bottom = screen_width, screen_height
        
        print(f"Agent initialized with ROI: {self.screenshot_roi}, Offset: ({self.offset_x}, {self.offset_y})")


    def execute_action(self, action: ActionType):
        """Execute validated action using pyautogui"""
        if isinstance(action, MouseClick):
            print(f"Clicking at absolute screen coordinates ({action.x}, {action.y})")
            
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
            print(f"Typing '{action.data}' at absolute screen coordinates ({action.x}, {action.y})")
            pyautogui.click(action.x, action.y)
            pyautogui.write(action.data)
            return True

        elif isinstance(action, DoneAction):
            print("Task completed successfully!")
            return True
        
        elif isinstance(action, GiveResults):
            print("Obtained results:", action.results)
            return True

        return False

    def parse_action(self, response: str) -> ActionType:
        """Parse and validate LLM response, adjusting coordinates for ROI using agent's offset, and clamping."""
        try:
            json_data = extract_json_from_text(response)
            print("Extracted JSON", json_data)
            if not json_data:
                raise ValueError("No valid JSON found in response")
                
            action_type = json_data.get("action")
            
            if action_type in ["fill", "click", "right_click", "double_click"]:
                
                local_coords = json_data.get("coordinates")
                if not isinstance(local_coords, list) or len(local_coords) != 2:
                    raise ValueError(f"Action '{action_type}' requires a 'coordinates' list of [x, y].")
                    
                x_rel, y_rel = float(local_coords[0]), float(local_coords[1])
                
                # 1. Transform: Adjust coordinates from local (ROI-based) to global (screen-based)
                x_abs = x_rel + self.offset_x
                y_abs = y_rel + self.offset_y
                
                # 2. Clamping: Ensure the global coordinates are within the ROI boundaries
                # Clamp X-coordinate
                x_abs_clamped = max(self.roi_left, min(x_abs, self.roi_right))
                # Clamp Y-coordinate
                y_abs_clamped = max(self.roi_top, min(y_abs, self.roi_bottom))

                if x_abs != x_abs_clamped or y_abs != y_abs_clamped:
                     print(f"Warning: Clamped coordinates. Global [{x_abs}, {y_abs}] -> Clamped [{x_abs_clamped}, {y_abs_clamped}]")
                
                # Overwrite the 'coordinates' field with the clamped global, absolute coordinates
                json_data["coordinates"] = [x_abs_clamped, y_abs_clamped]
                print(f"Adjusted coords: Local [{x_rel}, {y_rel}] -> Global Clamped [{x_abs_clamped}, {y_abs_clamped}]")

            # Try all possible action types
            for model in [MouseClick, ScrollAction, FillAction, DoneAction, GiveResults]:
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
        
        pyautogui.FAILSAFE = True
        
        while self.current_step < self.max_steps:
            self.current_step += 1
            print(f"\n--- Step {self.current_step}/{self.max_steps} ---")
            
            # Capture screenshot of the ROI
            screenshot = ImageGrab.grab(bbox=self.screenshot_roi)
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
            - Give results: {{"action": "give_results", "results": "<text>"}}

            Important: 
            - The screenshot provided to you is cropped. All coordinates [x, y] MUST be relative to the top-left corner of the image you see (i.e., local coordinates starting at [0, 0]).
            - Coordinates must be a list of two numbers [x, y].
            """

            # Generate prompt for LLM
            prompt = f"""
            You are an AI assistant that interacts with computer interfaces. 
            Current task: {task}
            
            Output ONLY a JSON object with one of these structures:
            - Click: {{"action": "click", "coordinates": [x, y]}}

            Important: 
            - The screenshot provided to you is cropped. All coordinates [x, y] MUST be relative to the top-left corner of the image you see (i.e., local coordinates starting at [0, 0]).
            - Coordinates must be a list of two numbers [x, y].
            """
            
            print("Prompt:", prompt)
            
            # Assuming LLM initialization is handled externally or not needed here
            # model = "qwen3-vl:4b"
            # response = ollama_generate("http://localhost:11434", model, prompt, temp=0.6)
            # Placeholder for actual LLM response if needed for testing:
            # response = '{"action": "click", "coordinates": [150, 200]}' 
            # response = response.replace("```", "").replace("json", "")
            
            # Using placeholder LLM call for completeness
            # NOTE: You must uncomment and configure your actual LLM call here:
            try:
                model = "qwen3-vl:4b" # Replace with your active model
                model = "hf.co/mradermacher/UI-TARS-1.5-7B-GGUF:Q8_0"

                response = ollama_generate("http://localhost:11434", model, prompt, temp=0.6)
                response = response.replace("```", "").replace("json", "")
            except Exception as e:
                 print(f"Ollama call failed, using dummy response: {e}")
                 response = '{"action": "done"}' # Fallback
            
            print("LLM Response:", response)
            
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
                
            time.sleep(1)
        
        print("Maximum steps reached. Task incomplete.")
        return False

# --- (get_user_roi() remains unchanged) ---

def get_user_roi() -> Optional[tuple]:
    """Prompts the user to select a region of interest using two points recorded via pyautogui.position()."""
    print("\n--- Screen Region of Interest (ROI) Setup ---")
    
    use_roi = input("Do you want to define a screen rectangle (ROI) for the agent? (y/n): ").lower()
    if use_roi != 'y':
        print("Using full screen for capture.")
        return None

    # Step 1: Get Top-Left Corner
    print("\nStarting 2-Point ROI Selection:")
    
    # Pause and wait for user to position mouse
    input("1. Move your mouse to the **TOP-LEFT** corner of the area, then press **ENTER** here.")
    
    # Record the position
    x1, y1 = pyautogui.position()
    print(f"Top-Left corner recorded: ({x1}, {y1})")
    
    # Step 2: Get Bottom-Right Corner
    # Pause and wait for user to position mouse
    input("2. Move your mouse to the **BOTTOM-RIGHT** corner of the area, then press **ENTER** here.")

    # Record the position
    x2, y2 = pyautogui.position()
    print(f"Bottom-Right corner recorded: ({x2}, {y2})")

    # 3. Calculate the bounding box (left, top, right, bottom)
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    
    roi = (left, top, right, bottom)
    print(f"\nFinal ROI selected: {roi}")
    return roi

# Initialize agent
if __name__ == "__main__":
    # Get ROI and pass it to the agent
    roi_coords = get_user_roi()

    # Create agent, which stores the offset and boundaries
    agent = ComputerAgent(screenshot_roi=roi_coords)
    
    # Get user task
    user_task = input("Enter the task you want me to perform: ")
    
    # Run task
    agent.run_task(user_task)