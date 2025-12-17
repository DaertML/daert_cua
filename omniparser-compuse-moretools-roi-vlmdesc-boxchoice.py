from pydantic import BaseModel, Field, ValidationError, validator, ConfigDict
from typing import Literal, Union, Optional
import time
import numpy as np
import pyautogui
import json
import os
import re
# NOTE: Assuming 'utils' is available in the environment and contains these functions.
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model 
device = 'cuda'
from PIL import ImageGrab
BOX_TRESHOLD = 0.03
import requests 

# --- Existing Helper Functions ---

def simple_extract_json_from_text(text):
    """Simple JSON extraction using regex."""
    json_pattern = re.compile(r'\{.*?\}')
    json_match = json_pattern.search(text)
    
    if json_match:
        json_str = json_match.group()
        try:
            json_dict = json.loads(json_str)
            return json_dict
        except json.JSONDecodeError:
            print("Invalid JSON format")
            return None
    else:
        print("No JSON object found in the text")
        return None

def extract_json_from_text(response: str) -> dict:
    """Robust JSON extraction from model response."""
    # First try: Extract JSON using regex
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    
    # Second try: Look for JSON-like structure after replacing common LLM output mistakes
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
    """Helper function to call Ollama API for generation."""
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

# --- Pydantic Action Models (Execution Interface - requires ABSOLUTE coordinates) ---

class MouseClick(BaseModel):
    action: Literal["click", "right_click", "double_click"]
    coordinates: list = Field(..., description="List of [x, y] coordinates (absolute screen coordinates)") 

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('coordinates', pre=True)
    def validate_coordinates(cls, value):
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(c, (int, float)) for c in value):
            raise ValueError("Coordinates must be a list of exactly two numbers [x, y]")
        return value

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

class FillAction(BaseModel):
    action: Literal["fill"]
    coordinates: list = Field(..., description="List of [x, y] coordinates (absolute screen coordinates) for text field")
    data: str = Field(..., min_length=1, description="Text to input")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @validator('coordinates', pre=True)
    def validate_coordinates(cls, value):
        if not isinstance(value, list) or len(value) != 2 or not all(isinstance(c, (int, float)) for c in value):
            raise ValueError("Coordinates must be a list of exactly two numbers [x, y]")
        return value

    @property
    def x(self) -> float:
        return self.coordinates[0]

    @property
    def y(self) -> float:
        return self.coordinates[1]

# --- NEW Pydantic Action Models (LLM Output Interface - uses box_index) ---

class ClickBox(BaseModel):
    action: Literal["click", "right_click", "double_click"]
    box_index: int = Field(..., description="The index of the UI element box to click.") 

class FillBox(BaseModel):
    action: Literal["fill"]
    box_index: int = Field(..., description="The index of the UI element box to fill.")
    data: str = Field(..., min_length=1, description="Text to input")

# --- Non-Coordinate Actions (Remain the same) ---

class ScrollAction(BaseModel):
    action: Literal["scroll"]
    direction: Literal["up", "down"]
    amount: int = Field(1, description="Number of scroll steps")

class RunCommand(BaseModel):
    action: Literal["runcmd"]
    command: str = Field(..., min_length=1, description="Command to run on the terminal")

class GiveResults(BaseModel):
    action: Literal["give_results"]
    results:  str = Field(..., min_length=1, description="Verbose description of the results obtained during the actions")

class DoneAction(BaseModel):
    action: Literal["done"]

ActionType = Union[ClickBox, FillBox, MouseClick, ScrollAction, FillAction, DoneAction, GiveResults, RunCommand]


class ComputerAgent:
    def __init__(self, screenshot_roi: Optional[tuple] = None):
        self.max_steps = 100
        self.current_step = 0
        self.last_coordinates = None
        
        # Store the VLM parsed content (raw dicts with 'box_2d') for coordinate lookup
        self.last_parsed_content: Optional[list[dict]] = None 
        
        pyautogui.PAUSE = 0.5
        pyautogui.FAILSAFE = False
        
        # Store the ROI (left, top, right, bottom)
        self.screenshot_roi = screenshot_roi 
        
        # Set up offsets (global screen coordinates)
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
        """Execute validated action using pyautogui (coordinate-based)."""
        print(f"Executing action: {action}")
        
        if isinstance(action, MouseClick):
            print(f"Clicking at absolute screen coordinates ({action.x}, {action.y})")
            
            # Move mouse first, then perform the click
            pyautogui.moveTo(action.x, action.y, duration=0.1)
            time.sleep(0.1)  # Small delay to ensure movement is complete
            
            if action.action == "click":
                pyautogui.click()
            elif action.action == "right_click":
                pyautogui.rightClick()
            elif action.action == "double_click":
                pyautogui.doubleClick()
                
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
            
            # Move mouse first, then click and type
            pyautogui.moveTo(action.x, action.y, duration=0.1)
            time.sleep(0.1)
            pyautogui.click()
            time.sleep(0.1)
            pyautogui.write(action.data)
            return True

        elif isinstance(action, DoneAction):
            print("Task completed successfully!")
            return True
        
        elif isinstance(action, GiveResults):
            print("Obtained results:", action.results)
            return True
        
        elif isinstance(action, RunCommand):
            confirm = input("Do you want to run? (y/n): "+action.command+"\n")
            if confirm == "y":
                # NOTE: os.system result is often non-zero on failure, but we assume success if confirmed
                os.system(action.command) 
                return True
            else:
                print("Cancelling command execution")
                return False

        print(f"Unknown action type: {type(action)}")
        return False

    def parse_action(self, response: str) -> Optional[Union[MouseClick, FillAction, ScrollAction, DoneAction, GiveResults, RunCommand]]:
        """Parse LLM response, resolve box_index to absolute coordinates, and validate/transform."""
        try:
            json_data = extract_json_from_text(response)
            print("Extracted JSON", json_data)
            if not json_data:
                raise ValueError("No valid JSON found in response")
                
            # The list of possible models to try for parsing the raw JSON
            parsing_models = [ClickBox, FillBox, MouseClick, ScrollAction, FillAction, DoneAction, GiveResults, RunCommand]
            
            # 1. First, try to parse the JSON into a known Pydantic model
            parsed_action = None
            for model in parsing_models:
                try:
                    parsed_action = model(**json_data)
                    break
                except ValidationError as e:
                    print(f"Failed to parse as {model}: {e}")
                    continue
            
            if not parsed_action:
                raise ValueError("No matching action type found in JSON data.")
                
            print(f"Successfully parsed action: {type(parsed_action)} - {parsed_action}")
                
            # 2. Check if the parsed action is Box-based and needs coordinate resolution
            if isinstance(parsed_action, (ClickBox, FillBox)):
                
                box_index = parsed_action.box_index
                
                if not self.last_parsed_content:
                    raise ValueError("Cannot execute box-based action: VLM data (last_parsed_content) is missing.")
                
                # Check if index is valid
                if not 0 <= box_index < len(self.last_parsed_content):
                    raise ValueError(f"Box index {box_index} is out of bounds (0 to {len(self.last_parsed_content) - 1})")
                    
                # Extract box data: 'box_2d' contains [xmin, ymin, xmax, ymax] in LOCAL (ROI) coordinates
                box_data = self.last_parsed_content[box_index]
                
                # Handle both dictionary and string formats for box_data
                if isinstance(box_data, dict):
                    box_coords = box_data.get("box_2d", [0, 0, 0, 0])
                else:
                    # If it's a string or other type, try to extract coordinates from description
                    print(f"Warning: Box data is not a dictionary: {type(box_data)}")
                    # Use a default position (center of ROI) as fallback
                    roi_width = self.roi_right - self.roi_left
                    roi_height = self.roi_bottom - self.roi_top
                    box_coords = [roi_width * 0.4, roi_height * 0.4, roi_width * 0.6, roi_height * 0.6]
                
                xmin, ymin, xmax, ymax = box_coords
                
                # Calculate the centroid (local to ROI)
                x_rel = (xmin + xmax) / 2.0
                y_rel = (ymin + ymax) / 2.0
                
                # Transform to Global Coordinates
                x_abs = x_rel + self.offset_x
                y_abs = y_rel + self.offset_y
                
                # Add coordinate validation and clamping
                x_abs_clamped = max(self.roi_left, min(x_abs, self.roi_right))
                y_abs_clamped = max(self.roi_top, min(y_abs, self.roi_bottom))

                if x_abs != x_abs_clamped or y_abs != y_abs_clamped:
                     print(f"Warning: Clamped coordinates. Global [{x_abs}, {y_abs}] -> Clamped [{x_abs_clamped}, {y_abs_clamped}]")
                
                # 3. Construct the executable action (MouseClick or FillAction)
                resolved_action_data = {
                    "action": parsed_action.action,
                    "coordinates": [x_abs_clamped, y_abs_clamped], # ABSOLUTE coordinates
                }
                
                print(f"Resolved Box {box_index}: Local Centroid [{x_rel:.2f}, {y_rel:.2f}] -> Global Clamped [{x_abs_clamped:.2f}, {y_abs_clamped:.2f}]")

                if isinstance(parsed_action, FillBox):
                    resolved_action_data["data"] = parsed_action.data
                    return FillAction(**resolved_action_data)
                else:
                    return MouseClick(**resolved_action_data)
            
            # 4. If the action was not box-based, return the original parsed_action
            return parsed_action
            
        except Exception as e:
            print(f"Action parsing error: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def get_ui_elements_from_screenshot(self):
        """Get UI elements from screenshot with proper error handling."""
        try:
            # 1. Capture screenshot of the ROI
            if self.screenshot_roi:
                screenshot = ImageGrab.grab(bbox=self.screenshot_roi)
            else:
                screenshot = ImageGrab.grab()  # Full screen
            screenshot.save('screenshot.png')
            
            # 2. Get UI elements and VLM descriptions from screenshot
            ocr_bbox_rslt, _ = check_ocr_box('screenshot.png', False, 'xyxy')
            text, ocr_bbox = ocr_bbox_rslt
            
            # 3. Call get_som_labeled_img and handle return values properly
            result = get_som_labeled_img( 
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
            
            # *** FIX: Handle the return values properly ***
            print(f"get_som_labeled_img returned {len(result)} items")
            
            # Debug the structure safely
            for i, item in enumerate(result):
                item_type = type(item)
                if hasattr(item, '__len__'):
                    print(f"Item {i} type: {item_type}, length: {len(item)}")
                    # Safely check first element if it's a list/sequence
                    if len(item) > 0 and hasattr(item, '__getitem__'):
                        try:
                            first_elem = item[0]
                            print(f"  First element type: {type(first_elem)}")
                        except (KeyError, IndexError, TypeError):
                            print(f"  Could not access first element")
                else:
                    print(f"Item {i} type: {item_type}, no length attribute")
            
            # Based on the error output, the structure seems to be:
            # Item 0: string (image data?) 
            # Item 1: dictionary (with UI element data)
            # Item 2: list of descriptions?
            
            descriptions_for_prompt = []
            self.last_parsed_content = []
            
            # Try to extract UI element data from the dictionary (item 1)
            if len(result) >= 2 and isinstance(result[1], dict):
                ui_data_dict = result[1]
                print(f"UI data dictionary keys: {list(ui_data_dict.keys())}")
                
                # Convert dictionary to list format for consistent handling
                # The dictionary seems to contain the actual UI element data
                for key, value in ui_data_dict.items():
                    if isinstance(value, dict) and 'box_2d' in value:
                        # This looks like a UI element with coordinates
                        self.last_parsed_content.append(value)
                        # Create description from available data
                        description = value.get('caption', f'UI Element {key}')
                        descriptions_for_prompt.append(description)
                    elif isinstance(value, str):
                        # String description
                        descriptions_for_prompt.append(value)
            
            # If we didn't get data from the dictionary, try other items
            if not self.last_parsed_content and len(result) > 2 and isinstance(result[2], list):
                descriptions_for_prompt = result[2]
                # Create dummy box data for descriptions
                for i, desc in enumerate(descriptions_for_prompt):
                    # Create a default box in the center of the ROI
                    roi_width = self.roi_right - self.roi_left
                    roi_height = self.roi_bottom - self.roi_top
                    default_box = {
                        'box_2d': [roi_width * 0.4, roi_height * 0.4, roi_width * 0.6, roi_height * 0.6],
                        'caption': desc
                    }
                    self.last_parsed_content.append(default_box)
            
            print(f"Extracted {len(self.last_parsed_content)} UI elements")
            return descriptions_for_prompt
            
        except Exception as e:
            print(f"Error getting UI elements: {e}")
            import traceback
            traceback.print_exc()
            return []

    def run_task(self, task: str):
        """
        Main agent loop to execute tasks, using VLM/OCR data in the LLM prompt.
        """
        print(f"Starting task: {task}")
        self.current_step = 0
        
        pyautogui.FAILSAFE = True
        
        while self.current_step < self.max_steps:
            self.current_step += 1
            print(f"\n--- Step {self.current_step}/{self.max_steps} ---")
            
            # Get UI elements with proper error handling
            descriptions_for_prompt = self.get_ui_elements_from_screenshot()
            
            # Generate prompt for LLM, including VLM/OCR results with INDEXES
            ui_elements_description = ""
            
            if descriptions_for_prompt and isinstance(descriptions_for_prompt, list):
                for i, description in enumerate(descriptions_for_prompt):
                    if isinstance(description, str):
                        ui_elements_description += f"Box {i}: {description}\n"
                    else:
                        ui_elements_description += f"Box {i}: {str(description)}\n"
            elif self.last_parsed_content:
                # Fallback: build from parsed content
                for i, content_data in enumerate(self.last_parsed_content):
                    if isinstance(content_data, dict):
                        description = content_data.get('caption', f'UI Element {i}')
                        ui_elements_description += f"Box {i}: {description}\n"
                    else:
                        ui_elements_description += f"Box {i}: {str(content_data)}\n"
            else:
                ui_elements_description = "No UI elements detected."
            
            # If no UI elements were found, provide a fallback prompt
            if ui_elements_description == "No UI elements detected.":
                prompt = f"""
                You are an AI assistant that interacts with computer interfaces. 
                Current task: {task}

                No UI elements were detected in the screenshot. This might mean:
                1. The screen is blank or the application didn't load
                2. The detection model failed to recognize elements
                3. You need to perform a different action first

                Available actions:
                - Scroll: {{"action": "scroll", "direction": "up|down", "amount": <int>}}
                - Run Command: {{"action": "runcmd", "command": "<text>"}}
                - Task complete: {{"action": "done"}}

                Suggest an appropriate action given the task: {task}
                """
            else:
                prompt = f"""
                You are an AI assistant that interacts with computer interfaces. 
                Current task: {task}

                Available UI Elements (VLM/OCR Output):
                {ui_elements_description}
                
                Solve the problem with the least of effort; running commands take less than using the computer mouse and keyboard.

                Output ONLY a JSON object with one of these structures:
                - Click: {{"action": "click", "box_index": <int>}}
                - Right click: {{"action": "right_click", "box_index": <int>}}
                - Double click: {{"action": "double_click", "box_index": <int>}}
                - Scroll: {{"action": "scroll", "direction": "up|down", "amount": <int>}}
                - Fill text: {{"action": "fill", "box_index": <int>, "data": "<text>"}}
                - Task complete: {{"action": "done"}}
                - Give results: {{"action": "give_results", "results": "<text>"}}
                - Run Command: {{"action": "runcmd", "command": "<text>"}}

                Important: 
                - For click and fill actions, you MUST use the 'box_index' from the VLM/OCR Output to specify the target element. DO NOT use 'coordinates'.
                - Only use box_index values that are shown in the Available UI Elements (0 to {max(0, len(descriptions_for_prompt) - 1)}).
                """
            
            print("Prompt:", prompt)
            
            # Get LLM response
            try:
                model = "hf.co/mradermacher/UI-TARS-1.5-7B-GGUF:Q8_0" 
                response = ollama_generate("http://localhost:11434", model, prompt, temp=0.6)
                if response:
                    response = response.replace("```", "").replace("json", "")
                else:
                    response = '{"action": "done"}'
            except Exception as e:
                 print(f"Ollama call failed, using dummy response: {e}")
                 response = '{"action": "done"}'
            
            print("LLM Response:", response)

            # Process and execute action
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

# --- ROI Selection Function ---

def get_user_roi() -> Optional[tuple]:
    """Prompts the user to select a region of interest using two points recorded via pyautogui.position()."""
    print("\n--- Screen Region of Interest (ROI) Setup ---")
    
    use_roi = input("Do you want to define a specific screen rectangle (ROI) for the agent? (y/n): ").lower()
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

# --- Initialization Block ---
if __name__ == "__main__":
    # Initialize models (from your existing code)
    print("Initializing models...")
    # NOTE: These paths must exist in your environment
    som_model = get_yolo_model(model_path='weights/icon_detect/best.pt').to(device)
    caption_model_processor = get_caption_model_processor(
        model_name="florence2", 
        model_name_or_path="weights/icon_caption_florence", 
        device=device
    )
    
    # Get ROI and pass it to the agent
    roi_coords = get_user_roi()
    
    # Create agent
    agent = ComputerAgent(screenshot_roi=roi_coords)
    
    # Get user task
    user_task = input("Enter the task you want me to perform: ")
    
    # Run task
    agent.run_task(user_task)