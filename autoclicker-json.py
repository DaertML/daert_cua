from PIL import ImageGrab
import re
import requests
import json
from utils import get_som_labeled_img, check_ocr_box, get_caption_model_processor, get_yolo_model
import torch
from ultralytics import YOLO
from PIL import Image
device = 'cuda'

import base64
import matplotlib.pyplot as plt
import io

import pyautogui

def left_click(x,y):
    # Move the mouse to the coordinates
    pyautogui.moveTo(x, y, duration=1)  # duration is the time in seconds it takes for the mouse to move to the coordinates

    # Perform a left-click
    pyautogui.click()

def extract_json_from_text(text):
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

def llamacpp_generate(urlllm, prompt, temp=0.6):
    headers = {"Content-Type": "application/json"}
    data = {"prompt": prompt, "temperature": temp, "n_predict":20}
    data = json.dumps(data)
    res = requests.post(url=urlllm, data=data, headers=headers)
    return res.json()["content"]

som_model = get_yolo_model(model_path='weights/icon_detect/best.pt')
som_model.to(device)
print('model to {}'.format(device))

# two choices for caption model: fine-tuned blip2 or florence2

# caption_model_processor = get_caption_model_processor(model_name="blip2", model_name_or_path="weights/icon_caption_blip2", device=device)
caption_model_processor = get_caption_model_processor(model_name="florence2", model_name_or_path="weights/icon_caption_florence", device=device)

som_model.device, type(som_model) 

cnt = 0
image_path = 'imgs/google_page.png'
# image_path = 'imgs/windows_home.png'
image_path = 'imgs/windows_multitab.png'
draw_bbox_config = {
    'text_scale': 0.8,
    'text_thickness': 2,
    'text_padding': 3,
    'thickness': 3,
}
BOX_TRESHOLD = 0.03

while True:
    # Take a screenshot and save it as 'screenshot.png'
    image_path = 'screenshot.png'

    #image = Image.open(image_path)
    #image_rgb = image.convert('RGB')

    # Capture the screenshot
    screenshot = ImageGrab.grab()

    # Save the screenshot to a file
    screenshot.save(image_path)

    print("Screenshot saved as screenshot.png")

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9})
    text, ocr_bbox = ocr_bbox_rslt

    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(image_path, som_model, BOX_TRESHOLD = BOX_TRESHOLD, output_coord_in_ratio=False, ocr_bbox=ocr_bbox,draw_bbox_config=draw_bbox_config, caption_model_processor=caption_model_processor, ocr_text=text,use_local_semantics=True, iou_threshold=0.1)

    # plot dino_labled_img it is in base64
    plt.figure(figsize=(12,12))

    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    plt.axis('off')

    plt.imshow(image)
    plt.show()
    # print(label_coordinates)
    #print(parsed_content_list)

    prompt = """
You are a robot that makes use of the interface and decides to click on certain buttons on fill in certain fields to complete a user task.
You will receive a text based description of all the elements in the screen, you only output the ID of the button or field to use.

Output the choice in JSON format, it should contain two fields: "action" and "id". In case it is a text field, "data" is also needed.

If it is a button, choose the {"action": "click", "id": id_number} action. If it is a text field, provide the {"action": "fill", "id": id_number, "data": data_string}; giving in data the contents to write in the field.
ID in both cases is the id of the element to interact with from the given list of elements.

Here are some examples:
Task: go to yahoo.com
Screen:
['Text Box ID 0: Task Manager', 'Text Box ID 1: storage', 'Text Box ID 2: InPrivate', 'Text Box ID 3: Google', 'Text Box ID 4: https:/ WWW.googlecom', 'Text Box ID 5: Processes', 'Text Box ID 6: Run new task', 'Text Box ID 7: End task', 'Text Box ID 8: New folder', 'Text Box ID 9: Google', 'Text Box ID 10: finetune/ldm-ft__', 'Text Box ID 11: Gradio', 'Text Box ID 12: Pipelines', 'Text Box ID 13: Recent', 'Text Box ID 14: haotian-liu/LLaVA:', 'Text Box ID 15: Processes', 'Text Box ID 16: 67%', 'Text Box ID 17: 5496', 'Text Box ID 18: Status', 'Text Box ID 19: CPU', 'Text Box ID 20: Memory', 'Text Box ID 21: Disk', 'Text Box ID 22: About', 'Text Box ID 23: Store', 'Text Box ID 24: Gmail', 'Text Box ID 25: Images', 'Text Box ID 26: Sign in', 'Text Box ID 27: Performance', 'Text Box ID 28: Microscft', 'Text Box ID 29: 15.686', 'Text Box ID 30: 1,829,9 MB', 'Text Box ID 31: MBI:', 'Text Box ID 32: Microscft Tean;', 'Text Box ID 33: 142.9MB', 'Text Box ID 34: MBY:', 'Text Box ID 35: App history', 'Text Box ID 36: Microsoft Azure Storage Explo_', 'Text Box ID 37: Efficiency_', 'Text Box ID 38: 0.83', 'Text Box ID 39: 245,0 MB', 'Text Box ID 40: MB/s', 'Text Box ID 41: Startup apps', 'Text Box ID 42: WebViewz Manager', 'Text Box ID 43: 75,9MB', 'Text Box ID 44: MBY:', 'Text Box ID 45: Users', 'Text Box ID 46: Service Host: Storage Service', 'Text Box ID 47: 096', 'Text Box ID 48: 1,1 MB', 'Text Box ID 49: MBY:', 'Text Box ID 50: Details', 'Text Box ID 51: Services', 'Text Box ID 52: Google', 'Text Box ID 53: Google Search', 'Text Box ID 54: Feeling Lucky', 'Text Box ID 55: Discover', 'Text Box ID 56: the ways Chrome keeps you safe while you browse', 'Text Box ID 57: Our third decade of climate action: join us', 'Text Box ID 58: Settings', 'Text Box ID 59: Advertising', 'Text Box ID 60: Business', 'Text Box ID 61: How Search works', 'Text Box ID 62: Privacy', 'Text Box ID 63: Terms', 'Text Box ID 64: Settings', 'Text Box ID 65: 3.53 PM', 'Text Box ID 66: Search', 'Text Box ID 67: Microsoft', 'Text Box ID 68: 10/25/2024', 'Text Box ID 69: Edge', 'Icon Box ID 70: Microsoft Edge browser.', 'Icon Box ID 71: Microsoft 365.', 'Icon Box ID 72: Settings', 'Icon Box ID 73: Image', 'Icon Box ID 74: Image', 'Icon Box ID 75: Microsoft Edge browser.', 'Icon Box ID 76: Microsoft Edge browser.', 'Icon Box ID 77: Teams.', 'Icon Box ID 78: Uncomm&ent Selection', 'Icon Box ID 79: Microsoft OneNote.', 'Icon Box ID 80: Find', 'Icon Box ID 81: Microsoft Outlook.', 'Icon Box ID 82: Image', 'Icon Box ID 83: Maximize', 'Icon Box ID 84: Close', 'Icon Box ID 85: Dictate', 'Icon Box ID 86: Line Spacing', 'Icon Box ID 87: creating a new document or file.', 'Icon Box ID 88: Five-point star', 'Icon Box ID 89: a search function.', 'Icon Box ID 90: Increase', 'Icon Box ID 91: More options', 'Icon Box ID 92: the Windows operating system.', 'Icon Box ID 93: Hyperlink', 'Icon Box ID 94: App launcher or menu.', 'Icon Box ID 95: Health monitoring', 'Icon Box ID 96: Microsoft Outlook.', 'Icon Box ID 97: minimizing a window.', 'Icon Box ID 98: uBlock Origin (Ctrl+T)', 'Icon Box ID 99: Undo', 'Icon Box ID 100: Pentagon', 'Icon Box ID 101: Settings', 'Icon Box ID 102: 1.0%', 'Icon Box ID 103: Back', 'Icon Box ID 104: Rectangle', 'Icon Box ID 105: minimizing a window.', 'Icon Box ID 106: Redo', 'Icon Box ID 107: opening a folder.', 'Icon Box ID 108: Justified', 'Icon Box ID 109: a loading or progress bar.', 'Icon Box ID 110: Label', 'Icon Box ID 111: Google.', 'Icon Box ID 112: Maximize window', 'Icon Box ID 113: Close', 'Icon Box ID 114: Close', 'Icon Box ID 115: Google Chrome web browser.']
Action: {"action":"fill", "data":"yahoo.com", "id":4}

Task: close window
Screen:
['Text Box ID 0: Task Manager', 'Text Box ID 1: storage', 'Text Box ID 2: InPrivate', 'Text Box ID 3: Google', 'Text Box ID 4: https:/ WWW.googlecom', 'Text Box ID 5: Processes', 'Text Box ID 6: Run new task', 'Text Box ID 7: End task', 'Text Box ID 8: New folder', 'Text Box ID 9: Google', 'Text Box ID 10: finetune/ldm-ft__', 'Text Box ID 11: Gradio', 'Text Box ID 12: Pipelines', 'Text Box ID 13: Recent', 'Text Box ID 14: haotian-liu/LLaVA:', 'Text Box ID 15: Processes', 'Text Box ID 16: 67%', 'Text Box ID 17: 5496', 'Text Box ID 18: Status', 'Text Box ID 19: CPU', 'Text Box ID 20: Memory', 'Text Box ID 21: Disk', 'Text Box ID 22: About', 'Text Box ID 23: Store', 'Text Box ID 24: Gmail', 'Text Box ID 25: Images', 'Text Box ID 26: Sign in', 'Text Box ID 27: Performance', 'Text Box ID 28: Microscft', 'Text Box ID 29: 15.686', 'Text Box ID 30: 1,829,9 MB', 'Text Box ID 31: MBI:', 'Text Box ID 32: Microscft Tean;', 'Text Box ID 33: 142.9MB', 'Text Box ID 34: MBY:', 'Text Box ID 35: App history', 'Text Box ID 36: Microsoft Azure Storage Explo_', 'Text Box ID 37: Efficiency_', 'Text Box ID 38: 0.83', 'Text Box ID 39: 245,0 MB', 'Text Box ID 40: MB/s', 'Text Box ID 41: Startup apps', 'Text Box ID 42: WebViewz Manager', 'Text Box ID 43: 75,9MB', 'Text Box ID 44: MBY:', 'Text Box ID 45: Users', 'Text Box ID 46: Service Host: Storage Service', 'Text Box ID 47: 096', 'Text Box ID 48: 1,1 MB', 'Text Box ID 49: MBY:', 'Text Box ID 50: Details', 'Text Box ID 51: Services', 'Text Box ID 52: Google', 'Text Box ID 53: Google Search', 'Text Box ID 54: Feeling Lucky', 'Text Box ID 55: Discover', 'Text Box ID 56: the ways Chrome keeps you safe while you browse', 'Text Box ID 57: Our third decade of climate action: join us', 'Text Box ID 58: Settings', 'Text Box ID 59: Advertising', 'Text Box ID 60: Business', 'Text Box ID 61: How Search works', 'Text Box ID 62: Privacy', 'Text Box ID 63: Terms', 'Text Box ID 64: Settings', 'Text Box ID 65: 3.53 PM', 'Text Box ID 66: Search', 'Text Box ID 67: Microsoft', 'Text Box ID 68: 10/25/2024', 'Text Box ID 69: Edge', 'Icon Box ID 70: Microsoft Edge browser.', 'Icon Box ID 71: Microsoft 365.', 'Icon Box ID 72: Settings', 'Icon Box ID 73: Image', 'Icon Box ID 74: Image', 'Icon Box ID 75: Microsoft Edge browser.', 'Icon Box ID 76: Microsoft Edge browser.', 'Icon Box ID 77: Teams.', 'Icon Box ID 78: Uncomm&ent Selection', 'Icon Box ID 79: Microsoft OneNote.', 'Icon Box ID 80: Find', 'Icon Box ID 81: Microsoft Outlook.', 'Icon Box ID 82: Image', 'Icon Box ID 83: Maximize', 'Icon Box ID 84: Close', 'Icon Box ID 85: Dictate', 'Icon Box ID 86: Line Spacing', 'Icon Box ID 87: creating a new document or file.', 'Icon Box ID 88: Five-point star', 'Icon Box ID 89: a search function.', 'Icon Box ID 90: Increase', 'Icon Box ID 91: More options', 'Icon Box ID 92: the Windows operating system.', 'Icon Box ID 93: Hyperlink', 'Icon Box ID 94: App launcher or menu.', 'Icon Box ID 95: Health monitoring', 'Icon Box ID 96: Microsoft Outlook.', 'Icon Box ID 97: minimizing a window.', 'Icon Box ID 98: uBlock Origin (Ctrl+T)', 'Icon Box ID 99: Undo', 'Icon Box ID 100: Pentagon', 'Icon Box ID 101: Settings', 'Icon Box ID 102: 1.0%', 'Icon Box ID 103: Back', 'Icon Box ID 104: Rectangle', 'Icon Box ID 105: minimizing a window.', 'Icon Box ID 106: Redo', 'Icon Box ID 107: opening a folder.', 'Icon Box ID 108: Justified', 'Icon Box ID 109: a loading or progress bar.', 'Icon Box ID 110: Label', 'Icon Box ID 111: Google.', 'Icon Box ID 112: Maximize window', 'Icon Box ID 113: Close', 'Icon Box ID 114: Close', 'Icon Box ID 115: Google Chrome web browser.']
Action: {"action":"click", "id":84}

Task: maximize window
Screen:
['Text Box ID 0: Task Manager', 'Text Box ID 1: storage', 'Text Box ID 2: InPrivate', 'Text Box ID 3: Google', 'Text Box ID 4: https:/ WWW.googlecom', 'Text Box ID 5: Processes', 'Text Box ID 6: Run new task', 'Text Box ID 7: End task', 'Text Box ID 8: New folder', 'Text Box ID 9: Google', 'Text Box ID 10: finetune/ldm-ft__', 'Text Box ID 11: Gradio', 'Text Box ID 12: Pipelines', 'Text Box ID 13: Recent', 'Text Box ID 14: haotian-liu/LLaVA:', 'Text Box ID 15: Processes', 'Text Box ID 16: 67%', 'Text Box ID 17: 5496', 'Text Box ID 18: Status', 'Text Box ID 19: CPU', 'Text Box ID 20: Memory', 'Text Box ID 21: Disk', 'Text Box ID 22: About', 'Text Box ID 23: Store', 'Text Box ID 24: Gmail', 'Text Box ID 25: Images', 'Text Box ID 26: Sign in', 'Text Box ID 27: Performance', 'Text Box ID 28: Microscft', 'Text Box ID 29: 15.686', 'Text Box ID 30: 1,829,9 MB', 'Text Box ID 31: MBI:', 'Text Box ID 32: Microscft Tean;', 'Text Box ID 33: 142.9MB', 'Text Box ID 34: MBY:', 'Text Box ID 35: App history', 'Text Box ID 36: Microsoft Azure Storage Explo_', 'Text Box ID 37: Efficiency_', 'Text Box ID 38: 0.83', 'Text Box ID 39: 245,0 MB', 'Text Box ID 40: MB/s', 'Text Box ID 41: Startup apps', 'Text Box ID 42: WebViewz Manager', 'Text Box ID 43: 75,9MB', 'Text Box ID 44: MBY:', 'Text Box ID 45: Users', 'Text Box ID 46: Service Host: Storage Service', 'Text Box ID 47: 096', 'Text Box ID 48: 1,1 MB', 'Text Box ID 49: MBY:', 'Text Box ID 50: Details', 'Text Box ID 51: Services', 'Text Box ID 52: Google', 'Text Box ID 53: Google Search', 'Text Box ID 54: Feeling Lucky', 'Text Box ID 55: Discover', 'Text Box ID 56: the ways Chrome keeps you safe while you browse', 'Text Box ID 57: Our third decade of climate action: join us', 'Text Box ID 58: Settings', 'Text Box ID 59: Advertising', 'Text Box ID 60: Business', 'Text Box ID 61: How Search works', 'Text Box ID 62: Privacy', 'Text Box ID 63: Terms', 'Text Box ID 64: Settings', 'Text Box ID 65: 3.53 PM', 'Text Box ID 66: Search', 'Text Box ID 67: Microsoft', 'Text Box ID 68: 10/25/2024', 'Text Box ID 69: Edge', 'Icon Box ID 70: Microsoft Edge browser.', 'Icon Box ID 71: Microsoft 365.', 'Icon Box ID 72: Settings', 'Icon Box ID 73: Image', 'Icon Box ID 74: Image', 'Icon Box ID 75: Microsoft Edge browser.', 'Icon Box ID 76: Microsoft Edge browser.', 'Icon Box ID 77: Teams.', 'Icon Box ID 78: Uncomm&ent Selection', 'Icon Box ID 79: Microsoft OneNote.', 'Icon Box ID 80: Find', 'Icon Box ID 81: Microsoft Outlook.', 'Icon Box ID 82: Image', 'Icon Box ID 83: Maximize', 'Icon Box ID 84: Close', 'Icon Box ID 85: Dictate', 'Icon Box ID 86: Line Spacing', 'Icon Box ID 87: creating a new document or file.', 'Icon Box ID 88: Five-point star', 'Icon Box ID 89: a search function.', 'Icon Box ID 90: Increase', 'Icon Box ID 91: More options', 'Icon Box ID 92: the Windows operating system.', 'Icon Box ID 93: Hyperlink', 'Icon Box ID 94: App launcher or menu.', 'Icon Box ID 95: Health monitoring', 'Icon Box ID 96: Microsoft Outlook.', 'Icon Box ID 97: minimizing a window.', 'Icon Box ID 98: uBlock Origin (Ctrl+T)', 'Icon Box ID 99: Undo', 'Icon Box ID 100: Pentagon', 'Icon Box ID 101: Settings', 'Icon Box ID 102: 1.0%', 'Icon Box ID 103: Back', 'Icon Box ID 104: Rectangle', 'Icon Box ID 105: minimizing a window.', 'Icon Box ID 106: Redo', 'Icon Box ID 107: opening a folder.', 'Icon Box ID 108: Justified', 'Icon Box ID 109: a loading or progress bar.', 'Icon Box ID 110: Label', 'Icon Box ID 111: Google.', 'Icon Box ID 112: Maximize window', 'Icon Box ID 113: Close', 'Icon Box ID 114: Close', 'Icon Box ID 115: Google Chrome web browser.']
Action: {"action":"click", "id":83}

Task: open task manager
Screen:
['Text Box ID 0: Task Manager', 'Text Box ID 1: storage', 'Text Box ID 2: InPrivate', 'Text Box ID 3: Google', 'Text Box ID 4: https:/ WWW.googlecom', 'Text Box ID 5: Processes', 'Text Box ID 6: Run new task', 'Text Box ID 7: End task', 'Text Box ID 8: New folder', 'Text Box ID 9: Google', 'Text Box ID 10: finetune/ldm-ft__', 'Text Box ID 11: Gradio', 'Text Box ID 12: Pipelines', 'Text Box ID 13: Recent', 'Text Box ID 14: haotian-liu/LLaVA:', 'Text Box ID 15: Processes', 'Text Box ID 16: 67%', 'Text Box ID 17: 5496', 'Text Box ID 18: Status', 'Text Box ID 19: CPU', 'Text Box ID 20: Memory', 'Text Box ID 21: Disk', 'Text Box ID 22: About', 'Text Box ID 23: Store', 'Text Box ID 24: Gmail', 'Text Box ID 25: Images', 'Text Box ID 26: Sign in', 'Text Box ID 27: Performance', 'Text Box ID 28: Microscft', 'Text Box ID 29: 15.686', 'Text Box ID 30: 1,829,9 MB', 'Text Box ID 31: MBI:', 'Text Box ID 32: Microscft Tean;', 'Text Box ID 33: 142.9MB', 'Text Box ID 34: MBY:', 'Text Box ID 35: App history', 'Text Box ID 36: Microsoft Azure Storage Explo_', 'Text Box ID 37: Efficiency_', 'Text Box ID 38: 0.83', 'Text Box ID 39: 245,0 MB', 'Text Box ID 40: MB/s', 'Text Box ID 41: Startup apps', 'Text Box ID 42: WebViewz Manager', 'Text Box ID 43: 75,9MB', 'Text Box ID 44: MBY:', 'Text Box ID 45: Users', 'Text Box ID 46: Service Host: Storage Service', 'Text Box ID 47: 096', 'Text Box ID 48: 1,1 MB', 'Text Box ID 49: MBY:', 'Text Box ID 50: Details', 'Text Box ID 51: Services', 'Text Box ID 52: Google', 'Text Box ID 53: Google Search', 'Text Box ID 54: Feeling Lucky', 'Text Box ID 55: Discover', 'Text Box ID 56: the ways Chrome keeps you safe while you browse', 'Text Box ID 57: Our third decade of climate action: join us', 'Text Box ID 58: Settings', 'Text Box ID 59: Advertising', 'Text Box ID 60: Business', 'Text Box ID 61: How Search works', 'Text Box ID 62: Privacy', 'Text Box ID 63: Terms', 'Text Box ID 64: Settings', 'Text Box ID 65: 3.53 PM', 'Text Box ID 66: Search', 'Text Box ID 67: Microsoft', 'Text Box ID 68: 10/25/2024', 'Text Box ID 69: Edge', 'Icon Box ID 70: Microsoft Edge browser.', 'Icon Box ID 71: Microsoft 365.', 'Icon Box ID 72: Settings', 'Icon Box ID 73: Image', 'Icon Box ID 74: Image', 'Icon Box ID 75: Microsoft Edge browser.', 'Icon Box ID 76: Microsoft Edge browser.', 'Icon Box ID 77: Teams.', 'Icon Box ID 78: Uncomm&ent Selection', 'Icon Box ID 79: Microsoft OneNote.', 'Icon Box ID 80: Find', 'Icon Box ID 81: Microsoft Outlook.', 'Icon Box ID 82: Image', 'Icon Box ID 83: Maximize', 'Icon Box ID 84: Close', 'Icon Box ID 85: Dictate', 'Icon Box ID 86: Line Spacing', 'Icon Box ID 87: creating a new document or file.', 'Icon Box ID 88: Five-point star', 'Icon Box ID 89: a search function.', 'Icon Box ID 90: Increase', 'Icon Box ID 91: More options', 'Icon Box ID 92: the Windows operating system.', 'Icon Box ID 93: Hyperlink', 'Icon Box ID 94: App launcher or menu.', 'Icon Box ID 95: Health monitoring', 'Icon Box ID 96: Microsoft Outlook.', 'Icon Box ID 97: minimizing a window.', 'Icon Box ID 98: uBlock Origin (Ctrl+T)', 'Icon Box ID 99: Undo', 'Icon Box ID 100: Pentagon', 'Icon Box ID 101: Settings', 'Icon Box ID 102: 1.0%', 'Icon Box ID 103: Back', 'Icon Box ID 104: Rectangle', 'Icon Box ID 105: minimizing a window.', 'Icon Box ID 106: Redo', 'Icon Box ID 107: opening a folder.', 'Icon Box ID 108: Justified', 'Icon Box ID 109: a loading or progress bar.', 'Icon Box ID 110: Label', 'Icon Box ID 111: Google.', 'Icon Box ID 112: Maximize window', 'Icon Box ID 113: Close', 'Icon Box ID 114: Close', 'Icon Box ID 115: Google Chrome web browser.']
Action: {"action":"click", "id":0}

Task: search 'halloween' in google
Screen:
['Text Box ID 0: Task Manager', 'Text Box ID 1: storage', 'Text Box ID 2: InPrivate', 'Text Box ID 3: Google', 'Text Box ID 4: https:/ WWW.googlecom', 'Text Box ID 5: Processes', 'Text Box ID 6: Run new task', 'Text Box ID 7: End task', 'Text Box ID 8: New folder', 'Text Box ID 9: Google', 'Text Box ID 10: finetune/ldm-ft__', 'Text Box ID 11: Gradio', 'Text Box ID 12: Pipelines', 'Text Box ID 13: Recent', 'Text Box ID 14: haotian-liu/LLaVA:', 'Text Box ID 15: Processes', 'Text Box ID 16: 67%', 'Text Box ID 17: 5496', 'Text Box ID 18: Status', 'Text Box ID 19: CPU', 'Text Box ID 20: Memory', 'Text Box ID 21: Disk', 'Text Box ID 22: About', 'Text Box ID 23: Store', 'Text Box ID 24: Gmail', 'Text Box ID 25: Images', 'Text Box ID 26: Sign in', 'Text Box ID 27: Performance', 'Text Box ID 28: Microscft', 'Text Box ID 29: 15.686', 'Text Box ID 30: 1,829,9 MB', 'Text Box ID 31: MBI:', 'Text Box ID 32: Microscft Tean;', 'Text Box ID 33: 142.9MB', 'Text Box ID 34: MBY:', 'Text Box ID 35: App history', 'Text Box ID 36: Microsoft Azure Storage Explo_', 'Text Box ID 37: Efficiency_', 'Text Box ID 38: 0.83', 'Text Box ID 39: 245,0 MB', 'Text Box ID 40: MB/s', 'Text Box ID 41: Startup apps', 'Text Box ID 42: WebViewz Manager', 'Text Box ID 43: 75,9MB', 'Text Box ID 44: MBY:', 'Text Box ID 45: Users', 'Text Box ID 46: Service Host: Storage Service', 'Text Box ID 47: 096', 'Text Box ID 48: 1,1 MB', 'Text Box ID 49: MBY:', 'Text Box ID 50: Details', 'Text Box ID 51: Services', 'Text Box ID 52: Google', 'Text Box ID 53: Google Search', 'Text Box ID 54: Feeling Lucky', 'Text Box ID 55: Discover', 'Text Box ID 56: the ways Chrome keeps you safe while you browse', 'Text Box ID 57: Our third decade of climate action: join us', 'Text Box ID 58: Settings', 'Text Box ID 59: Advertising', 'Text Box ID 60: Business', 'Text Box ID 61: How Search works', 'Text Box ID 62: Privacy', 'Text Box ID 63: Terms', 'Text Box ID 64: Settings', 'Text Box ID 65: 3.53 PM', 'Text Box ID 66: Search', 'Text Box ID 67: Microsoft', 'Text Box ID 68: 10/25/2024', 'Text Box ID 69: Edge', 'Icon Box ID 70: Microsoft Edge browser.', 'Icon Box ID 71: Microsoft 365.', 'Icon Box ID 72: Settings', 'Icon Box ID 73: Image', 'Icon Box ID 74: Image', 'Icon Box ID 75: Microsoft Edge browser.', 'Icon Box ID 76: Microsoft Edge browser.', 'Icon Box ID 77: Teams.', 'Icon Box ID 78: Uncomm&ent Selection', 'Icon Box ID 79: Microsoft OneNote.', 'Icon Box ID 80: Find', 'Icon Box ID 81: Microsoft Outlook.', 'Icon Box ID 82: Image', 'Icon Box ID 83: Maximize', 'Icon Box ID 84: Close', 'Icon Box ID 85: Dictate', 'Icon Box ID 86: Line Spacing', 'Icon Box ID 87: creating a new document or file.', 'Icon Box ID 88: Five-point star', 'Icon Box ID 89: a search function.', 'Icon Box ID 90: Increase', 'Icon Box ID 91: More options', 'Icon Box ID 92: the Windows operating system.', 'Icon Box ID 93: Hyperlink', 'Icon Box ID 94: App launcher or menu.', 'Icon Box ID 95: Health monitoring', 'Icon Box ID 96: Microsoft Outlook.', 'Icon Box ID 97: minimizing a window.', 'Icon Box ID 98: uBlock Origin (Ctrl+T)', 'Icon Box ID 99: Undo', 'Icon Box ID 100: Pentagon', 'Icon Box ID 101: Settings', 'Icon Box ID 102: 1.0%', 'Icon Box ID 103: Back', 'Icon Box ID 104: Rectangle', 'Icon Box ID 105: minimizing a window.', 'Icon Box ID 106: Redo', 'Icon Box ID 107: opening a folder.', 'Icon Box ID 108: Justified', 'Icon Box ID 109: a loading or progress bar.', 'Icon Box ID 110: Label', 'Icon Box ID 111: Google.', 'Icon Box ID 112: Maximize window', 'Icon Box ID 113: Close', 'Icon Box ID 114: Close', 'Icon Box ID 115: Google Chrome web browser.']
Action: {"action":"fill", "data":"halloween", "id":3}

Task: open mozilla firefox
Screen: """+ ",".join(parsed_content_list)
    print(parsed_content_list)
    prompt += "\nAction: "
    res = llamacpp_generate("http://localhost:8080/completion", prompt)
    print(res)
    try:
       resjson = extract_json_from_text(res)
       print("SUCCESS RESJON")
    except:
       resjson = {}
       resjson["id"] = res.split("ID ")[-1].split(":")[0]
       resjson["action"] = "click"
    print("TRYING CLICK", resjson)

    if resjson["action"] == "click":
        print("COORDS",label_coordinates, type(label_coordinates))
        coord = label_coordinates[str(resjson["id"])]
        print("COORD:", coord)
        print("CLICKING")
        left_click(coord[0],coord[1])
