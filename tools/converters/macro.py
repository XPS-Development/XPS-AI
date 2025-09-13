"""
Пока есть зависимость от введённых координат кнопок open xps и 1 региона свёрнутого окна приложения; 

Если в диалоговом окне вместо пути до папки появляется //// - выключить макрос, раскомментировать 146, закомментировать 145 (исправлю в ближайшее время)

Некоторые функции уже не нужны - уберу

"""
import pyautogui
import time
import subprocess
import pyperclip
import shutil
from pathlib import Path

APPLICATION_PATH = "/Program Files (x86)/XPSPEAK/XPSPEAK" 
FOLDER_PATH = "/Users/User/Desktop/xpsdir"  
TARGET_PATH = "/Users/User/Desktop/exported_spec"
ADAPTED_FOLDER_PATH = '\\'+'Users'+'\\'+'User'+'\\'+'Desktop'+'\\'+'xpsdir'
COORDINATES_BUTTON1 = (74, 320) # координаты кнопки open xps
COORDINATES_BUTTON2 = (COORDINATES_BUTTON1[0]-35, COORDINATES_BUTTON1[1]-305) # data
COORDINATES_BUTTON3 = (COORDINATES_BUTTON1[0]-2, COORDINATES_BUTTON1[1]-164) # export spec
COORDINATES_BUTTON4 = (COORDINATES_BUTTON1[0]+3, COORDINATES_BUTTON1[1]-139) # export param

REGION_BASE_X = 66 # координаты кнопки 1 в столбце регионов
REGION_BASE_Y = 397
REGION_Y_SPACING = 17

DELAY = 0.2

file_extensions = ['.par', '.dat']
NUM_REGIONS = 10

def check_and_create_folders():
    folder_path = Path(FOLDER_PATH)
    target_path = Path(TARGET_PATH)
    
    if not folder_path.exists():
        print(f"Creating folder: {FOLDER_PATH}")
        folder_path.mkdir(parents=True)
    
    if not target_path.exists():
        print(f"Creating folder: {TARGET_PATH}")
        target_path.mkdir(parents=True)

def type_text_safe(text, delay=0.01):
    for char in text:
        pyautogui.write(char)
        time.sleep(delay)

def check_and_fix(expected_text, max_attempts=2):
    for attempt in range(max_attempts):
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(DELAY)
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(DELAY)
        actual_text = pyperclip.paste()

        if actual_text == expected_text:
            return True
    
        pyautogui.press('backspace')
        time.sleep(DELAY)
        type_text_safe(expected_text)
    
    return False
    
def write_with_check(text):
    pyautogui.write(text)
    time.sleep(DELAY)

    if not check_and_fix(text):
        print('Failed to verify text input')

    time.sleep(DELAY)

def save_file_with_region_number(region_number, file_type, original_filename):
    time.sleep(DELAY)
    
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(DELAY)
    
    name_without_ext = Path(original_filename).stem
    if file_type == 'par':
        new_filename = f"{name_without_ext}_{region_number}.par"
    else:
        new_filename = f"{name_without_ext}_{region_number}.dat"
    
    type_text_safe(new_filename)
    time.sleep(DELAY)
    
    pyautogui.press('enter')
    time.sleep(DELAY)

def get_region_coordinates(region_number):
    x = REGION_BASE_X
    y = REGION_BASE_Y + (region_number - 1) * REGION_Y_SPACING
    return (x, y)

def check_region_validity(region_coords):

    """Check if region contains empty data by copying text from specific coordinates"""

    pyperclip.copy('')
    time.sleep(DELAY)
    
    check_x = region_coords[0] + 309
    check_y = region_coords[1]
    check_coords = (check_x, check_y)
    
    pyautogui.click(check_coords)
    time.sleep(0.05)
    pyautogui.click(check_coords)
    time.sleep(DELAY)
    
    pyautogui.hotkey('ctrl', 'c')
    time.sleep(DELAY)
    
    copied_text = pyperclip.paste().strip()
    
    print(f"Copied text: '{copied_text}'")
    
    if copied_text:
        print(f"Region valid: found text '{copied_text}'")
        return True
    else:
        print(f"Region invalid: no text found")
        return False

def process_region(files, region_number, skip_files):
    time.sleep(DELAY)
    region_coords = get_region_coordinates(region_number)
    
    print(f"Processing region {region_number} at coordinates {region_coords}")
    
    for filename in files:
        if filename in skip_files:
            print(f"Skipping file {filename} (marked as invalid)")
            continue
            
        print(f"Processing file {filename} for region {region_number}")
        
        pyautogui.click(COORDINATES_BUTTON1)
        time.sleep(DELAY)
        
        pyautogui.write(ADAPTED_FOLDER_PATH)
        time.sleep(DELAY)
        #type_text_safe(ADAPTED_FOLDER_PATH)
        pyautogui.press('enter')
        time.sleep(DELAY)
        
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(DELAY)
        pyautogui.press('backspace')
        time.sleep(DELAY)
        pyautogui.write(filename)
        time.sleep(DELAY)
        pyautogui.press('enter')
        time.sleep(DELAY)

        pyautogui.click(region_coords)
        
        if not check_region_validity(region_coords):
            print(f"Region {region_number} invalid for file {filename}, skipping this file for all regions")
            skip_files.add(filename)
            continue
        
        pyautogui.click(COORDINATES_BUTTON2)
        time.sleep(DELAY)
        pyautogui.click(COORDINATES_BUTTON3)
        time.sleep(DELAY)
        
        save_file_with_region_number(region_number, 'dat', filename)
        pyautogui.press('enter')
        time.sleep(DELAY)
        
        pyautogui.click(COORDINATES_BUTTON2)
        time.sleep(DELAY)
        pyautogui.click(COORDINATES_BUTTON4)
        time.sleep(DELAY)
        
        save_file_with_region_number(region_number, 'par', filename)
        time.sleep(DELAY)
        pyautogui.press('enter')
        time.sleep(DELAY)
        
        print(f"File {filename} processed for region {region_number}")
    
    return skip_files

def process_files():
    subprocess.Popen(APPLICATION_PATH)
    time.sleep(DELAY)
    
    folder_path = Path(FOLDER_PATH)
    files = [f.name for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() == '.xps']
    
    skip_files = set()
    
    for region_number in range(1, NUM_REGIONS + 1):
        skip_files = process_region(files, region_number, skip_files)
    
def move_files(source_folder, target_folder, extensions):
    source_path = Path(source_folder)
    target_path = Path(target_folder)
    
    target_path.mkdir(parents=True, exist_ok=True)

    if not source_path.exists():
        print(f'Error: folder {source_folder} does not exist!')
        return
    
    moved_count = 0
    for file_path in source_path.iterdir():
        if file_path.is_file():
            if any(file_path.suffix.lower() == ext for ext in extensions):
                try:
                    shutil.move(str(file_path), str(target_path / file_path.name))
                    moved_count += 1
                except Exception as e:
                    print(f'Error moving {file_path.name}: {e}')

    print(f'Moved {moved_count} files')

if __name__ == "__main__":
    check_and_create_folders()
    
    print("Macro will start in 3 seconds.")
    time.sleep(3)
    
    process_files()
    print("All files processed!")

    move_files(FOLDER_PATH, TARGET_PATH, file_extensions)
    print('Done!')