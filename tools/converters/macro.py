import subprocess
import time
import uiautomation as auto
import pyautogui
from pathlib import Path

APP_PATH = Path("/Program Files (x86)/XPSPEAK/XPSPEAK.exe")
SOURCE_DIR = Path("/Users/User/Desktop/xpsdir")
OUTPUT_DIR = Path("/Users/User/Desktop/exported_spec")

DELAY = 0.1

def get_controls(control, depth=0) -> list:
    controls = []
    for child in control.GetChildren():
        controls.append(child)
        controls.extend(get_controls(child, depth + 1))
    return controls

def click_button(window, name, alt_names) -> bool:
    controls = get_controls(window)
    
    for control in controls:
        control_name = control.Name.lower()
            
        if name.lower() in control_name:
            control.Click()
            time.sleep(DELAY)
            return True
            
        for alt in alt_names:
            if alt.lower() in control_name:
                control.Click()
                time.sleep(DELAY)
                return True
    
    return False

def set_source() -> None:
    pyautogui.write(str(SOURCE_DIR))
    time.sleep(DELAY)
    pyautogui.press('enter')
    time.sleep(DELAY)

def set_target() -> None:
    pyautogui.write(str(OUTPUT_DIR))
    time.sleep(DELAY)
    pyautogui.press('enter')
    time.sleep(DELAY)

def save_file(name, num) -> None:
    base_name = Path(name).stem
    file_name = f"{base_name}_{num}"
    
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(DELAY)
    pyautogui.press('delete')
    time.sleep(DELAY)
    
    pyautogui.write(file_name)
    time.sleep(DELAY)
    pyautogui.press('enter')
    time.sleep(DELAY)
    
    pyautogui.press('enter')
    time.sleep(DELAY)

def close_dialogs():
    for window in auto.GetRootControl().GetChildren():
        if window.Name != "XPS Peak Processing" and window.Name != "Панель задач":
            pyautogui.press('escape')
            time.sleep(DELAY)
            break

def open_file(name) -> bool:
    close_dialogs()
    
    main_window = auto.WindowControl(searchDepth=1, Name="XPS Peak Processing")
    open_button = main_window.ButtonControl(Name="Open XPS")
    open_button.Click()
    time.sleep(DELAY)
    
    time.sleep(DELAY)
    
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(DELAY)
    pyautogui.press('delete')
    time.sleep(DELAY)
    
    set_source()
    
    pyautogui.write(name)
    time.sleep(DELAY)
    pyautogui.press('enter')
    time.sleep(DELAY * 10)
    
    return True

def check_peak(num) -> bool:
    time.sleep(DELAY * 3)
    
    all_windows = auto.GetRootControl().GetChildren()
 
    for window in all_windows:
        window_name = window.Name
        if "Peak" in window_name and any(c.isdigit() for c in window_name):
            if "XPS Peak Processing" not in window_name:
                return True
    
    return False

def optimal_offset(main_window) -> int:
    files = [f.name for f in SOURCE_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.xps']
    calib_file = files[0]
    open_file(calib_file)
    
    for offset in range(5, 51, 5):
        for num in range(1, 11):
            button = main_window.ButtonControl(Name=str(num))
            rect = button.BoundingRectangle
            y = rect.top + (rect.height() // 2)
            x = rect.right + offset
            
            button.Click()
            time.sleep(DELAY)
            
            pyautogui.click(x, y)
            time.sleep(DELAY * 3)
            
            if check_peak(num):
                close_dialogs()
                return offset
    
    close_dialogs()
    return

def click_peak_button(main_window, num, peak_offset) -> bool:
    button = main_window.ButtonControl(Name=str(num))
    rect = button.BoundingRectangle
    
    y = rect.top + (rect.height() // 2)
    x = rect.right + peak_offset
    
    pyautogui.click(x, y)
    time.sleep(DELAY * 3)
    
    return check_peak(num)

def is_active(main_window, num, peak_offset) -> bool:
    button = main_window.ButtonControl(Name=str(num))
    button.Click()
    time.sleep(DELAY)
    
    if click_peak_button(main_window, num, peak_offset):
        return True
    else:
        return False

def get_active(main_window, name, peak_offset) -> list:
    active = []
    
    for num in range(1, 11):
        if is_active(main_window, num, peak_offset):
            active.append(num)
        
        time.sleep(DELAY)
    
    print(f"Active regions {name}: {active}")
    return active

def process_region(num, name) -> bool:
    region_window = auto.WindowControl(searchDepth=1, Name=f"Region {num}")
    
    if not click_button(region_window, "Data", ["Data", "Данные"]):
        return False
    
    if not click_button(region_window, "Export (peak parameters)", ["peak", "parameters"]):
        return False
    
    set_target()
    save_file(name, num)
    
    region_window = auto.WindowControl(searchDepth=1, Name=f"Region {num}")
    
    if not click_button(region_window, "Data", ["Data", "Данные"]):
        return False
    
    if not click_button(region_window, "Export (spectrum)", ["spectrum"]):
        return False
    
    set_target()
    save_file(name, num)
    
    return True

def main() -> None:
    subprocess.Popen(str(APP_PATH))
    time.sleep(DELAY * 30)
    
    main_window = auto.WindowControl(searchDepth=1, Name="XPS Peak Processing")
    
    peak_offset = optimal_offset(main_window)
    
    files = [f.name for f in SOURCE_DIR.iterdir() if f.is_file() and f.suffix.lower() == '.xps']
    
    for file in files:
        open_file(file)
        active = get_active(main_window, file, peak_offset)
        
        for num in active:
            button = main_window.ButtonControl(Name=str(num))
            button.Click()
            time.sleep(DELAY)
            process_region(num, file)
            time.sleep(DELAY)
        
    print("Done")

if __name__ == "__main__":
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()