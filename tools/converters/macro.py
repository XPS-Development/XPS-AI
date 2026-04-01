import json
import re
import subprocess
import time
import tkinter as tk
from pathlib import Path

import pyautogui
import uiautomation as auto

APP_PATH = Path("/Program Files (x86)/XPSPEAK/XPSPEAK.exe")
SOURCE_DIR = Path("/Users/User/Desktop/xpsdir")
OUTPUT_DIR = Path("/Users/User/Desktop/exported_spec")

DELAY = 0.05
APP_WINDOW_NAME = "XPS Peak Processing"
OPEN_BUTTON_NAME = "Open XPS"
DATA_BUTTON_NAMES = ["Data"]
BACKGROUND_BUTTON_NAMES = ["Background"]
CLOSE_BUTTON_NAMES = ["Close"]
KNOWN_BG = ("shirley", "linear", "tougaard", "none")


def all_controls(control, depth=0) -> list:
    controls = []
    for child in control.GetChildren():
        controls.append(child)
        controls.extend(all_controls(child, depth + 1))
    return controls


def click_button(window, name, alt_names) -> bool:
    controls = all_controls(window)

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


def controls_by_type(window, type_name) -> list:
    controls = all_controls(window)
    return [control for control in controls if control.ControlTypeName == type_name]


def ctrl_rect(control) -> tuple:
    rect = control.BoundingRectangle
    return rect.left, rect.top, rect.right, rect.bottom


def ctrl_text(control) -> str:
    name = str(control.Name).strip()
    if name:
        return name
    return ""


def set_clip(text) -> None:
    root = tk.Tk()
    root.withdraw()
    root.clipboard_clear()
    root.clipboard_append(text)
    root.update()
    root.destroy()


def get_clip() -> str:
    root = tk.Tk()
    root.withdraw()
    text = root.clipboard_get()
    root.destroy()
    return str(text).strip()


def copy_dropdown(button) -> str:
    set_clip("")
    button.Click()
    time.sleep(DELAY)
    pyautogui.hotkey("ctrl", "a")
    time.sleep(DELAY)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(DELAY)
    text = get_clip()
    pyautogui.press("escape")
    time.sleep(DELAY)
    return text


def parse_ave(text) -> int | None:
    match = re.search(r"\d+", text)
    if match:
        return int(match.group(0))
    return None


def parse_number(text) -> int | float | None:
    cleaned = str(text).strip().replace(" ", "").replace(",", ".")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None
    num_text = match.group(0)
    if "." in num_text:
        return float(num_text)
    return int(num_text)


def read_ave(background_window, dropdown_button) -> int | None:
    left, top, right, bottom = ctrl_rect(dropdown_button)
    center_y = (top + bottom) / 2
    controls = all_controls(background_window)

    best_text = ""
    best_score = None
    for control in controls:
        c_left, c_top, c_right, c_bottom = ctrl_rect(control)
        c_center_y = (c_top + c_bottom) / 2
        if abs(c_center_y - center_y) > 12:
            continue
        if c_right > right:
            continue
        text = ctrl_text(control)
        if not re.search(r"\d+", text):
            continue
        score = abs(c_center_y - center_y) * 10 + abs(c_right - left)
        if best_score is None or score < best_score:
            best_score = score
            best_text = text

    if best_text:
        return parse_ave(best_text)
    return parse_ave(copy_dropdown(dropdown_button))


def read_bg_type(background_window, dropdown_button) -> str:
    text = copy_dropdown(dropdown_button)
    if text and text.lower() != "open":
        return text

    left, top, right, bottom = ctrl_rect(dropdown_button)
    center_y = (top + bottom) / 2
    controls = all_controls(background_window)

    best_text = ""
    best_score = None
    for control in controls:
        c_left, c_top, c_right, c_bottom = ctrl_rect(control)
        c_center_y = (c_top + c_bottom) / 2
        if abs(c_center_y - center_y) > 12:
            continue
        if c_right > right:
            continue
        candidate = ctrl_text(control).strip()
        if not candidate:
            continue
        lower = candidate.lower()
        if not any(name in lower for name in KNOWN_BG):
            continue
        score = abs(c_center_y - center_y) * 10 + abs(c_right - left)
        if best_score is None or score < best_score:
            best_score = score
            best_text = candidate

    return best_text


def copy_num_at(x, y) -> int | float | None:
    set_clip("")
    pyautogui.doubleClick(int(x), int(y))
    time.sleep(DELAY)
    pyautogui.hotkey("ctrl", "c")
    time.sleep(DELAY)
    return parse_number(get_clip())


def read_left_value(button, max_steps=24, step_px=5) -> int | float | None:
    left, top, right, bottom = ctrl_rect(button)
    center_y = (top + bottom) / 2
    start_x = left - 8
    for i in range(max_steps):
        x = start_x - i * step_px
        value = copy_num_at(x, center_y)
        if value is not None:
            return value
    return None


def read_opt_values(background_window) -> tuple:
    optimise_buttons = []
    for control in all_controls(background_window):
        if control.ControlTypeName != "ButtonControl":
            continue
        name = str(control.Name).lower()
        if "optimise" in name or "optimize" in name:
            optimise_buttons.append(control)

    if len(optimise_buttons) < 2:
        return None, None

    optimise_buttons = sorted(optimise_buttons, key=lambda control: ctrl_rect(control)[1])
    slope = read_left_value(optimise_buttons[0])
    b1 = read_left_value(optimise_buttons[1])
    return slope, b1


def find_bg_window(num):
    candidates = [f"Region {num - 1}", f"Region {num}"] if num > 0 else [f"Region {num}"]
    windows = auto.GetRootControl().GetChildren()

    for name in candidates:
        for window in windows:
            if window.Name != name:
                continue
            dropdowns = [control for control in all_controls(window) if control.AutomationId == "DropDown"]
            if len(dropdowns) >= 2:
                return window
    return None


def save_bg_file(name, num, background_data) -> None:
    base_name = Path(name).stem
    path = OUTPUT_DIR / f"{base_name}_{num}.bg.json"
    with open(path, "w", encoding="utf-8") as file:
        json.dump(background_data, file, ensure_ascii=False, indent=2)


def save_bg_meta(num, name) -> bool:
    region_window = auto.WindowControl(searchDepth=1, Name=f"Region {num}")
    if not click_button(region_window, "Background", BACKGROUND_BUTTON_NAMES):
        return False

    time.sleep(DELAY * 3)
    background_window = find_bg_window(num)
    if background_window is None:
        return False

    dropdowns = [control for control in all_controls(background_window) if control.AutomationId == "DropDown"]
    if len(dropdowns) < 2:
        return False
    dropdowns = sorted(dropdowns, key=lambda control: ctrl_rect(control)[0])

    no_of_ave_value = read_ave(background_window, dropdowns[0])
    background_type = read_bg_type(background_window, dropdowns[1])
    slope, b1 = read_opt_values(background_window)

    background_type_lower = background_type.lower()
    background_data = {
        "no. of ave. pts at end-points": no_of_ave_value,
        "background type": background_type_lower,
    }
    if "shirley" in background_type_lower:
        background_data["slope"] = slope
    if "tougaard" in background_type_lower:
        background_data["b1"] = b1
    save_bg_file(name, num, background_data)

    click_button(background_window, "Close", CLOSE_BUTTON_NAMES)
    return True


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
    pyautogui.press('enter')
    time.sleep(DELAY)

    pyautogui.press('enter')
    pyautogui.press('enter')
    time.sleep(DELAY * 10)


def close_popups() -> None:
    for window in auto.GetRootControl().GetChildren():
        if window.Name != APP_WINDOW_NAME:
            pyautogui.press('escape')
            time.sleep(DELAY)
            break


def open_xps(name) -> bool:
    close_popups()

    main_window = auto.WindowControl(searchDepth=1, Name=APP_WINDOW_NAME)
    open_button = main_window.ButtonControl(Name=OPEN_BUTTON_NAME)
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
    pyautogui.press('enter')
    time.sleep(DELAY)
    return True


def has_peak_window(num) -> bool:
    time.sleep(DELAY * 3)
    all_windows = auto.GetRootControl().GetChildren()

    for window in all_windows:
        window_name = window.Name
        if "Peak" in window_name and any(char.isdigit() for char in window_name):
            if APP_WINDOW_NAME not in window_name:
                return True

    return False


def find_peak_offset(main_window) -> int:
    files = [file.name for file in SOURCE_DIR.iterdir() if file.is_file() and file.suffix.lower() == '.xps']
    calib_file = files[0]
    open_xps(calib_file)

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

            if has_peak_window(num):
                close_popups()
                return offset

    close_popups()
    return


def click_peak(main_window, num, peak_offset) -> bool:
    button = main_window.ButtonControl(Name=str(num))
    rect = button.BoundingRectangle

    y = rect.top + (rect.height() // 2)
    x = rect.right + peak_offset

    pyautogui.click(x, y)
    time.sleep(DELAY * 3)
    return has_peak_window(num)


def is_region_active(main_window, num, peak_offset) -> bool:
    button = main_window.ButtonControl(Name=str(num))
    button.Click()
    time.sleep(DELAY)

    if click_peak(main_window, num, peak_offset):
        return True
    else:
        return False


def get_active_regions(main_window, name, peak_offset) -> list:
    active = []

    for num in range(1, 11):
        if is_region_active(main_window, num, peak_offset):
            active.append(num)

        time.sleep(DELAY)

    print(f"Active regions {name}: {active}")
    return active


def export_region(num, name) -> bool:
    region_window = auto.WindowControl(searchDepth=1, Name=f"Region {num}")

    if not click_button(region_window, "Data", DATA_BUTTON_NAMES):
        return False

    if not click_button(region_window, "Export (peak parameters)", ["peak", "parameters"]):
        return False

    set_target()
    save_file(name, num)

    region_window = auto.WindowControl(searchDepth=1, Name=f"Region {num}")

    if not click_button(region_window, "Data", DATA_BUTTON_NAMES):
        return False

    if not click_button(region_window, "Export (spectrum)", ["spectrum"]):
        return False

    set_target()
    save_file(name, num)
    save_bg_meta(num, name)
    return True


def main():
    subprocess.Popen(str(APP_PATH))
    time.sleep(DELAY * 30)

    main_window = auto.WindowControl(searchDepth=1, Name=APP_WINDOW_NAME)
    peak_offset = find_peak_offset(main_window)
    files = [file.name for file in SOURCE_DIR.iterdir() if file.is_file() and file.suffix.lower() == '.xps']

    for file in files:
        open_xps(file)
        active = get_active_regions(main_window, file, peak_offset)

        for num in active:
            button = main_window.ButtonControl(Name=str(num))
            button.Click()
            time.sleep(DELAY)
            export_region(num, file)
            time.sleep(DELAY)

    print("Done")


if __name__ == "__main__":
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    main()
