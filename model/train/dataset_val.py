import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def interpolate(x, y, num=256):
    f = interp1d(x, y, kind='linear', fill_value='extrapolate')
    new_x = np.linspace(x[0], x[-1], num, dtype=np.float32)
    new_y = f(new_x)
    return new_x, new_y

def resize_to_256(x, y):
    if len(y) != 256:
        x, y = interpolate(x, y, 256)
    return x, y

def fix_negative_intensities(y):
    y_fixed = y.copy()
    negative_indices = np.where(y < 0)[0]
    
    max_negative_streak = 0
    current_streak = 0
    for val in y:
        if val < 0:
            current_streak += 1
            max_negative_streak = max(max_negative_streak, current_streak)
        else:
            current_streak = 0
    
    if max_negative_streak >= 10:
        return None
    
    for idx in negative_indices:
        left_idx = idx - 1
        while left_idx >= 0 and y[left_idx] < 0:
            left_idx -= 1
        
        right_idx = idx + 1
        while right_idx < len(y) and y[right_idx] < 0:
            right_idx += 1
        
        if left_idx >= 0 and right_idx < len(y):
            x_left, y_left = left_idx, y[left_idx]
            x_right, y_right = right_idx, y[right_idx]
            y_fixed[idx] = y_left + (y_right - y_left) * (idx - x_left) / (x_right - x_left)
        elif left_idx >= 0:
            y_fixed[idx] = y[left_idx]
        elif right_idx < len(y):
            y_fixed[idx] = y[right_idx]
    
    return y_fixed

def negative_intensities(peak_intensities) -> bool:   
    return np.any(np.array(peak_intensities) < 0)

def create_mask(x, from_x, to_x) -> np.ndarray:  
    zeros = np.zeros_like(x)
    zeros[(x > from_x) & (x < to_x)] = 1
    return zeros

def process_spectrum(spectrum_data, json_filename, spectrum_name, width_peak, width_max, print_data):
    start_x = spectrum_data['BE']['start']
    step_x = spectrum_data['BE']['step']
    x_num = spectrum_data['BE']['num_points']

    x = np.linspace(start_x, start_x + step_x * (x_num - 1), x_num)
    y = np.array(spectrum_data['raw_intensity'], dtype=np.float32)
    
    if np.any(y < 0):
        if print_data:
            negative_count = np.sum(y < 0)
            print(f"Fixing {negative_count} negative points in '{spectrum_name}' (file: {json_filename})")
        y_fixed = fix_negative_intensities(y)
        if y_fixed is None:
            if print_data:
                print(f"Skipping '{spectrum_name}' (file: {json_filename}) - too many negative points (>=5)")
            return None, None, None, None
        y = y_fixed
    
    y = (y - y.min()) / (y.max() - y.min())

    x, y = resize_to_256(x, y)

    y_log = np.log(10 * y + 1)
    y_log = (y_log - y_log.min()) / (y_log.max() - y_log.min())

    peak_mask = np.zeros_like(x)
    max_mask = np.zeros_like(x)

    peaks_dict = spectrum_data['peaks']
    valid_peaks_count = 0
    
    for peak_num, peak_info in peaks_dict.items():
        position = peak_info['position']
        fwhm = peak_info['fwhm']
        peak_intensities = peak_info['intensity']

        if negative_intensities(peak_intensities):
            if print_data:
                print(f"Peak {peak_num} in '{spectrum_name}' (file: {json_filename}) with negative intensity")
            continue

        if len(peak_intensities) > 0:
            peak_mask += create_mask(x, position - width_peak * fwhm, position + width_peak * fwhm)
            max_mask += create_mask(x, position - width_max, position + width_max)
            valid_peaks_count += 1
    
    peak_mask[peak_mask > 0] = 1
    max_mask[max_mask > 0] = 1
    
    return y, y_log, peak_mask, max_mask

def generate_dataset(
        json_dir,
        output_dir,
        width_peak=1.1,
        width_max=0.15,
        print_data=False):
    
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(json_dir.glob('*.json'))
    
    file_counter = 0
    total_peaks_count = 0
    spectra_with_peaks = 0
    total_spectra = 0
    fixed_negative_intensity_count = 0
    total_negative_points = 0
    skipped_too_many_negatives = 0
    
    for json_file in json_files:
        json_filename = json_file.name
        if print_data:
            print(f"{json_filename}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for spectrum_name, spectrum_data in data['spectra'].items():
            total_spectra += 1
            
            y_raw = np.array(spectrum_data['raw_intensity'], dtype=np.float32)
            negative_points = np.sum(y_raw < 0)
            if negative_points > 0:
                fixed_negative_intensity_count += 1
                total_negative_points += negative_points
            
            result = process_spectrum(spectrum_data, json_filename, spectrum_name, width_peak, width_max, print_data)
            
            if result[0] is None:
                skipped_too_many_negatives += 1
                continue
                
            y, y_log, peak_mask, max_mask = result
                
            data_array = np.stack((y, y_log, peak_mask, max_mask), axis=1)
            df = pd.DataFrame(data_array)
            output_file = output_dir / f'{file_counter}.csv'
            df.to_csv(output_file, header=False, index=False)
            
            peaks_dict = spectrum_data['peaks']
            valid_peaks_in_spectrum = 0
            
            for peak_num, peak_info in peaks_dict.items():
                peak_intensities = peak_info['intensity']
                
                if not negative_intensities(peak_intensities):
                    valid_peaks_in_spectrum += 1
            
            total_peaks_count += valid_peaks_in_spectrum
            if valid_peaks_in_spectrum > 0:
                spectra_with_peaks += 1
            file_counter += 1
    
    if print_data:
        print(f"Total spectra processed: {total_spectra}")
        print(f"Spectra with negative intensities: {fixed_negative_intensity_count}")
        print(f"Total negative points found: {total_negative_points}")
        print(f"Spectra skipped (too many negatives): {skipped_too_many_negatives}")
        print(f"Remaining spectra: {file_counter}")
        print(f"Total peaks in spectra: {total_peaks_count}")
        if spectra_with_peaks > 0:
            print(f"Average number of peaks: {total_peaks_count / spectra_with_peaks}")
        else:
            print(f"Average number of peaks: 0")
    
    return {
        'total_spectra': total_spectra,
        'fixed_negative_intensity_count': fixed_negative_intensity_count,
        'total_negative_points': total_negative_points,
        'skipped_too_many_negatives': skipped_too_many_negatives,
        'file_counter': file_counter,
        'total_peaks_count': total_peaks_count,
        'spectra_with_peaks': spectra_with_peaks
    }

def view_labeled_data(output_dir, file_index):
    file_path = Path(output_dir) / f'{file_index}.csv'
    
    array = np.loadtxt(file_path, delimiter=',')
    y = array[:, 0]
    y_log = array[:, 1]
    peak_mask = array[:, 2]
    max_mask = array[:, 3]
    
    x = np.arange(len(y))
    
    plt.plot(x, y, 'k-')
    
    if peak_mask.sum() > 0:
        plt.fill_between(x, y, y.min(), where=peak_mask > 0, 
                       alpha=0.3, color='red', label='Peak region')
    if max_mask.sum() > 0:
        plt.fill_between(x, y, y.min(), where=max_mask > 0, 
                       alpha=0.5, color='blue', label='Max region')

    plt.legend()
    plt.show()

def main():
    json_dir = '/Users/User/Desktop/Json_spec'
    output_dir = 'model/train/data/val_dataset'
    
    results = generate_dataset(
        json_dir, 
        output_dir,
        print_data=True
    )
    view_labeled_data(output_dir, 0)
    
    return results

if __name__ == "__main__":
    results = main()