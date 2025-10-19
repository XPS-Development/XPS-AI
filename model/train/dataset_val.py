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

def check_peak_monotonicity(peak_intensities, trim_percentage=0.05, limit=1.25) -> bool:        
    intensities = np.array(peak_intensities)
    gradient = np.gradient(intensities)
    num_points = len(gradient)
    trim_points = int(trim_percentage * num_points)

    if num_points > 2 * trim_points:
        gradient[:trim_points] = 0
        gradient[-trim_points:] = 0
    
    return np.any(gradient > limit)

def negative_intensities(peak_intensities) -> bool:   
    return np.any(np.array(peak_intensities) < 0)

def create_mask(x, from_x, to_x) -> np.ndarray:  
    zeros = np.zeros_like(x)
    zeros[(x > from_x) & (x < to_x)] = 1
    return zeros

def plot_peak_intensity(peak_intensities, peak_num, spectrum_name, json_filename, show_gradient=True):
    plt.figure(figsize=(10, 6))
    
    if show_gradient:
        plt.subplot(2, 1, 1)
    
    plt.plot(peak_intensities)
    title_type = "gradient filtered" if show_gradient else "area filtered"
    plt.title(f'Peak {peak_num} in spectrum "{spectrum_name}" ({title_type})\n(file: {json_filename})')
    plt.grid(True, alpha=0.3)
    
    if show_gradient:
        plt.subplot(2, 1, 2)
        gradient = np.gradient(peak_intensities)
        plt.plot(gradient, 'r-')
        plt.axhline(y=5.0)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def process_spectrum(spectrum_data, json_filename, spectrum_name, config):
    start_x = spectrum_data['BE']['start']
    step_x = spectrum_data['BE']['step']
    x_num = spectrum_data['BE']['num_points']

    x = np.linspace(start_x, start_x + step_x * (x_num - 1), x_num)
    y = np.array(spectrum_data['raw_intensity'], dtype=np.float32)
    if np.any(y < 0):
        if config['print_data']:
            print(f"Skipping '{spectrum_name}' (file: {json_filename}) - negative intensities")
        return None, None, None, None
    
    y = (y - y.min()) / (y.max() - y.min())

    x, y = resize_to_256(x, y)

    y_log = np.log(10 * y + 1)
    y_log = (y_log - y_log.min()) / (y_log.max() - y_log.min())

    peak_mask = np.zeros_like(x)
    max_mask = np.zeros_like(x)

    peaks_dict = spectrum_data['peaks']
    total_area = sum(peak_info['area'] for peak_info in peaks_dict.values())
    valid_peaks_count = 0
    
    for peak_num, peak_info in peaks_dict.items():
        position = peak_info['position']
        fwhm = peak_info['fwhm']
        area = peak_info['area']
        peak_intensities = peak_info['intensity']
        
        relative_area = area / total_area 
        
        if relative_area < config['min_relative_area']:
            if config['print_data']:
                print(f"Peak {peak_num} in '{spectrum_name}' (file: {json_filename}) with relative area {relative_area:.4f} < {config['min_relative_area']}")
                plot_peak_intensity(peak_intensities, peak_num, spectrum_name, json_filename, show_gradient=False)
            continue

        if negative_intensities(peak_intensities):
            if config['print_data']:
                print(f"Peak {peak_num} in '{spectrum_name}' (file: {json_filename}) with negative intensity")
            continue

        if len(peak_intensities) > 0:
            if check_peak_monotonicity(peak_intensities, config['trim_percentage']):
                peak_mask += create_mask(x, position - config['width_peak'] * fwhm, position + config['width_peak'] * fwhm)
                max_mask += create_mask(x, position - config['width_max'], position + config['width_max'])
                valid_peaks_count += 1
            else:
                if config['print_data']:
                    print(f"Peak {peak_num} in '{spectrum_name}' (file: {json_filename}) filtered by gradient")
                    plot_peak_intensity(peak_intensities, peak_num, spectrum_name, json_filename, show_gradient=True)
    
    peak_mask[peak_mask > 0] = 1
    max_mask[max_mask > 0] = 1
    
    return y, y_log, peak_mask, max_mask

def generate_dataset(
        json_dir,
        output_dir,
        width_peak = 1.23,
        width_max = 0.2,
        min_relative_area = 0.002,
        trim_percentage = 0.05,
        **kwargs):
    
    config = {
        'width_peak': kwargs.get('width_peak', width_peak),
        'width_max': kwargs.get('width_max', width_max),
        'min_relative_area': kwargs.get('min_relative_area', min_relative_area),
        'trim_percentage': kwargs.get('trim_percentage', trim_percentage),
        'print_data': kwargs.get('print_data', False)
    }
    
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_files = list(json_dir.glob('*.json'))
    
    file_counter = 0
    total_peaks_count = 0
    spectra_with_peaks = 0
    total_spectra = 0
    filtered_negative_intensity = 0
    peaks_stats = []
    
    for json_file in json_files:
        json_filename = json_file.name
        if config['print_data']:
            print(f"{json_filename}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        spectra = list(data['spectra'].values())
        spectrum_names = list(data['spectra'].keys())
            
        for spectrum_data, spectrum_name in zip(spectra, spectrum_names):
            total_spectra += 1
            result = process_spectrum(spectrum_data, json_filename, spectrum_name, config)
            
            if result[0] is None:
                filtered_negative_intensity += 1
                continue
                
            y, y_log, peak_mask, max_mask = result
                
            data_array = np.stack((y, y_log, peak_mask, max_mask), axis=1)
            df = pd.DataFrame(data_array)
            output_file = output_dir / f'{file_counter}.csv'
            df.to_csv(output_file, header=False, index=False)
            
            peaks_dict = spectrum_data['peaks']
            valid_peaks_in_spectrum = 0
            total_area = sum(peak_info['area'] for peak_info in peaks_dict.values())
            
            for peak_num, peak_info in peaks_dict.items():
                area = peak_info['area']
                relative_area = area / total_area 
                peak_intensities = peak_info['intensity']
                
                is_valid = (
                    relative_area >= config['min_relative_area'] and
                    not negative_intensities(peak_intensities) and
                    check_peak_monotonicity(peak_intensities, config['trim_percentage'])
                )
                
                peak_stat = {
                    'file': json_filename,
                    'spectrum': spectrum_name,
                    'peak_number': peak_num,
                    'area': area,
                    'fwhm': peak_info['fwhm'],
                    'position': peak_info['position'],
                    'relative_area': relative_area,
                    'valid': is_valid
                }
                peaks_stats.append(peak_stat)
                
                if is_valid:
                    valid_peaks_in_spectrum += 1
            
            total_peaks_count += valid_peaks_in_spectrum
            if valid_peaks_in_spectrum > 0:
                spectra_with_peaks += 1
            file_counter += 1
    
    if config['print_data']:
        print(f"Total spectra processed: {total_spectra}")
        print(f"Filtered due to negative intensities: {filtered_negative_intensity}")
        print(f"Remaining spectra: {file_counter}")
        print(f"Total peaks in spectra: {total_peaks_count}")
        print(f"Average number of peaks: {total_peaks_count / spectra_with_peaks}")
    
    return {
        'total_spectra': total_spectra,
        'filtered_negative_intensity': filtered_negative_intensity,
        'file_counter': file_counter,
        'total_peaks_count': total_peaks_count,
        'spectra_with_peaks': spectra_with_peaks,
        'peaks_stats': peaks_stats
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
    json_dir = 'C:/Users/User/Desktop/Json_spec'
    output_dir = 'C:/Users/User/Desktop/RealXPSDataset'
    
    results = generate_dataset(
        json_dir, 
        output_dir,
        min_relative_area=0.002,
        print_data=False
    )
    view_labeled_data(output_dir, 0)
    
    return results

if __name__ == "__main__":
    results = main()