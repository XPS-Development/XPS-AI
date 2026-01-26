import re
import pandas as pd
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def parse_par_file(par_file_path):
    peaks_data = {}
    
    with open(par_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()

    table_pattern = r'Peak.*?Position.*?Area.*?FWHM.*?%GL.*?\n(.*?)(?:\n\n|\Z)'
    match = re.search(table_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if match:
        table_data = match.group(1).strip()
        lines = table_data.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or '---' in line:
                continue
        
            clean_line = re.sub(r'[|\+]', '', line).strip()
            parts = re.split(r'\s+', clean_line)
            
            if len(parts) >= 5:
                peak_number = int(parts[0])
                position = float(parts[1].replace(',', '.'))
                area = float(parts[2].replace(',', '.'))
                fwhm = float(parts[3].replace(',', '.'))
                gl = float(parts[4].replace(',', '.'))
                    
                peaks_data[peak_number] = {
                        'position': position,
                        'area': area,
                        'fwhm': fwhm,
                        'gl': gl
                    }
    else:
        print("Peak table not found in .par file")
    
    return peaks_data

def parse_dat_file(dat_file_path):
    data_lines = []
    with open(dat_file_path, 'r', encoding='utf-8', errors='ignore') as file:
        lines = file.readlines()
    
    data_started = False
    
    for line in lines:
        line = line.strip()
        
        if not line or line.startswith('#'):
            continue
            
        if 'B.E.' in line and 'Raw Intensity' in line:
            data_started = True
            continue
            
        if data_started:
            clean_line = line.replace(',', '.')
            numbers = re.findall(r'[-]?\d+\.?\d*', clean_line)
            
            if numbers:
                numeric_values = [float(num) for num in numbers]
                data_lines.append(numeric_values)
    
    if data_lines:
        max_cols = max(len(row) for row in data_lines)
            
        aligned_data = []
        for row in data_lines:
            if len(row) < max_cols:
                 row.extend([0.0] * (max_cols - len(row)))
            aligned_data.append(row)
            
        df = pd.DataFrame(aligned_data)
        return df
    else:
        return None

def process_file_pair(folder_path, base_filename):
    par_file = folder_path / f"{base_filename}.par"
    dat_file = folder_path / f"{base_filename}.dat"
    
    if not par_file.exists() or not dat_file.exists():
        return None

    peaks_data = parse_par_file(par_file)
    if not peaks_data:
        return None

    num_peaks = len(peaks_data)

    dat_df = parse_dat_file(dat_file)
    if dat_df is None or dat_df.empty:
        return None
    
    be_values = dat_df.iloc[:, 0].tolist()
    if len(be_values) >= 2:
        start_be = be_values[0]
        step_be = round(be_values[1] - be_values[0], 3)
        num_points = len(be_values)
    else:
        start_be = be_values[0] if be_values else 0
        step_be = 0
        num_points = len(be_values)
    
    raw_intensity = dat_df.iloc[:, 1].tolist()
    
    peak_intensities = {}
    
    sorted_peak_nums = sorted(peaks_data.keys())
    
    for i, peak_num in enumerate(sorted_peak_nums):
        col_idx = 4 + i  
        if col_idx < len(dat_df.columns):
            intensity_data = dat_df.iloc[:, col_idx].tolist()
            peak_intensities[peak_num] = intensity_data
        else:
            peak_intensities[peak_num] = [0.0] * num_points
    
    result_dict = {
        'BE': {
            'start': start_be,
            'step': step_be,
            'num_points': num_points
        },
        'raw_intensity': raw_intensity,
        'peaks': {}
    }
    
    for peak_num in sorted_peak_nums:
        peak_data = peaks_data[peak_num]
    
        if peak_num in peak_intensities:
            peak_data['intensity'] = peak_intensities[peak_num]
        
        result_dict['peaks'][f'peak_{peak_num}'] = peak_data
    
    return result_dict

def process_folder(folder_path, output_folder=None):
    all_results = {}
    
    par_files = [f for f in folder_path.iterdir() if f.suffix == '.par']
    
    if output_folder:
        output_folder.mkdir(exist_ok=True)
    
    for par_file in par_files:
        base_filename = par_file.stem
        
        result = process_file_pair(folder_path, base_filename)
        if result:
            all_results[base_filename] = result
            
            json_filename = f"{base_filename}_results.json"
            if output_folder:
                json_path = output_folder / json_filename
            else:
                json_path = Path(json_filename)
                
            save_single_result_to_json(result, json_path, base_filename)
    
    return all_results

def convert_for_json(obj):
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_for_json(value) for key, value in obj.items()}
    return obj

def save_single_result_to_json(result, output_path, filename):
    json_ready_data = convert_for_json(result)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_ready_data, f, indent=2, ensure_ascii=False)

def merge_json_files(json_dir, output_dir=None):
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)
    else:
        output_dir = json_dir
    
    file_groups = defaultdict(list)
    
    for json_file in json_dir.glob('*_results.json'):
        base_name = re.sub(r'_\d+_results\.json$', '', json_file.name)
        base_name = re.sub(r'_results\.json$', '', base_name)
        base_name = re.sub(r'_\d+$', '', base_name)
        
        file_groups[base_name].append(json_file)
    
    merged_files = []
    
    for base_name, files in file_groups.items():
        merged_data = {}
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            file_key = file_path.stem
            merged_data[file_key] = {
                'BE': file_data.get('BE', {}),
                'raw_intensity': file_data.get('raw_intensity', []),
                'peaks': file_data.get('peaks', {})
            }
        
        result = {
            'spectra': merged_data,
            'total_files': len(files),
            'total_peaks': sum(len(data['peaks']) for data in merged_data.values())
        }
        
        output_file = output_dir / f"{base_name}_merged_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        merged_files.append(output_file)
    
    delete_non_merged_files(json_dir, merged_files)
    
    return merged_files

def delete_non_merged_files(json_dir, merged_files):
    files_to_keep = {file.name for file in merged_files}
    
    for json_file in json_dir.glob('*_results.json'):
        if json_file.name not in files_to_keep:
            json_file.unlink()

if __name__ == "__main__":
    folder_path = Path(r"C:\Users\User\Desktop\exported_spec")
    output_folder = Path(r"C:\Users\User\Desktop\Json_spec")
        
    if folder_path.exists():
        results = process_folder(folder_path, output_folder=output_folder)
        
        if results:
            merged_files = merge_json_files(output_folder, output_folder)
            print("Done")
        else:
            print("No results found")
    else:
        print("Input folder does not exist")