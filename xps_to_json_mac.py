import os
import re
import pandas as pd
import json
import numpy as np

def parse_par_file(par_file_path):
    """
    Parses .par file and returns dictionary with peak data
    """
    peaks_data = {}
    
    try:
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
                    try:
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
                        print(f"Found peak {peak_number}: position={position}")
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error processing line: {line} - {e}")
                        continue
        else:
            print("Peak table not found in .par file")
            print("File content:")
            print(content[:500] + "..." if len(content) > 500 else content)
    
    except Exception as e:
        print(f"Error reading .par file {par_file_path}: {e}")
    
    return peaks_data

def parse_dat_file_simple(dat_file_path):
    """
    Simple .dat file parser - reads all numerical data
    """
    try:
        print(f"Reading .dat file: {dat_file_path}")
        
        data_lines = []
        with open(dat_file_path, 'r', encoding='utf-8', errors='ignore') as file:
            for line in file:
                line = line.strip()
                
                if (line.startswith('#') or 
                    any(x in line for x in ['File', 'B.E.', 'Raw Intensity', 'Conditions', 'Row', 'column', 'symbol']) or
                    re.search(r'[a-zA-Zа-яА-Я]', line) and not re.search(r'[0-9]', line)):
                    continue
                
                if re.search(r'[-]?\d+[,.]?\d*', line):
                    clean_line = line.replace(',', '.')
                    numbers = re.findall(r'[-]?\d+\.?\d*', clean_line)
                    
                    if numbers:
                        try:
                            numeric_values = [float(num) for num in numbers]
                            data_lines.append(numeric_values)
                        except ValueError:
                            continue
        
        if data_lines:
            max_cols = max(len(row) for row in data_lines)
            
            aligned_data = []
            for row in data_lines:
                if len(row) < max_cols:
                    row.extend([0.0] * (max_cols - len(row)))
                aligned_data.append(row)
            
            df = pd.DataFrame(aligned_data)
            print(f"Read {len(df)} rows, {len(df.columns)} columns from .dat file")
            return df
        else:
            print("No numerical data found in .dat file")
            return None
            
    except Exception as e:
        print(f"Error reading .dat file {dat_file_path}: {e}")
        return None

def process_file_pair(folder_path, base_filename):
    """
    Processes .par and .dat file pair with the same name
    """
    par_file = os.path.join(folder_path, base_filename + '.par')
    dat_file = os.path.join(folder_path, base_filename + '.dat')
    
    if not os.path.exists(par_file):
        print(f".par file not found: {par_file}")
        return None
    if not os.path.exists(dat_file):
        print(f".dat file not found: {dat_file}")
        return None

    peaks_data = parse_par_file(par_file)
    if not peaks_data:
        print("Failed to extract data from .par file")
        return None

    dat_df = parse_dat_file_simple(dat_file)
    if dat_df is None or dat_df.empty:
        print("Failed to read data from .dat file")
        return None
    
    print(f".dat data: {len(dat_df)} rows, {len(dat_df.columns)} columns")
    print(f"First 5 rows of .dat:")
    print(dat_df.head())
    
    result_dict = {}
    
    sorted_peak_numbers = sorted(peaks_data.keys())
    
    for i, peak_num in enumerate(sorted_peak_numbers):
        peak_data = peaks_data[peak_num]
        
        peak_col_index = 3 + i  
        
        if peak_col_index < len(dat_df.columns):
            result_dict[f'peak_{peak_num}'] = {
                'position': peak_data['position'],
                'area': peak_data['area'],
                'fwhm': peak_data['fwhm'],
                'gl': peak_data['gl'],
                'B.E.': dat_df.iloc[:, 0].tolist(),
                'intensity': dat_df.iloc[:, peak_col_index].tolist()
            }
            print(f"Added peak {peak_num} from column {peak_col_index}")
        else:
            print(f"Column {peak_col_index} not found in .dat file for peak {peak_num}")
            print(f"Available columns: 0-{len(dat_df.columns)-1}")
    
    return result_dict

def process_folder(folder_path, output_folder=None):
    """
    Processes all .par and .dat file pairs in specified folder
    Saves each result to separate JSON file
    """
    all_results = {}
    
    par_files = [f for f in os.listdir(folder_path) if f.endswith('.par')]
    print(f"Found .par files: {len(par_files)}")
    print(f"Files: {par_files}")
    
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
    
    for par_file in par_files:
        base_filename = os.path.splitext(par_file)[0]
        print(f"Processing files: {base_filename}")
        
        
        result = process_file_pair(folder_path, base_filename)
        if result:
            all_results[base_filename] = result
            print(f"Successfully processed {len(result)} peaks")
            
            json_filename = f"{base_filename}_results.json"
            if output_folder:
                json_path = os.path.join(output_folder, json_filename)
            else:
                json_path = json_filename
                
            success = save_single_result_to_json(result, json_path, base_filename)
            if success:
                print(f"Results for {base_filename} saved to: {json_path}")
        else:
            print(f"Failed to process files {base_filename}")
    
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
    """
    Saves results of one file pair to JSON
    """
    try:
        json_ready_data = convert_for_json(result)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_ready_data, f, indent=2, ensure_ascii=False)
        
        print(f" File {filename}: processed {len(result)} peaks")
        total_points = sum(len(peak_data['B.E.']) for peak_data in result.values())
        print(f"Total data points: {total_points}")
        
        return True
        
    except Exception as e:
        print(f"Error saving JSON for {filename}: {e}")
        return False

def print_summary(results):
    """
    Prints brief results statistics
    """
    if not results:
        print("No results to display")
        return
    
    total_files = len(results)
    total_peaks = sum(len(peaks) for peaks in results.values())
    
    print(f"Processed files: {total_files}")
    print(f"Total peaks: {total_peaks}")
    print(f"\nFile details:")
    
    for filename, peaks in results.items():
        print(f"  {filename}: {len(peaks)} peaks")
        for peak_name, peak_data in peaks.items():
            print(f"    {peak_name}: {len(peak_data['B.E.'])} data points")

if __name__ == "__main__":
    folder_path = "C:\\Users\\User\\Desktop\\exported_spec"
    output_folder = "C:\\Users\\User\\Desktop\\Json_spec"  
        
    if os.path.exists(folder_path):
        print(f"Processing folder: {folder_path}")
        results = process_folder(folder_path, output_folder=output_folder)
        
        if results:
            total_files = len(results)
            total_peaks = sum(len(peaks) for peaks in results.values())
            
            print(f"Processed files: {total_files}")
            print(f"Total peaks: {total_peaks}")
            print(f"\nFile details:")
            
            for filename, peaks in results.items():
                print(f"  {filename}: {len(peaks)} peaks")
                total_points = sum(len(peak_data['B.E.']) for peak_data in peaks.values())
                print(f"    Total data points: {total_points}")
            
        else:
            print("Failed to process any file pairs")
    else:
        print("Specified folder does not exist!")