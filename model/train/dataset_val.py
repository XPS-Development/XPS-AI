import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ValXPSDataGenerator:
    def __init__(
            self,
            json_dir,
            output_dir,
            width_peak = 1.23,
            width_max = 0.2,
            min_relative_area = 0.002,
            ):
        self.json_dir = Path(json_dir)
        self.output_dir = Path(output_dir)
        self.width_peak = width_peak
        self.width_max = width_max
        self.min_relative_area = min_relative_area

        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def check_peak_monotonicity(self, peak_intensities, limit = 5.0) -> bool:        
        intensities = np.array(peak_intensities)
        gradient = np.gradient(intensities)
        num_points = len(gradient)
        trim_points = int(self.trim_percentage * num_points)

        if num_points > 2 * trim_points:
            gradient[:trim_points] = 0
            gradient[-trim_points:] = 0
        
        return np.any(gradient > limit)

    def negative_intensities(self, peak_intensities) -> bool:   
        for intensity in peak_intensities:
            if intensity < 0:
                return True
        return False

    def create_mask(self, x, from_x, to_x) -> np.ndarray:  
        zeros = np.zeros_like(x)
        zeros[(x > from_x) & (x < to_x)] = 1
        return zeros

    def process_spectrum(self, spectrum_data, json_filename, spectrum_name) -> tuple:
        start_x = spectrum_data['BE']['start']
        step_x = spectrum_data['BE']['step']
        x_num = spectrum_data['BE']['num_points']

        x = np.linspace(start_x, start_x + step_x * (x_num - 1), x_num)
        y = np.array(spectrum_data['raw_intensity'], dtype=np.float32)

        if np.any(y < 0):
            print(f"Пропускаем '{spectrum_name}' (файл: {json_filename}) - отрицательные интенсивности")
            return None, None, None, None

        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        
        y_log = np.log(10 * y + 1)
        
        y_log = (y_log - y_log.min()) / (y_log.max() - y_log.min() + 1e-8)

        peak_mask = np.zeros_like(x)
        max_mask = np.zeros_like(x)

        peaks_dict = spectrum_data.get('peaks', {})
        
        total_area = sum(peak_info['area'] for peak_info in peaks_dict.values())
        
        valid_peaks_count = 0
        
        for peak_num, peak_info in peaks_dict.items():
            position = peak_info['position']
            fwhm = peak_info['fwhm']
            area = peak_info['area']
            peak_intensities = peak_info.get('intensity', [])
            
            relative_area = area / total_area if total_area > 0 else 0
            
            if relative_area < self.min_relative_area:
                print(f"пик {peak_num} в '{spectrum_name}' (файл: {json_filename}) с относительной площадью {relative_area:.4f} < {self.min_relative_area}")
                continue
            
            if self.negative_intensities(peak_intensities):
                print(f"пик {peak_num} в '{spectrum_name}' (файл: {json_filename}) с отрицательной интенсивностью")
                continue
            
            if len(peak_intensities) > 0:
                if self.check_peak_monotonicity(peak_intensities):
                    peak_mask += self.create_mask(x, position - self.width_peak * fwhm, position + self.width_peak * fwhm)
                    max_mask += self.create_mask(x, position - self.width_max, position + self.width_max)
                    valid_peaks_count += 1
                else:
                    print(f"пик {peak_num} в '{spectrum_name}' (файл: {json_filename}) отфильтрован градиентом")
        
        peak_mask[peak_mask > 0] = 1
        max_mask[max_mask > 0] = 1
        
        return y, y_log, peak_mask, max_mask

    def generate_dataset(self):
        json_files = list(self.json_dir.glob('*.json'))
 
        file_counter = 0
        total_peaks_count = 0
        spectra_with_peaks = 0
        total_spectra = 0
        filtered_negative_intensity = 0
        
        for json_file in json_files:
            json_filename = json_file.name
            print(f"{json_filename}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            spectra = list(data['spectra'].values())
            spectrum_names = list(data['spectra'].keys())
                
            for spectrum_data, spectrum_name in zip(spectra, spectrum_names):
                total_spectra += 1
                result = self.process_spectrum(spectrum_data, json_filename, spectrum_name)
                
                if result[0] is None:
                    filtered_negative_intensity += 1
                    continue
                    
                y, y_log, peak_mask, max_mask = result
                    
                data_array = np.stack((y, y_log, peak_mask, max_mask), axis=1)
                df = pd.DataFrame(data_array)
                    
                output_file = self.output_dir / f'{file_counter}.csv'
                df.to_csv(output_file, header=False, index=False)
                
                peaks_dict = spectrum_data.get('peaks', {})
                valid_peaks_in_spectrum = 0
                
                total_area = sum(peak_info['area'] for peak_info in peaks_dict.values())
                
                for peak_info in peaks_dict.values():
                    area = peak_info['area']
                    relative_area = area / total_area if total_area > 0 else 0
                    
                    if relative_area < self.min_relative_area:
                        continue
                    if self.negative_intensities(peak_info.get('intensity', [])):
                        continue
                    if self.check_peak_monotonicity(peak_info.get('intensity', [])):
                        valid_peaks_in_spectrum += 1
                
                total_peaks_count += valid_peaks_in_spectrum
                spectra_with_peaks += 1
                file_counter += 1
    
        print(f"Всего обработано спектров: {total_spectra}")
        print(f"Отфильтровано из-за отрицательных интенсивностей: {filtered_negative_intensity}")
        print(f"Осталось спектров: {file_counter}")
        print(f"Всего пиков в спектрах: {total_peaks_count}")
        print(f"Среднее количество пиков: {total_peaks_count / spectra_with_peaks:.2f}")


    def view_labeled_data(self, file_index: int):
        file_path = self.output_dir / f'{file_index}.csv'
        
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

        plt.show()

def main():
    json_dir = 'C:/Users/User/Desktop/Json_spec'
    output_dir = 'C:/Users/User/Desktop/RealXPSDataset'
    
    generator = ValXPSDataGenerator(
        json_dir, 
        output_dir,
        min_relative_area=0.002
    )
    generator.generate_dataset()
    
    generator.view_labeled_data(0)

if __name__ == "__main__":
    main()