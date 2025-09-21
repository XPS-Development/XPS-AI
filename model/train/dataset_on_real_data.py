import json
from pathlib import Path
import numpy as np
import torch 
from torch.utils.data import Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib.pyplot as plt
from algorithm_peak_exception import extract_peaks, prep_features, fit_groups, apply_rules


class RealXPSDataset(Dataset):
    def __init__(
            self,
            json_dir,
            width_peak=1.23,
            width_max=0.2
            ):
        super().__init__()
        self.width_peak = width_peak
        self.width_max = width_max
        self.json_files = list(Path(json_dir).glob('*.json'))
        self.spectra_list = []
    
        self.classifier = self.train_classifier(json_dir)
        
        for json_file in self.json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
        
            spectra = list(data['spectra'].values())
            self.spectra_list.extend(spectra)
        
    def __len__(self):  
        return len(self.spectra_list)
    
    def __getitem__(self, index):
        spectrum_data = self.spectra_list[index]

        start_x = spectrum_data['BE']['start']
        step_x = spectrum_data['BE']['step']
        x_num = spectrum_data['BE']['num_points']

        x = np.linspace(start_x, start_x + step_x * (x_num - 1), x_num)
        y = np.array(spectrum_data['raw_intensity'], dtype=np.float32)
        
        y = np.clip(y, 1e-8, None)
    
        y_log = np.log(10 * y + 1)

        peak_mask = np.zeros_like(x)
        max_mask = np.zeros_like(x)

        peaks_dict = spectrum_data.get('peaks', {})
        for peak_num, peak_info in peaks_dict.items():
            position = peak_info['position']
            fwhm = peak_info['fwhm']
            area = peak_info['area']
            gl = peak_info['gl']

            if self.peak_exception(area, fwhm, gl) == True:
                peak_mask += self.create_mask(x, position - self.width_peak*fwhm, position + self.width_peak*fwhm)
                max_mask += self.create_mask(x, position - self.width_max, position + self.width_max)
            else:
                continue
        
        peak_mask[peak_mask > 0] = 1
        max_mask[max_mask > 0] = 1
        
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        y_log = (y_log - y_log.min()) / (y_log.max() - y_log.min() + 1e-8)
        
        features = torch.stack((
            torch.tensor(y, dtype=torch.float32),
            torch.tensor(y_log, dtype=torch.float32)
        ), dim=0)

        targets = torch.stack((
            torch.tensor(peak_mask, dtype=torch.float32),
            torch.tensor(max_mask, dtype=torch.float32)
        ), dim=0) 

        return features, targets, x
    
    def peak_exception(self, area, fwhm, gl):
        match = self.classifier[
            (self.classifier['area'] == area) &
            (self.classifier['fwhm'] == fwhm) &
            (self.classifier['gl'] == gl)
        ]
            
        if len(match) > 0:
            group_name = match['final_group_name'].values[0]
            if group_name == 'Outlier':
                 return False
            else:
                return True
                
    def train_classifier(self, json_dir):
        peaks_df = extract_peaks(json_dir)
        
        prep = prep_features(peaks_df)
        prep, group_names = fit_groups(prep, n_groups=3)
        prep['group_name'] = prep['group'].map(group_names)
        prep = apply_rules(prep)
        return prep

    def create_mask(self, x, from_x, to_x):  
        zeros = np.zeros_like(x)
        zeros[(x > from_x) & (x < to_x)] = 1
        return zeros
    
    def view_labeled_data(self, index):
        features, targets, x = self[index]
        y = features[0]
        peak_mask = targets[0]
        max_mask = targets[1]
        plt.plot(x, y, 'k')
        plt.gca().invert_xaxis() 
        plt.fill_between(x, y, y.min(), where=peak_mask > 0)
        plt.fill_between(x, y, y.min(), where=max_mask > 0)
        plt.xlabel('be')
        plt.ylabel('intensity')
        plt.show()
        
# dataset = RealXPSDataset('C:/Users/User/Desktop/Json_spec')
# dataset.view_labeled_data(0)