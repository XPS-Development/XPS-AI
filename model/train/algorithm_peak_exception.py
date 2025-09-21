import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def extract_peaks(json_dir):
    peaks_data = []
    json_files = list(Path(json_dir).glob('*.json'))
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for spectrum_name in data['spectra']:
            spectrum_data = data['spectra'][spectrum_name]
            if 'peaks' in spectrum_data:
                spectrum_peaks = []
                for peak_name in spectrum_data['peaks']:
                    peak_params = spectrum_data['peaks'][peak_name]
                    peak_info = {
                        'file': json_file.name,
                        'spectrum': spectrum_name,
                        'area': peak_params['area'],
                        'fwhm': peak_params['fwhm'],
                        'gl': peak_params['gl']
                    }
                    spectrum_peaks.append(peak_info)
                
                if spectrum_peaks:
                    areas = [p['area'] for p in spectrum_peaks]
                    max_area = max(areas)
                    for peak_info in spectrum_peaks:
                        peak_info['relative_area'] = peak_info['area'] / max_area
                        peaks_data.append(peak_info)
    
    return pd.DataFrame(peaks_data)

def prep_features(df):
    df = df.copy()
    df['ratio'] = df['area'] / (df['fwhm'] + 0.0001)
    df['log_relative_area'] = np.log(df['relative_area'] + 0.0001)
    df['log_fwhm'] = np.log(df['fwhm'] + 1)
    df['log_ratio'] = np.log(df['ratio'] + 1)
    
    return df

def fit_groups(df, n_groups=3):
    scaler = StandardScaler()
    kmeans = KMeans(n_clusters=n_groups, random_state=50)
    
    features = df[['log_relative_area', 'log_fwhm', 'log_ratio']]
    features = features.dropna()
    
    scaled_features = scaler.fit_transform(features)
    group_labels = kmeans.fit_predict(scaled_features)
    
    df = df.copy()
    df['group'] = None
    for i in range(len(features)):
        df_index = features.index[i]
        df.loc[df_index, 'group'] = group_labels[i]
    
    group_stats = []
    for group_id in range(n_groups):
        group_data = df[df['group'] == group_id]
        if len(group_data) > 0:
            stats = {
                'group_id': group_id,
                'mean_relative_area': group_data['relative_area'].mean(),
                'mean_fwhm': group_data['fwhm'].mean(),
                'mean_ratio': group_data['ratio'].mean(),
            }
            group_stats.append(stats)
    
    for i in range(len(group_stats)):
        for j in range(i+1, len(group_stats)):
            if group_stats[i]['mean_relative_area'] > group_stats[j]['mean_relative_area']:
                group_stats[i], group_stats[j] = group_stats[j], group_stats[i]
    
    group_names = {}
    for i in range(len(group_stats)):
        group_id = group_stats[i]['group_id']
        stats = group_stats[i]
        
        if i == 0 and stats['mean_relative_area'] < 0.1:
            group_names[group_id] = 'Outlier'
        elif i == len(group_stats) - 1 and stats['mean_relative_area'] > 0.5:
            group_names[group_id] = 'Good peak'
        else:
            group_names[group_id] = 'peak'
    
    return df, group_names

def apply_rules(df):
    df = df.copy()
    df['final_group_name'] = df['group_name']
    unique_files = df['file'].unique()
    
    for file in unique_files:
        file_data = df[df['file'] == file]
        unique_spectra = file_data['spectrum'].unique()
        
        for spectrum in unique_spectra:
            p = (df['spectrum'] == spectrum)
            spectrum_data = df[p]
            
            if len(spectrum_data) <= 1:
                continue
                
            max_area = spectrum_data['area'].max()
            
            for idx in spectrum_data.index:
                relative_area = df.loc[idx, 'area'] / max_area
                
                if relative_area > 0.3:
                    df.loc[idx, 'final_group_name'] = 'Good peak'
                elif relative_area < 0.05:
                    df.loc[idx, 'final_group_name'] = 'Outlier'
    
    return df

def save_to_csv(df, filename='peak_groups_analysis.csv'):
    result = df[['spectrum', 'area', 'fwhm', 'gl', 'relative_area', 'final_group_name']]
    result.to_csv(filename)

json_dir = 'C:/Users/User/Desktop/Json_spec'
all_peaks = extract_peaks(json_dir)
prep = prep_features(all_peaks)     
prep, group_names = fit_groups(prep, n_groups=3)
prep['group_name'] = prep['group'].map(group_names)
prep = apply_rules(prep)
save_to_csv(prep)
