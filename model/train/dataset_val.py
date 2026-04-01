import json
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from core.tools import interpolate


def init_stats() -> dict:
    return {
        "total_spectra": 0,
        "fixed_negative_intensity_count": 0,
        "total_negative_points": 0,
        "skipped_too_many_negatives": 0,
        "file_counter": 0,
        "total_peaks_count": 0,
        "spectra_with_peaks": 0,
    }

def fix_negative_intensities(y, max_negative_streak=10) -> np.ndarray | None:
    y_fixed = y.copy()
    negative_indices = np.where(y < 0)[0]

    max_streak = 0
    current_streak = 0
    for value in y:
        if value < 0:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    if max_streak >= max_negative_streak:
        return None

    for idx in negative_indices:
        left_idx = idx - 1
        while left_idx >= 0 and y[left_idx] < 0:
            left_idx -= 1

        right_idx = idx + 1
        while right_idx < len(y) and y[right_idx] < 0:
            right_idx += 1

        if left_idx >= 0 and right_idx < len(y):
            x_left = left_idx
            y_left = y[left_idx]
            x_right = right_idx
            y_right = y[right_idx]
            y_fixed[idx] = y_left + (y_right - y_left) * (idx - x_left) / (x_right - x_left)
        elif left_idx >= 0:
            y_fixed[idx] = y[left_idx]
        elif right_idx < len(y):
            y_fixed[idx] = y[right_idx]

    return y_fixed


def gauss_profile(x, position, fwhm) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-((x - position) ** 2) / (2.0 * sigma ** 2))


def lorentz_profile(x, position, fwhm) -> np.ndarray:
    gamma = fwhm / 2.0
    return 1.0 / (1.0 + ((x - position) / gamma) ** 2)


def peak_profile(x, position, fwhm, gl) -> np.ndarray:
    lorentz_fraction = np.clip(gl / 100.0, 0.0, 1.0)
    gauss_fraction = 1.0 - lorentz_fraction
    profile = gauss_fraction * gauss_profile(x, position, fwhm) + lorentz_fraction * lorentz_profile(x, position, fwhm)
    return profile.astype(np.float32)


def profile_mask(profile, threshold) -> np.ndarray:
    if profile.size == 0:
        return np.zeros_like(profile, dtype=np.float32)
    level = float(profile.max()) * threshold
    return (profile >= level).astype(np.float32)


def safe_normalize(y) -> np.ndarray:
    y_min = y.min()
    y_max = y.max()
    denom = y_max - y_min
    if denom <= 0:
        return np.zeros_like(y, dtype=np.float32)
    return ((y - y_min) / denom).astype(np.float32)


def process_spectrum(spectrum_data, source_name, peak_threshold, max_threshold, print_data) -> tuple | None:
    be = spectrum_data["BE"]
    start_x = be["start"]
    step_x = be["step"]
    x_num = int(be["num_points"])

    x = np.linspace(start_x, start_x + step_x * (x_num - 1), x_num, dtype=np.float32)
    y = np.array(spectrum_data["raw_intensity"], dtype=np.float32)

    negative_points = int(np.sum(y < 0))
    if negative_points > 0:
        if print_data:
            print(f"Fixing {negative_points} negative points in '{source_name}'")
        y_fixed = fix_negative_intensities(y)
        if y_fixed is None:
            if print_data:
                print(f"Skipping '{source_name}' - too many consecutive negative points")
            return None
        y = y_fixed

    y = safe_normalize(y)
    if len(y) != 256:
        x, y = interpolate(x, y, 256)

    y_log = np.log(10 * y + 1)
    y_log = safe_normalize(y_log)

    peak_mask = np.zeros_like(x, dtype=np.float32)
    max_mask = np.zeros_like(x, dtype=np.float32)

    valid_peaks = 0
    for peak in spectrum_data["peaks"]:
        position = peak.get("position")
        fwhm = peak.get("fwhm")
        gl = peak.get("gl", 50.0)
        if position is None or fwhm is None:
            continue
        if not np.isfinite(position) or not np.isfinite(fwhm) or not np.isfinite(gl):
            continue
        if fwhm <= 0:
            continue

        profile = peak_profile(x, position, fwhm, gl)
        peak_mask += profile_mask(profile, peak_threshold)
        max_mask += profile_mask(profile, max_threshold)
        valid_peaks += 1

    peak_mask[peak_mask > 0] = 1
    max_mask[max_mask > 0] = 1

    return y, y_log, peak_mask, max_mask, valid_peaks, negative_points


def generate_dataset(json_dir, output_dir, peak_threshold=0.2, max_threshold=0.93, print_data=False) -> dict:
    json_dir = Path(json_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = init_stats()
    json_files = sorted(json_dir.glob("*.json"))

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as file:
            data = json.load(file)

        spectrum_name = json_file.stem
        spectrum_data = data
        if "BE" not in spectrum_data or "raw_intensity" not in spectrum_data or "peaks" not in spectrum_data:
            continue

        stats["total_spectra"] += 1
        source_name = f"{json_file.name}:{spectrum_name}"

        result = process_spectrum(
            spectrum_data=spectrum_data,
            source_name=source_name,
            peak_threshold=peak_threshold,
            max_threshold=max_threshold,
            print_data=print_data,
        )

        if result is None:
            stats["skipped_too_many_negatives"] += 1
            continue

        y, y_log, peak_mask, max_mask, valid_peaks, negative_points = result
        if negative_points > 0:
            stats["fixed_negative_intensity_count"] += 1
            stats["total_negative_points"] += negative_points

        output_file = output_dir / f"{stats['file_counter']}.csv"
        data = pd.DataFrame(np.stack((y, y_log, peak_mask, max_mask), axis=1))
        data.to_csv(output_file, header=False, index=False)

        stats["total_peaks_count"] += valid_peaks
        if valid_peaks > 0:
            stats["spectra_with_peaks"] += 1
        stats["file_counter"] += 1

    if print_data:
        print(f"Total spectra processed: {stats['total_spectra']}")
        print(f"Spectra with negative intensities: {stats['fixed_negative_intensity_count']}")
        print(f"Total negative points found: {stats['total_negative_points']}")
        print(f"Spectra skipped (too many negatives): {stats['skipped_too_many_negatives']}")
        print(f"Remaining spectra: {stats['file_counter']}")
        print(f"Total peaks in spectra: {stats['total_peaks_count']}")
        if stats["spectra_with_peaks"] > 0:
            avg_peaks = stats["total_peaks_count"] / stats["spectra_with_peaks"]
            print(f"Average number of peaks: {avg_peaks}")
        else:
            print("Average number of peaks: 0")

    return stats


def main():
    peak_threshold = 0.2
    max_threshold = 0.93
    print_data = False
    train_json_dir = Path.home() / "Desktop" / "train"
    val_json_dir = Path.home() / "Desktop" / "val"
    train_out_dir = Path("../model/train/data/train")
    val_out_dir = Path("../model/train/data/val")

    parser = argparse.ArgumentParser(description="Generate train/val CSV datasets from XPS JSON files")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    train_out_dir = (project_root / train_out_dir).resolve()
    val_out_dir = (project_root / val_out_dir).resolve()

    if args.clean:
        shutil.rmtree(train_out_dir, ignore_errors=True)
        shutil.rmtree(val_out_dir, ignore_errors=True)

    train_stats = generate_dataset(
        json_dir=train_json_dir,
        output_dir=train_out_dir,
        peak_threshold=peak_threshold,
        max_threshold=max_threshold,
        print_data=print_data,
    )
    val_stats = generate_dataset(
        json_dir=val_json_dir,
        output_dir=val_out_dir,
        peak_threshold=peak_threshold,
        max_threshold=max_threshold,
        print_data=print_data,
    )

    print("Train stats:", train_stats)
    print("Val stats:", val_stats)


if __name__ == "__main__":
    main()
