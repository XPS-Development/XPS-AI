import json
import re
from pathlib import Path


NUM_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?")


def to_float(text) -> float:
    return float(text.replace(",", "."))


def parse_kv(line) -> tuple | None:
    clean = line.strip()
    if not clean:
        return None

    if ":" in clean:
        key, value = clean.split(":", 1)
    elif "=" in clean:
        key, value = clean.split("=", 1)
    else:
        parts = re.split(r"\s{2,}|\t+", clean)
        if len(parts) < 2:
            return None
        key = parts[0]
        value = parts[-1]

    key = key.strip()
    value = value.strip()
    if not key or not value:
        return None

    return key, value


def parse_par(par_path) -> list:
    peaks = []

    with open(par_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    in_table = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        low = line.lower()
        if "peak" in low and "position" in low and "fwhm" in low:
            in_table = True
            continue

        if in_table:
            nums = NUM_RE.findall(line)
            if len(nums) >= 5:
                peaks.append(
                    {
                        "position": to_float(nums[1]),
                        "area": to_float(nums[2]),
                        "fwhm": to_float(nums[3]),
                        "gl": to_float(nums[4]),
                    }
                )
                continue

            in_table = False

    return peaks


def parse_dat(dat_path) -> tuple:
    be_vals = []
    raw_vals = []
    bg_be_vals = []
    bg = {}

    with open(dat_path, "r", encoding="utf-8", errors="ignore") as file:
        lines = file.readlines()

    data_on = False

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        low = line.lower()
        if "b.e." in low and "raw intensity" in low:
            data_on = True
            continue

        if not data_on:
            parsed = parse_kv(line)
            if parsed:
                key, val_txt = parsed
                if "background" in key.lower():
                    nums = NUM_RE.findall(val_txt)
                    if len(nums) == 1:
                        bg[key] = to_float(nums[0])
                    elif nums:
                        bg[key] = [to_float(num) for num in nums]
                    else:
                        bg[key] = val_txt
            continue

        nums = NUM_RE.findall(line)
        if len(nums) < 2:
            continue

        be = to_float(nums[0])
        be_vals.append(be)
        raw_vals.append(to_float(nums[1]))

        if len(nums) >= 4:
            bg_be_vals.append(be)

    if bg_be_vals:
        bg.setdefault("low be", min(bg_be_vals))
        bg.setdefault("high be", max(bg_be_vals))
    elif be_vals:
        bg.setdefault("low be", min(be_vals))
        bg.setdefault("high be", max(be_vals))

    return be_vals, raw_vals, bg


def parse_bg_file(par_path) -> dict:
    bg_path = par_path.with_suffix(".bg.json")
    if not bg_path.exists():
        return {}

    with open(bg_path, "r", encoding="utf-8", errors="ignore") as file:
        return json.load(file)


def norm_bg_keys(bg_data) -> dict:
    out = {}
    for key, value in bg_data.items():
        out[str(key).lower()] = value
    return out


def build_json(par_path, dat_path) -> dict | None:
    peaks = parse_par(par_path)
    be_vals, raw_vals, bg = parse_dat(dat_path)
    bg_extra = parse_bg_file(par_path)
    bg = norm_bg_keys(bg)
    bg_extra = norm_bg_keys(bg_extra)
    bg.update(bg_extra)
    if "background type" in bg and bg["background type"] is not None:
        bg["background type"] = str(bg["background type"]).lower()

    if not be_vals or not raw_vals:
        return None

    if len(be_vals) > 1:
        step = round(be_vals[1] - be_vals[0], 6)
    else:
        step = 0.0

    return {
        "BE": {
            "start": be_vals[0],
            "step": step,
            "num_points": len(be_vals),
        },
        "raw_intensity": raw_vals,
        "peaks": peaks,
        "background": bg,
    }


def next_json_idx(out_dir) -> int:
    max_idx = 0
    for json_file in out_dir.glob("*.json"):
        stem = json_file.stem
        if stem.isdigit():
            idx = int(stem)
            if idx > max_idx:
                max_idx = idx
    return max_idx + 1


def convert_dir(in_dir, out_dir) -> list:
    out_dir.mkdir(parents=True, exist_ok=True)
    par_files = sorted(in_dir.glob("*.par"))

    converted = []
    idx = next_json_idx(out_dir)
    for par_file in par_files:
        dat_file = in_dir / f"{par_file.stem}.dat"
        if not dat_file.exists():
            continue

        spec = build_json(par_file, dat_file)
        if not spec:
            continue

        out_path = out_dir / f"{idx}.json"
        with open(out_path, "w", encoding="utf-8") as file:
            json.dump(spec, file, ensure_ascii=False, indent=2)
        converted.append(out_path)
        idx += 1

    return converted


def main():
    root = Path(__file__).resolve().parents[2]
    in_dir = (root / "../../Desktop/exported_spec").resolve()
    out_dir = (root / "../../Desktop/Json_spec").resolve()
    converted = convert_dir(in_dir, out_dir)
    print(f"Converted {len(converted)} file(s)")


if __name__ == "__main__":
    main()
