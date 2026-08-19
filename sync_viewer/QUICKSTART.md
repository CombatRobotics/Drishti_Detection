# Quick Start Guide

## Folder Structure

```
Drishti_Detection/
├── sync_viewer/                          ← NEW: All viewer code
│   ├── bag_camera_encoder_extractor.py   ← Step 1: Extract from bags
│   ├── sync_index.py                     ← Step 2: Index lookup library
│   ├── sync_camera_viewer_minimal.py     ← Step 3: Single-camera test viewer
│   ├── camera_topics_config.json         ← Camera list (edit to add cameras)
│   ├── README.md                         ← Full documentation
│   ├── EXTRACTION_EXPLAINED.md           ← How 3-bag extraction works
│   └── QUICKSTART.md                     ← This file
├── requirements.txt                      ← Updated with PySide6
└── ... (existing detection/fault code)
```

---

## Installation

```bash
# Install PySide6 (required for viewer)
pip install PySide6
```

---

## Usage Workflow

### 1. Extract a Session (One-Time)

```bash
cd sync_viewer

# Extract all cameras from a session
python3 bag_camera_encoder_extractor.py --session /media/.../Day-4

# Or extract just specific cameras (faster)
python3 bag_camera_encoder_extractor.py --session /media/.../Day-4 --cameras rail_left,plinth_left,dart_front_right

# With different playback speed (default 0.5x, try 1.0 for faster)
python3 bag_camera_encoder_extractor.py --session /media/.../Day-4 --rate 1.0
```

**Time:** 1–2 hours depending on playback rate. Creates `Day-4/sync_extract/`.

### 2. View Extracted Data

```bash
# Run the minimal viewer (Step 3 - for testing)
python3 sync_camera_viewer_minimal.py
# Then click "Open Folder" and select Day-4/sync_extract/

# (Coming soon: Full viewer with grid + overlays + threading)
```

---

## Understanding the Output

After extraction completes, check what was created:

```bash
# View manifest (what was extracted)
cat Day-4/sync_extract/manifest.json | jq .

# List extracted frames
ls -la Day-4/sync_extract/frames/rail_left/ | head -20

# Check index (timestamp mappings)
head -20 Day-4/sync_extract/index/rail_left.csv
head -20 Day-4/sync_extract/index/encoder_left.csv
```

**Key things to verify:**
- `manifest.json`: encoder and camera timestamp ranges overlap (validates --clock/sim-time fix)
- `frame_count` in manifest vs. expected count from bag's metadata.yaml
- Encoder CSV has ~60k samples (sampled at ~1000 Hz for a 58-minute session)

---

## Customizing: Add a New Camera

1. Edit `camera_topics_config.json`:

```json
{
  "cameras": [
    ... (existing cameras) ...
    {"key": "my_new_camera", "topic": "/path/to/my/camera/image/compressed"}
  ],
  "encoders": { "left": "/left_encoder_mm", "right": "/right_encoder_mm" }
}
```

2. Extract (no code changes needed):

```bash
python3 bag_camera_encoder_extractor.py --session /path/to/session --cameras my_new_camera
```

---

## Common Issues & Fixes

### "No rosbag2 folders (metadata.yaml) found"
- **Cause:** You didn't point to the session directory (the one containing plinth_cameras, railhead_cameras, etc.)
- **Fix:** Use the parent folder: `--session /media/.../Day-4/` not `/media/.../Day-4/plinth_cameras/`

### "ModuleNotFoundError: No module named 'PySide6'"
- **Fix:** `pip install PySide6`

### "rclpy not found"
- **Cause:** ROS2 not sourced
- **Fix:** `source /opt/ros/jazzy/setup.bash` (or your ROS2 distro)

### Extraction taking too long
- **Try:** `--rate 1.0` instead of default `--rate 0.5`
  - 0.5x = takes 2 hours, more stable
  - 1.0x = takes 1 hour, might drop a few frames under load
  - 2.0x = takes 30 min, higher risk of frame drops

### Frame count much lower than expected in manifest.json
- **Likely:** Network load or bag format issues during playback
- **Try:** Retry with slower rate: `--rate 0.25`

---

## What Each File Does

| File | Purpose |
|------|---------|
| `bag_camera_encoder_extractor.py` | Reads 3 bag folders in parallel, extracts raw JPEGs + encoder samples, writes index CSVs |
| `sync_index.py` | Python module to load index, lookup nearest frames, interpolate encoder values, find by distance |
| `sync_camera_viewer_minimal.py` | Minimal Qt GUI (Step 3): one camera, one slider, test the pipeline |
| `camera_topics_config.json` | List of all known cameras + their ROS topic names (edit to add new ones) |

---

## Data Layout (After Extraction)

```
Day-4/sync_extract/
├── manifest.json                    ← What was extracted
├── frames/
│   ├── rail_left/                   ← One subdirectory per camera
│   │   ├── 1783371529164761323.jpg  ← Filenames = nanosecond timestamps
│   │   ├── 1783371529197761323.jpg
│   │   └── ... (100k+ files)
│   ├── rail_right/
│   ├── plinth_left/
│   ├── plinth_right/
│   ├── dart_front_right/
│   └── zed_rgb/
└── index/
    ├── rail_left.csv                ← Timestamp index per camera
    ├── rail_right.csv
    ├── plinth_left.csv
    ├── plinth_right.csv
    ├── dart_front_right.csv
    ├── zed_rgb.csv
    ├── encoder_left.csv             ← Encoder time series
    └── encoder_right.csv
```

- **frames/** : Raw JPEG files, organized by camera
- **index/*.csv** : Lookup tables (timestamp → filepath)
- **manifest.json** : Metadata (what was extracted, timestamps, counts)

---

## Progress So Far

✅ **Step 1**: Extraction script (running on real data)
✅ **Step 2**: Index lookup (self-test passing)
✅ **Step 3**: Minimal viewer (ready to test)

**Next:**
- Complete extraction (wait for it to finish)
- Test Step 3 viewer
- Build Steps 4–7 (full grid, controls, overlays, threading)

---

## Support

- **Full docs:** See `README.md` and `EXTRACTION_EXPLAINED.md`
- **Parameters:** Run `python3 bag_camera_encoder_extractor.py --help`
- **Questions:** Check README.md for architecture overview and parameter docs

---

## One-Command Testing (When Extraction Done)

```bash
# Test the minimal viewer
python3 sync_camera_viewer_minimal.py
# Opens an empty viewer. Click "Open Folder" and select Day-4/sync_extract/
```

This validates that extraction → index → display pipeline works end-to-end before adding complexity.
