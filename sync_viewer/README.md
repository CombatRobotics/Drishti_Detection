# Synchronized Multi-Camera Bag Viewer

Complete toolkit for offline browsing of ROS2 rosbag2 recordings with synchronized camera feeds and encoder data.

## Quick Overview

### What's the Problem?
Your current scripts (`mcap_to_video.py`, `fault_detection_mcap.py`, etc.) extract frames from ONE camera at a time into video files or image dumps, with no interactive viewer. You need a **desktop UI to scrub through multiple camera feeds simultaneously**, synchronized by timestamp, to manually review rail defects, encoder distance at specific moments, etc.

### What Does This Solve?
1. **Extracts raw JPEG frames + encoder samples** from your 3-folder bag structure (plinth_cameras, railhead_cameras, track_measurement recorded in parallel)
2. **Stores indexed data** (timestamps, frame paths, encoder values) in CSV/JSON
3. **Provides a desktop viewer** to scrub, play, and inspect all cameras together
4. **No re-encoding** — uses raw JPEG bytes directly from the bag

---

## Architecture

### Step 1: Data Extraction (`bag_camera_encoder_extractor.py`)

**What it does:**
- Scans a session directory (e.g., `Day-4/`) for bag sub-folders
- Auto-discovers which bag contains which camera topics (via `metadata.yaml`)
- **Plays all needed bags simultaneously** using `ros2 bag play` (multiple concurrent processes, one per bag folder)
- **Only ONE bag publishes `/clock`** (the encoder-source bag) — this keeps encoder and image timestamps on the same axis (critical fix for header-less `Float32` messages)
- Decodes camera messages and writes **raw JPEG bytes** directly to disk (zero re-encoding, preserving exact quality)
- Records encoder samples with sim-time timestamps (synchronized to the bag's original recording clock)

**Where is data stored?**

```
Day-4/sync_extract/                    ← Output directory (default: <session>/sync_extract)
├── manifest.json                      ← Session metadata
├── frames/
│   ├── rail_left/
│   │   ├── 1783371529164761323.jpg   ← Filename = nanosecond timestamp (lossless, sortable)
│   │   ├── 1783371529197761323.jpg
│   │   └── ...
│   ├── rail_right/
│   │   └── ...
│   ├── plinth_left/
│   │   └── ...
│   ├── plinth_right/
│   │   └── ...
│   ├── dart_front_right/
│   │   └── ...
│   └── zed_rgb/
│       └── ...
└── index/
    ├── rail_left.csv                 ← Per-camera timestamp index
    ├── rail_right.csv
    ├── plinth_left.csv
    ├── plinth_right.csv
    ├── dart_front_right.csv
    ├── zed_rgb.csv
    ├── encoder_left.csv              ← Encoder time series (timestamp, distance_mm)
    └── encoder_right.csv
```

**What is manifest.json?**

A summary of what was extracted, for validation and provenance:

```json
{
  "schema_version": 1,
  "extraction_timestamp_iso": "2026-08-19T14:32:00+05:30",
  "output_dir": "/media/.../Day-4/sync_extract",
  "cameras": [
    {
      "key": "rail_left",
      "frame_count": 104691,
      "first_ts_ns": 1783371529164761323,
      "last_ts_ns": 1783375017750442599
    },
    ...
  ],
  "encoders": [
    {
      "side": "left",
      "sample_count": 3259430,
      "first_ts_ns": 1783371529164761323,
      "last_ts_ns": 1783375017750442599
    },
    ...
  ]
}
```

Use this to:
- Verify encoder/image timestamps overlap (validates `--clock` fix)
- Compare frame counts vs. bag's `metadata.yaml` to detect dropped frames
- Understand what was extracted when

---

### Step 2: Index Lookup (`sync_index.py`)

**What it does:**
- Loads CSVs into numpy arrays
- Provides lookup functions:
  - `nearest_frame(camera, timestamp)` — find the closest frame within 50ms (blanking threshold)
  - `interpolate_encoder_distance_mm(sync_index, timestamp)` — get distance at any moment
  - `find_timestamps_for_encoder_value(encoder_series, value)` — reverse lookup ("jump to where encoder distance = X")
- Handles non-monotonic encoder values (vehicle reversals, resets)
- Builds a "master timeline" (synthetic regular grid across all cameras' time range)

**Used by:** The viewer + validation scripts.

---

### Step 3: Desktop Viewer (`sync_camera_viewer_minimal.py` → full `sync_camera_viewer.py`)

**What UI are you building? Options:**

| Aspect | Option A (Chosen) | Option B | Why A? |
|--------|---|---|---|
| **GUI Toolkit** | PySide6 (Qt) | tkinter / PySimpleGUI | Qt is professional, responsive, full control |
| **Layout** | Grid (N cameras) | Fixed 2x3 tiles | Grid adapts to 1–6 cameras dynamically |
| **Playback** | QTimer + slider | Separate playback window | Slider is faster for scrubbing analysis |
| **Threading** | QThreadPool (decode off main thread) | Synchronous decode | Prevents UI freezing during long scrubs |
| **Overlay Rendering** | QPainter (at display time) | cv2.putText (burn into pixels) | No storage cost, toggleable, scalable font |

---

## Full UI Layout (Final, Step 4+)

```
┌─────────────────────────────────────────────────────────────┐
│ Sync Camera Viewer                              [_] [□] [X] │
├─────────────────────────────────────────────────────────────┤
│ [Open Folder]  ▶ Pause    Speed: 1x   🔘 Show Overlays      │
├──────────────────────┬──────────────────────────────────────┤
│ Camera Selector:     │     GRID (resizes based on selection) │
│ ☑ rail_left        │                                        │
│ ☑ rail_right       │   ┌─────────────┐  ┌─────────────┐    │
│ ☑ plinth_left      │   │ rail_left   │  │ rail_right  │    │
│ ☑ plinth_right     │   │             │  │             │    │
│ ☑ dart_front_right │   │ (400×300)   │  │ (400×300)   │    │
│ ☑ zed_rgb          │   │             │  │             │    │
│ [ All ]  [ None ]  │   │ Frame ▼ ▲   │  │ Frame ▼ ▲   │    │
│                    │   └─────────────┘  └─────────────┘    │
│                    │   ┌─────────────┐  ┌─────────────┐    │
│                    │   │ plinth_left │  │ plinth_right│    │
│                    │   │             │  │             │    │
│                    │   └─────────────┘  └─────────────┘    │
│                    │   ┌─────────────┐  ┌─────────────┐    │
│                    │   │dart_front_rt│  │  zed_rgb    │    │
│                    │   │             │  │  (sparse!)  │    │
│                    │   └─────────────┘  └─────────────┘    │
├──────────────────────┴──────────────────────────────────────┤
│ Timeline:  ◾━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│ 2026-07-07 02:28:52.164761 IST | Encoder Dist: 5234.2mm    │
├─────────────────────────────────────────────────────────────┤
│ Jump to timestamp (ns): [1783371529164761323________] [Go]  │
│ Jump to encoder dist (mm): [5234.2_________] [Go]           │
│ (If multiple matches, shows picker dialog)                  │
├─────────────────────────────────────────────────────────────┤
│ Status: Playing (frame 124 / 104691)                        │
└─────────────────────────────────────────────────────────────┘
```

**What are the overlays and where are they?**

Each camera tile shows a text overlay (top-left, semi-transparent background):

```
┌──────────────────────┐
│ rail_left            │
│ 2026-07-07 02:28:52  │  ← Camera name, timestamp, encoder distance
│ Encoder: 5234.2 mm   │     Rendered with QPainter, NOT baked into JPEG
│                      │     Toggle via "Show Overlays" checkbox
│                      │     Font size scales with tile size
│                      │
│                      │
│                      │  Click tile → Enlarged view (full-res, non-modal)
│                      │
│                      │
│                      │
│                      │
│                      │
│                      │
│                      │
│                      │
│ [◀ Prev]  [Next ▶]   │  ← Per-camera frame stepping (while paused)
└──────────────────────┘
```

---

## Parameters & Input Options

### 1. **Extraction Parameters** (`bag_camera_encoder_extractor.py`)

```bash
python3 bag_camera_encoder_extractor.py \
  --session /media/.../Day-4 \
  --output-dir /media/.../Day-4/sync_extract \
  --config camera_topics_config.json \
  --cameras rail_left,rail_right,plinth_left \
  --clock-rate 200 \
  --rate 0.5
```

| Parameter | Default | What It Does |
|-----------|---------|---|
| `--session` | **required** | Path to session folder (e.g., `Day-4`) |
| `--output-dir` | `<session>/sync_extract` | Where to store frames + index |
| `--config` | `./camera_topics_config.json` | Camera topic list (edit to add new cameras) |
| `--cameras` | all in config | Comma-separated camera keys to extract (e.g., `rail_left,zed_rgb`) |
| `--clock-rate` | 200 Hz | `/clock` publish frequency (for encoder timestamp sync) |
| `--rate` | 1.0x | Bag playback speed (0.5 = slower extraction, 2.0 = faster) |

**Camera config file** (`camera_topics_config.json` — edit this to add new cameras):

```json
{
  "cameras": [
    {"key": "rail_left", "topic": "/ace_camera_rail_left/.../image/compressed"},
    {"key": "rail_right", "topic": "/ace_camera_rail_right/.../image/compressed"},
    {"key": "plinth_left", "topic": "/ace_camera_plinth_left/.../image/compressed"},
    {"key": "plinth_right", "topic": "/ace_camera_plinth_right/.../image/compressed"},
    {"key": "dart_front_right", "topic": "/dart_camera_front_right/.../image/image_raw/compressed"},
    {"key": "zed_rgb", "topic": "/zed/zed_node/rgb/color/rect/image/compressed"}
  ],
  "encoders": {
    "left": "/left_encoder_mm",
    "right": "/right_encoder_mm"
  }
}
```

To add a new camera: add a row to `cameras` array with a unique `key` and the ROS topic name. No code changes needed.

### 2. **Viewer Parameters** (UI, interactive)

Once you open a folder in the viewer:

| Control | Purpose |
|---------|---------|
| **Camera Checkboxes** | Select which cameras to display; grid adapts automatically |
| **Timeline Slider** | Scrub through the entire session; click/drag to jump to any moment |
| **Jump to Timestamp** | Input a nanosecond timestamp directly (from `manifest.json` or sensor logs) |
| **Jump to Encoder Dist** | Input a distance in mm; system finds all moments where encoder ≈ that value (handles reversals) |
| **Play / Pause** | Play at current speed or pause for frame-by-frame inspection |
| **Speed Control** | 0.25x, 0.5x, 1x, 2x, 4x playback speed |
| **Frame Step (per camera)** | While paused, click tile's ◀ or ▶ buttons to step that camera's frames one-by-one |
| **Show Overlays** | Toggle camera name + timestamp + encoder distance text on/off |
| **Click Tile** | Opens full-resolution enlarged view of that camera (stays synced to timeline) |

---

## How to Load & Watch Videos in Parallel

### Typical Workflow:

1. **Extract once per session:**
   ```bash
   cd sync_viewer
   python3 bag_camera_encoder_extractor.py --session /media/.../Day-4
   ```
   Wait ~1–2 hours for full extraction (at 1.0x playback speed). Creates `Day-4/sync_extract/`.

2. **Open viewer:**
   ```bash
   python3 sync_camera_viewer_minimal.py  # or full version once complete
   ```

3. **In the UI:**
   - Click "Open Folder" → select `Day-4/sync_extract/`
   - Check which cameras you want to see (e.g., all 6, or just rail cameras)
   - Use slider to scrub through time — **all selected cameras stay synchronized** (nearest frame within 50ms is shown; blanks if no frame that close)
   - Play for review, pause to step frame-by-frame with ◀/▶ buttons
   - Click a camera tile to see full resolution
   - Jump to a specific encoder distance (e.g., "where did the fault happen?" → input that encoder distance → jumps to all moments matching that distance)

### Key Constraint: Time Synchronization

All 6 cameras are recorded simultaneously with **different frame rates**:
- `rail_left` / `rail_right`: ~30 fps
- `plinth_left` / `plinth_right`: ~24 fps
- `dart_front_right`: ~25 fps
- `zed_rgb`: ~1.1 fps (very sparse — you'll see it blanked most of the time)

The viewer **locks all cameras to a shared timeline** (master clock at 10ms steps). When you scrub, each camera shows its nearest frame within 50ms of the target timestamp. If a camera has no frame that close, its tile goes grey (blanked).

**Result:** You can watch multiple rigs simultaneously, and they stay time-aligned even though they have different frame rates and drop frames independently.

---

## Example: Finding a Fault

1. You suspect a fault occurred around encoder distance 5000mm
2. Click "Jump to encoder dist" → enter `5000`
3. Viewer finds all moments where encoder ≈ 5000mm (handles reversals if the vehicle backed up)
4. Pick one from the list → viewer jumps all cameras to that moment
5. Click "▶ Play" → watch all 6 cameras together
6. Click any tile → enlarge to full resolution for detailed inspection
7. Use ◀/▶ to step frame-by-frame while paused, examining each camera's view of the defect

---

## Implementation Status

| Step | Status | What |
|------|--------|------|
| 1 | ✅ DONE | Extraction script (running on real Day-4 data now) |
| 2 | ✅ DONE | Index lookup module (self-test passing) |
| 3 | ✅ DONE | Minimal single-camera viewer (ready to test) |
| 4 | 📋 NEXT | Full grid + camera selector |
| 5 | 📋 NEXT | Play/pause/speed/jump-to controls |
| 6 | 📋 NEXT | Overlay rendering (camera name, timestamp, distance) |
| 7 | 📋 NEXT | Threading (background decode, prevent UI freeze) |

Extraction currently running — ETA 60+ min at 0.5x playback. Once complete, test Step 3 viewer to validate the pipeline, then expand to Steps 4–7.
