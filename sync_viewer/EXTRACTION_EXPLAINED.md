# Step 1: Extraction — What Happens to Your 3 Bag Folders

## The Problem You Have

Your Day-4 session is recorded into **3 separate bag folders**, recorded in parallel:

```
Day-4/
├── day4_dhristi_railhead_cameras_20260707_022848/        ← Bag 1
│   ├── metadata.yaml
│   ├── day4_dhristi_railhead_cameras_20260707_022848_0.mcap
│   └── ... (more .mcap files)
│
├── day4_dhristi_plinth_cameras_20260707_022848/          ← Bag 2
│   ├── metadata.yaml
│   ├── day4_dhristi_plinth_cameras_20260707_022848_0.mcap
│   └── ... (more .mcap files)
│
└── day4_dhristi_track_measurement_20260707_022848/       ← Bag 3
    ├── metadata.yaml
    ├── day4_dhristi_track_measurement_20260707_022848_0.mcap
    └── ... (more .mcap files)
```

Each bag records a different camera rig **simultaneously** (same start/end times), so they share the same time axis but contain different topics:

```
Bag 1 (railhead)          Bag 2 (plinth)           Bag 3 (track_measurement)
─────────────────────────────────────────────────────────────────────────────
/ace_camera_rail_left     /ace_camera_plinth_left  /dart_camera_front_right
/ace_camera_rail_right    /ace_camera_plinth_right /zed/zed_node/rgb/...
/left_encoder_mm          /left_encoder_mm         /left_encoder_mm
/right_encoder_mm         /right_encoder_mm        /right_encoder_mm
/zed/.../pose             /zed/.../pose            /zed/.../odom
                                                   /left_wheel_encoder
                                                   /right_wheel_encoder
```

**Key issue:** Each bag has encoder topics `/left_encoder_mm` and `/right_encoder_mm` — if you naively play all 3 bags at once, you'd get 3x the encoder data (triplicate).

---

## The Solution: Intelligent Multi-Bag Playback

Here's what `bag_camera_encoder_extractor.py` does:

### Step 1a: Bag Discovery

```python
discover_session("Day-4/")
```

Scans all sub-folders, reads each `metadata.yaml`, and builds a map:

```
Topic Map:
  /ace_camera_rail_left        → Bag 1 (railhead)       ✓
  /ace_camera_rail_right       → Bag 1 (railhead)       ✓
  /ace_camera_plinth_left      → Bag 2 (plinth)         ✓
  /ace_camera_plinth_right     → Bag 2 (plinth)         ✓
  /dart_camera_front_right     → Bag 3 (track)          ✓
  /zed/zed_node/rgb/...        → Bag 3 (track)          ✓
  /left_encoder_mm             → Bag 1, Bag 2, Bag 3    ⚠️ (3x!)
  /right_encoder_mm            → Bag 1, Bag 2, Bag 3    ⚠️ (3x!)
```

### Step 1b: Encoder Source Selection

Pick exactly ONE bag as the encoder source:

```
You request cameras: rail_left, rail_right, plinth_left, dart_front_right, zed_rgb

Bags needed for cameras:
  - Bag 1 (has rail_left, rail_right)
  - Bag 2 (has plinth_left)
  - Bag 3 (has dart_front_right, zed_rgb)

Encoder source = Bag 1 (railhead)
  ✓ Because it's needed for cameras (no extra process)
  ✓ AND it has the encoder topics
```

### Step 1c: Parallel Playback (The Key Trick)

**Start 3 concurrent `ros2 bag play` processes, one per bag folder:**

```bash
# Process 1: Play Bag 1 (railhead)
# WITH --clock 200 (publishes /clock using original recording timestamps)
ros2 bag play Day-4/railhead_cameras_.../ \
  --topics \
    /ace_camera_rail_left \
    /ace_camera_rail_right \
    /left_encoder_mm \
    /right_encoder_mm \
  --clock 200 \
  --rate 0.5

# Process 2: Play Bag 2 (plinth)
# WITHOUT --clock (doesn't publish /clock, just plays its topics)
ros2 bag play Day-4/plinth_cameras_.../ \
  --topics \
    /ace_camera_plinth_left \
    /ace_camera_plinth_right \
  --rate 0.5

# Process 3: Play Bag 3 (track)
# WITHOUT --clock
ros2 bag play Day-4/track_measurement_.../ \
  --topics \
    /dart_camera_front_right \
    /zed/zed_node/rgb/color/rect/image/compressed \
  --rate 0.5
```

**Why this works:**

1. **Process 1 publishes `/clock`** using the bag's original timestamps — this is the master time source
2. **Processes 2 & 3** play their topics independently, but since they started together (back-to-back, <1sec apart) and record nearly identical content (same session, same hardware clocks), they stay synchronized naturally
3. **One ROS2 Node** (the extractor) subscribes to ALL topics from ALL 3 processes simultaneously
4. **Each message carries its original `header.stamp`** (for camera images) or gets timestamped via `get_clock()` (for encoder samples that have no header)

### Step 1d: Inside the Node: How Timestamps Align

```python
class MultiCameraEncoderCapture(Node):
    def __init__(self):
        # Set use_sim_time=True so get_clock() uses /clock (Process 1's clock)
        super().__init__("extractor", parameter_overrides=[Parameter("use_sim_time", ..., True)])
```

**For camera messages:**
```python
def _camera_cb(self, key, msg):
    ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
    # Use the RECORDED timestamp from the bag's original recording
```

**For encoder messages (no header, so use the node's clock):**
```python
def _encoder_cb(self, side, msg):
    ts_ns = self.get_clock().now().nanoseconds  
    # Because use_sim_time=True, this returns /clock time (Process 1's original timestamps)
    # So it's on the SAME TIME AXIS as the camera messages
```

**Result:** Encoder samples are timestamped on the same axis as camera frames, so interpolation later works perfectly.

---

## Data Flow Diagram

```
┌─────────────────────────────────────┐
│ 3 Bag Folders                       │
│ (separate recordings, same session) │
└───────┬──────────┬──────────────────┘
        │          │
   ┌────▼──────┐   │   ┌──────────────────────┐
   │Bag 1:     │   │   │Bag 2: plinth         │
   │railhead   │   │   │ - plinth cameras     │
   │ - rail    │   │   │ - encoder (dup)      │
   │   cameras │   │   │ - pose               │
   │ - encoder │   │   └──────────────────────┘
   │ - pose    │   │
   └────┬──────┘   │   ┌──────────────────────┐
        │          └──▶│Bag 3: track_meas.    │
        │              │ - dart camera        │
        │              │ - zed rgb            │
        │              │ - encoder (dup)      │
        │              │ - lidar, IMU, etc    │
        │              └──────────────────────┘
        │
        ▼
  ┌──────────────────────────────────────────────────┐
  │ ros2 bag play (3 concurrent processes)           │
  │                                                  │
  │ [Process 1] --clock 200 ──┐ Master time source  │
  │ [Process 2]              ├─▶ /clock topic       │
  │ [Process 3]              │   (original recorcd  │
  │                          │    timestamps)       │
  └────────────┬─────────────┴──────────────────────┘
               │
        ┌──────▼──────────────────────────┐
        │ One ROS2 Node (extractor)       │
        │ Subscribes to:                  │
        │  - /ace_camera_rail_left        │
        │  - /ace_camera_rail_right       │
        │  - /ace_camera_plinth_left      │
        │  - /ace_camera_plinth_right     │
        │  - /dart_camera_front_right     │
        │  - /zed/zed_node/rgb/...        │
        │  - /left_encoder_mm             │
        │  - /right_encoder_mm            │
        └──────┬───────────────────────────┘
               │
     ┌─────────▼─────────────────────┐
     │ Extract & Write               │
     │                               │
     │ For each camera:              │
     │  - Decode JPEG in msg.data    │
     │  - Write bytes to disk        │
     │  - Record (timestamp, path)   │
     │                               │
     │ For each encoder:             │
     │  - Record (timestamp, value)  │
     │                               │
     │ All in INDEX/ CSVs            │
     └─────────┬─────────────────────┘
               │
     ┌─────────▼──────────────────────────────┐
     │ Output: Day-4/sync_extract/            │
     │                                        │
     │ frames/                                │
     │  ├── rail_left/                        │
     │  ├── rail_right/                       │
     │  ├── plinth_left/                      │
     │  ├── plinth_right/                     │
     │  ├── dart_front_right/                 │
     │  └── zed_rgb/                          │
     │                                        │
     │ index/                                 │
     │  ├── rail_left.csv                     │
     │  ├── ... (5 more cameras)              │
     │  ├── encoder_left.csv                  │
     │  └── encoder_right.csv                 │
     │                                        │
     │ manifest.json                          │
     └────────────────────────────────────────┘
```

---

## The Critical Technical Fix: `--clock` + `use_sim_time`

**Problem:** Encoder messages (`std_msgs/Float32`) have NO header, so they arrive with wall-clock timestamps (whenever `ros2 bag play` got around to publishing them). This is a DIFFERENT time axis from camera `header.stamp` values (recorded in the original session).

**Solution:**

1. **Process 1 publishes `/clock`** with original timestamps: `ros2 bag play ... --clock 200`
2. **Extractor sets `use_sim_time=True`**: Makes `self.get_clock().now()` return `/clock` time
3. **Result:** Encoder callbacks get timestamps on the original recording axis, same as camera frames

**Validation:** Check manifest.json's encoder and camera time ranges — they should overlap within ~1 second.

---

## What If You Want Fewer Cameras?

```bash
python3 bag_camera_encoder_extractor.py \
  --session Day-4 \
  --cameras rail_left,zed_rgb
```

Extractor will:
1. Discover Bag 1 has `rail_left`, Bag 3 has `zed_rgb`
2. Play ONLY those bags (Bag 2 skipped entirely, no wasted resources)
3. Still play Bag 1 with `--clock` (encoder source)
4. Extract only 2 camera folders + 2 encoder CSVs
5. Output is much smaller

---

## Performance Notes

- **Extraction speed:** Real-time (1.0x) takes ~58 minutes (bag duration)
- **Can be faster:** Use `--rate 2.0` for 2x speedup (58 min → 29 min), trade-off: may drop a few frames under network load
- **Can be slower:** Use `--rate 0.5` for 0.5x (more stable, 116 min total) — useful if your system is busy
- **Output size:** ~8–20 GB per session (depends on number of cameras extracted)
  - 104k rail frames × 150KB/frame ≈ 15GB per camera
  - Encoder CSVs are tiny (<100MB each)

---

## When Extraction Is Done

Check the output:

```bash
ls -lh Day-4/sync_extract/
cat Day-4/sync_extract/manifest.json | jq .
```

Then open the viewer:

```bash
cd sync_viewer
python3 sync_camera_viewer_minimal.py
# Click "Open Folder" → Day-4/sync_extract/
```

All 6 cameras load, synchronized to the same timeline, with encoder distance interpolated at each moment.
