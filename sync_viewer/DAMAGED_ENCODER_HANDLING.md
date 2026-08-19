# Handling Damaged Encoders & Camera Disconnection

## Problem: What If Sensors Fail Mid-Session?

Your rig has two potential failure modes:

### 1. **Camera Disconnection** (Handled ✅)
Some cameras may stop publishing frames partway through the session:
- ZED camera drops out after encoder distance 8500mm
- Dart camera drops out after 4000mm
- Rail and plinth cameras stay live for entire session

**How it's handled:**
- Each camera has its own index CSV with its own timestamp range
- When you scrub to a moment a camera doesn't have data for, the viewer shows a BLANK (grey) tile
- Rail/plinth tiles stay populated, disconnected cameras blank out
- This is automatic — no configuration needed

### 2. **Damaged Encoder** (New Feature ✨)
One of your two encoders might get damaged during the session:
- Left encoder sensors fail → values go to zero or constant
- Right encoder sensors fail → values erratic or invalid

**Solution:** Use the working encoder for both sides

---

## Configuration: Encoder Selection

Edit `camera_topics_config.json`:

```json
{
  "cameras": [...],
  "encoders": {
    "left": "/left_encoder_mm",
    "right": "/right_encoder_mm",
    "primary_source": "average"
  }
}
```

### Options for `primary_source`:

| Value | What It Does | When to Use |
|-------|---|---|
| `"average"` (default) | Average left + right encoder | Both encoders working normally |
| `"left"` | Use **only** left encoder | Right encoder is damaged/failed |
| `"right"` | Use **only** right encoder | Left encoder is damaged/failed |

---

## Example Scenarios

### Scenario 1: Both Encoders Working

```
Configuration:
  "primary_source": "average"

Behavior:
  Overlay shows: "Encoder: 5234.2 mm"
  (average of left + right)
```

### Scenario 2: Right Encoder Failed

Your right encoder stops working at encoder distance 7000mm (stuck at constant value).

```
Configuration:
  "primary_source": "left"

Before fix (if using average):
  0—————2000mm: Encoder: 1234.5 mm (correct, average of both)
  2000—7000mm: Encoder: 1234.5 mm (correct, average of both)
  7000—10000mm: Encoder: 1200.0 mm (WRONG! right is stuck, left still moving)

After fix (using left only):
  0—————10000mm: Encoder: 1234.5 mm (correct throughout)
  (Uses left encoder values directly, right encoder ignored)
```

### Scenario 3: Left Encoder Failed

```
Configuration:
  "primary_source": "right"

Behavior:
  Right encoder values used for all interpolation and display
```

---

## How to Use It

### During Extraction:

No changes needed. The extraction script extracts both encoder topics by default. The selection happens during viewing.

```bash
python3 bag_camera_encoder_extractor.py --session /path/to/session
# Extracts both /left_encoder_mm and /right_encoder_mm
```

### During Viewing:

If you discover an encoder is damaged after extraction:

1. **Update config** (edit `camera_topics_config.json`):
   ```json
   "primary_source": "left"    ← if right damaged
   "primary_source": "right"   ← if left damaged
   ```

2. **Restart viewer** (reopen the extraction folder):
   ```bash
   python3 sync_camera_viewer_minimal.py
   # Click "Open Folder" → Day-4/sync_extract
   ```

3. **Check overlay** — it should now show the selected encoder only:
   ```
   Encoder: 5234.2 mm (left)
   or
   Encoder: 5234.2 mm (right)
   ```

---

## In Code (For Developers)

### Extraction (no changes):
```python
# Both encoders extracted by default
python3 bag_camera_encoder_extractor.py --session /path
# Creates encoder_left.csv and encoder_right.csv
```

### Index Loading (with encoder selection):
```python
from sync_index import load_all

# Load with left encoder only (right damaged)
idx = load_all(
    '/path/to/sync_extract',
    primary_encoder_source='left'  # <-- NEW parameter
)

# Or use right (left damaged)
idx = load_all(
    '/path/to/sync_extract',
    primary_encoder_source='right'
)

# Or average both (default, both working)
idx = load_all(
    '/path/to/sync_extract',
    primary_encoder_source='average'  # default
)
```

### Viewer Integration:
```python
# Load encoder preference from config
config = json.load(open('camera_topics_config.json'))
encoder_source = config.get('encoders', {}).get('primary_source', 'average')

# Pass to index loader
idx = load_all(output_dir, primary_encoder_source=encoder_source)
```

---

## Camera Disconnection: Expected Behavior

When a camera disconnects partway through:

```
Timeline:
0———————————4000mm (dart on)———————8500mm (zed off)———————10000mm

Viewing at encoder distance 5000mm (dart still on):
Rail:   [████ frame ████]  ← Showing frame
Plinth: [████ frame ████]  ← Showing frame
Dart:   [████ frame ████]  ← Showing frame
ZED:    [░░░░░░░░░░░░░░░]  ← Blank (not recorded yet at this distance)

Viewing at encoder distance 9000mm (both dart and zed off):
Rail:   [████ frame ████]  ← Showing frame
Plinth: [████ frame ████]  ← Showing frame
Dart:   [░░░░░░░░░░░░░░░]  ← Blank (disconnected at 4000mm)
ZED:    [░░░░░░░░░░░░░░░]  ← Blank (disconnected at 8500mm)
```

**This is expected and correct behavior** — the blanking prevents misleading stale frames.

---

## Validation Checklist

After extraction, check:

- ✅ `manifest.json` encoder timestamp ranges
- ✅ Encoder CSVs exist: `encoder_left.csv`, `encoder_right.csv`
- ✅ Both CSVs have data
- ✅ Camera indices have different date ranges if cameras disconnected

Example:
```bash
head -5 index/encoder_left.csv
head -5 index/encoder_right.csv

# Should see timestamps in both
# If one is all zeros or constant, that encoder is likely damaged
```

---

## Future Improvement

Currently encoder selection happens at viewer load-time. Future versions could:
- Auto-detect damaged encoders (constant values, NaN, etc.)
- Show warning in UI if an encoder looks bad
- Allow toggling encoder source mid-viewing
- Log which encoder was used in the manifest

