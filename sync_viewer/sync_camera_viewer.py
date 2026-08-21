#!/usr/bin/env python3
"""
Synchronized Multi-Camera Desktop Viewer
==========================================

Scrub through rosbag2 recordings with synchronized camera feeds and encoder data.
Supports flexible camera subset selection, per-camera frame stepping, and jump-to controls.

Usage:
    python3 sync_camera_viewer.py
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple, Dict, List

import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QPushButton, QLabel, QSlider, QComboBox, QCheckBox, QLineEdit, QGridLayout,
    QScrollArea, QDialog, QListWidget, QListWidgetItem, QSpinBox
)
from PySide6.QtCore import Qt, QTimer, QSize, Signal, QObject, QThread
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPainter, QBrush
from PySide6.QtCore import QRect


# ============================================================================
# Sync Index Module (inline for simplicity)
# ============================================================================

class SyncIndex:
    """Loaded index with lookup methods."""

    def __init__(self, output_dir: str, manifest: Dict, camera_keys: List[str]):
        self.output_dir = Path(output_dir)
        self.manifest = manifest
        self.camera_keys = camera_keys

        # Load camera indices
        self.camera_data = {}
        for cam_info in manifest['cameras']:
            key = cam_info['key']
            if key not in camera_keys:
                continue
            csv_path = self.output_dir / 'index' / f"{key}.csv"
            timestamps = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    timestamps.append(int(row['timestamp_ns']))
            self.camera_data[key] = np.array(timestamps, dtype=np.int64)

        # Load encoder data
        self.encoder_left = None
        self.encoder_right = None
        self.encoder_avg = None
        self._load_encoders()

        # Compute global time range
        self.global_min_ts = min(
            [min(ts) for ts in self.camera_data.values()] +
            ([min(self.encoder_left)] if self.encoder_left is not None else []) +
            ([min(self.encoder_right)] if self.encoder_right is not None else [])
        )
        self.global_max_ts = max(
            [max(ts) for ts in self.camera_data.values()] +
            ([max(self.encoder_left)] if self.encoder_left is not None else []) +
            ([max(self.encoder_right)] if self.encoder_right is not None else [])
        )

    def _load_encoders(self):
        """Load encoder CSVs and build averaged series."""
        left_ts, left_vals = self._read_encoder_csv('encoder_left')
        right_ts, right_vals = self._read_encoder_csv('encoder_right')

        if left_ts is not None:
            self.encoder_left = left_ts
            self._encoder_left_vals = left_vals
        if right_ts is not None:
            self.encoder_right = right_ts
            self._encoder_right_vals = right_vals

        # Build average series on union of timestamps
        if left_ts is not None and right_ts is not None:
            union_ts = np.union1d(left_ts, right_ts)
            left_interp = np.interp(union_ts, left_ts, left_vals, left=left_vals[0], right=left_vals[-1])
            right_interp = np.interp(union_ts, right_ts, right_vals, left=right_vals[0], right=right_vals[-1])
            self.encoder_avg = union_ts
            self._encoder_avg_vals = (left_interp + right_interp) / 2.0
        elif left_ts is not None:
            self.encoder_avg = left_ts
            self._encoder_avg_vals = left_vals
        elif right_ts is not None:
            self.encoder_avg = right_ts
            self._encoder_avg_vals = right_vals

    def _read_encoder_csv(self, filename: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Read encoder CSV and return (timestamps, values) or (None, None)."""
        csv_path = self.output_dir / 'index' / f"{filename}.csv"
        if not csv_path.exists():
            return None, None

        timestamps = []
        values = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps.append(int(row['timestamp_ns']))
                values.append(float(row['value_mm']))

        if not timestamps:
            return None, None
        return np.array(timestamps, dtype=np.int64), np.array(values, dtype=np.float32)

    def nearest_frame(self, camera_key: str, ts_ns: int, max_delta_ns: int = 50_000_000) -> Optional[Tuple[str, int]]:
        """Find nearest frame for a camera within max_delta_ns. Returns (filename, actual_ts_ns) or None."""
        if camera_key not in self.camera_data:
            return None

        timestamps = self.camera_data[camera_key]
        idx = np.searchsorted(timestamps, ts_ns)

        candidates = []
        if idx > 0:
            candidates.append(idx - 1)
        if idx < len(timestamps):
            candidates.append(idx)

        best_idx = None
        best_delta = float('inf')
        for i in candidates:
            delta = abs(timestamps[i] - ts_ns)
            if delta < best_delta:
                best_delta = delta
                best_idx = i

        if best_idx is not None and best_delta <= max_delta_ns:
            actual_ts = int(timestamps[best_idx])
            filename = f"{actual_ts}.jpg"
            return filename, actual_ts

        return None

    def interpolate_encoder_distance_mm(self, ts_ns: int) -> Optional[float]:
        """Interpolate encoder distance at timestamp. Returns float or None."""
        if self.encoder_avg is None:
            return None

        if ts_ns < self.encoder_avg[0] or ts_ns > self.encoder_avg[-1]:
            return None

        return float(np.interp(ts_ns, self.encoder_avg, self._encoder_avg_vals))

    def next_frame_ts(self, camera_key: str, current_ts: int) -> Optional[int]:
        """Get next frame timestamp for a camera."""
        if camera_key not in self.camera_data:
            return None
        timestamps = self.camera_data[camera_key]
        idx = np.searchsorted(timestamps, current_ts)
        if idx < len(timestamps) - 1:
            return int(timestamps[idx + 1])
        return None

    def prev_frame_ts(self, camera_key: str, current_ts: int) -> Optional[int]:
        """Get previous frame timestamp for a camera."""
        if camera_key not in self.camera_data:
            return None
        timestamps = self.camera_data[camera_key]
        idx = np.searchsorted(timestamps, current_ts, side='right')
        if idx > 1:
            return int(timestamps[idx - 2])
        return None

    def find_encoder_matches(self, target_value: float, tolerance: float = 10.0) -> List[int]:
        """Find all timestamps where encoder ≈ target_value."""
        if self.encoder_avg is None:
            return []

        mask = np.abs(self._encoder_avg_vals - target_value) <= tolerance
        return [int(ts) for ts in self.encoder_avg[mask]]


# ============================================================================
# Camera Tile Widget
# ============================================================================

class CameraTile(QWidget):
    """Single camera tile with frame display, controls, and resizable corners."""

    step_clicked = Signal(str, str)  # camera_key, direction ('prev' or 'next')

    def __init__(self, camera_key: str, index: SyncIndex, frames_dir: Path):
        super().__init__()
        self.camera_key = camera_key
        self.index = index
        self.frames_dir = frames_dir
        self.current_pixmap = None
        self.current_ts = None
        self.blank = False
        self.original_size = None

        # Default size (resizable)
        self.tile_width = 450
        self.tile_height = 320
        self.setMinimumSize(300, 250)
        self.setMaximumSize(1200, 900)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)

        # Frame display area
        self.frame_label = QLabel()
        self.frame_label.setStyleSheet("border: 1px solid #444; background-color: #222;")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setScaledContents(False)
        layout.addWidget(self.frame_label, 1)

        # Camera name label
        name_label = QLabel(self.camera_key)
        name_label.setStyleSheet("font-weight: bold; color: #aaa; font-size: 11px;")
        name_label.setMaximumHeight(18)
        layout.addWidget(name_label)

        # Step buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(3)
        self.prev_btn = QPushButton("◀ Prev")
        self.next_btn = QPushButton("Next ▶")
        self.prev_btn.setMaximumHeight(22)
        self.next_btn.setMaximumHeight(22)
        self.prev_btn.setStyleSheet("font-size: 10px;")
        self.next_btn.setStyleSheet("font-size: 10px;")
        self.prev_btn.clicked.connect(lambda: self.step_clicked.emit(self.camera_key, 'prev'))
        self.next_btn.clicked.connect(lambda: self.step_clicked.emit(self.camera_key, 'next'))
        btn_layout.addWidget(self.prev_btn)
        btn_layout.addWidget(self.next_btn)
        layout.addLayout(btn_layout)

        # Resize tracking
        self.resizing = False
        self.resize_start_pos = None
        self.setCursor(Qt.ArrowCursor)

    def set_frame(self, ts_ns: Optional[int], blank: bool = False):
        """Load and display frame scaled to tile size."""
        self.current_ts = ts_ns
        self.blank = blank

        if blank or ts_ns is None:
            self.current_pixmap = None
            self._draw_blank()
            return

        # Load frame
        frame_result = self.index.nearest_frame(self.camera_key, ts_ns)
        if frame_result is None:
            self.current_pixmap = None
            self._draw_blank()
            return

        filename, actual_ts = frame_result
        frame_path = self.frames_dir / self.camera_key / filename

        if not frame_path.exists():
            self.current_pixmap = None
            self._draw_blank()
            return

        # Decode JPEG
        data = cv2.imread(str(frame_path))
        if data is None:
            self.current_pixmap = None
            self._draw_blank()
            return

        # Convert BGR to RGB and create QPixmap
        rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.original_size = (w, h)
        bytes_per_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.current_pixmap = QPixmap.fromImage(qt_img)

        # Scale to fit tile (maintain aspect ratio)
        frame_display_height = self.height() - 70  # Account for label and buttons
        frame_display_width = self.width() - 10
        scaled = self.current_pixmap.scaledToWidth(
            max(100, frame_display_width), Qt.SmoothTransformation
        )
        self.frame_label.setPixmap(scaled)

    def _draw_blank(self):
        """Draw a grey blank tile."""
        blank_pixmap = QPixmap(self.width() - 10, max(100, self.height() - 70))
        blank_pixmap.fill(QColor(50, 50, 50))
        self.frame_label.setPixmap(blank_pixmap)

    def mousePressEvent(self, event):
        """Detect if user clicks on resize corner."""
        corner_size = 20
        if (event.pos().x() > self.width() - corner_size and
            event.pos().y() > self.height() - corner_size):
            self.resizing = True
            self.resize_start_pos = event.pos()
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle tile resizing."""
        corner_size = 20
        if (event.pos().x() > self.width() - corner_size and
            event.pos().y() > self.height() - corner_size):
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

        if self.resizing and self.resize_start_pos:
            delta = event.pos() - self.resize_start_pos
            new_width = max(300, self.width() + delta.x())
            new_height = max(250, self.height() + delta.y())
            self.resize(new_width, new_height)
            self.resize_start_pos = event.pos()

    def mouseReleaseEvent(self, event):
        """End resizing."""
        self.resizing = False
        self.resize_start_pos = None
        self.setCursor(Qt.ArrowCursor)

    def paintEvent(self, event):
        """Draw resize handle indicator."""
        super().paintEvent(event)
        painter = QPainter(self)
        corner_size = 15
        painter.setOpacity(0.5)
        painter.fillRect(
            self.width() - corner_size, self.height() - corner_size,
            corner_size, corner_size,
            QColor(100, 150, 200)
        )
        painter.end()


# ============================================================================
# Main Viewer Window
# ============================================================================

class SyncCameraViewer(QMainWindow):
    """Main multi-camera viewer window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Synchronized Multi-Camera Viewer")
        self.setGeometry(100, 100, 1400, 900)

        self.index = None
        self.output_dir = None
        self.frames_dir = None
        self.current_master_ts = None
        self.is_playing = False
        self.playback_speed = 1.0
        self.selected_cameras = set()
        self.camera_tiles = {}

        self.STEP_NS = 10_000_000  # 10ms master timeline step

        self.setup_ui()
        self.setup_playback_timer()

    def setup_ui(self):
        """Build UI layout."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Toolbar
        toolbar_layout = QHBoxLayout()

        open_btn = QPushButton("Open Folder")
        open_btn.clicked.connect(self.open_folder)
        toolbar_layout.addWidget(open_btn)

        toolbar_layout.addSpacing(20)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_playback)
        toolbar_layout.addWidget(self.play_btn)

        toolbar_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x", "4.0x"])
        self.speed_combo.setCurrentIndex(2)
        self.speed_combo.currentTextChanged.connect(self.on_speed_changed)
        toolbar_layout.addWidget(self.speed_combo)

        toolbar_layout.addStretch()
        main_layout.addLayout(toolbar_layout)

        # Main content: splitter of camera selector + grid
        splitter = QSplitter(Qt.Horizontal)

        # Camera selector panel
        selector_widget = QWidget()
        selector_layout = QVBoxLayout(selector_widget)
        selector_layout.addWidget(QLabel("Cameras:"))

        self.camera_checks = {}
        self.select_all_btn = QPushButton("All")
        self.select_all_btn.clicked.connect(self.select_all_cameras)
        selector_layout.addWidget(self.select_all_btn)

        self.select_none_btn = QPushButton("None")
        self.select_none_btn.clicked.connect(self.select_no_cameras)
        selector_layout.addWidget(self.select_none_btn)

        self.camera_scroll = QWidget()
        self.camera_scroll_layout = QVBoxLayout(self.camera_scroll)
        scroll_area = QScrollArea()
        scroll_area.setWidget(self.camera_scroll)
        scroll_area.setWidgetResizable(True)
        selector_layout.addWidget(scroll_area)

        splitter.addWidget(selector_widget)

        # Grid area (scrollable)
        grid_widget = QWidget()
        self.grid_layout = QGridLayout(grid_widget)
        self.grid_layout.setSpacing(10)
        grid_scroll = QScrollArea()
        grid_scroll.setWidget(grid_widget)
        grid_scroll.setWidgetResizable(True)
        splitter.addWidget(grid_scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        main_layout.addWidget(splitter, 1)  # Give maximum stretch to camera grid

        # Timeline controls (compact)
        timeline_layout = QHBoxLayout()
        timeline_layout.setContentsMargins(0, 5, 0, 5)
        timeline_layout.setSpacing(10)

        self.time_label = QLabel("No data loaded")
        self.time_label.setMaximumWidth(250)
        timeline_layout.addWidget(self.time_label)

        self.encoder_label = QLabel()
        self.encoder_label.setMaximumWidth(200)
        timeline_layout.addWidget(self.encoder_label)

        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.setMinimum(0)
        timeline_layout.addWidget(self.slider)

        main_layout.addLayout(timeline_layout)

        # Jump controls (compact, single row)
        jump_layout = QHBoxLayout()
        jump_layout.setContentsMargins(0, 0, 0, 5)
        jump_layout.setSpacing(5)

        jump_layout.addWidget(QLabel("Time (ns):"))
        self.ts_input = QLineEdit()
        self.ts_input.setMaximumWidth(180)
        self.ts_input.setMaximumHeight(24)
        jump_layout.addWidget(self.ts_input)
        jump_ts_btn = QPushButton("Go")
        jump_ts_btn.setMaximumWidth(40)
        jump_ts_btn.setMaximumHeight(24)
        jump_ts_btn.clicked.connect(self.jump_to_timestamp)
        jump_layout.addWidget(jump_ts_btn)

        jump_layout.addSpacing(15)

        jump_layout.addWidget(QLabel("Encoder (mm):"))
        self.encoder_input = QLineEdit()
        self.encoder_input.setMaximumWidth(120)
        self.encoder_input.setMaximumHeight(24)
        jump_layout.addWidget(self.encoder_input)
        jump_encoder_btn = QPushButton("Go")
        jump_encoder_btn.setMaximumWidth(40)
        jump_encoder_btn.setMaximumHeight(24)
        jump_encoder_btn.clicked.connect(self.jump_to_encoder)
        jump_layout.addWidget(jump_encoder_btn)

        jump_layout.addStretch()
        main_layout.addLayout(jump_layout)

    def setup_playback_timer(self):
        """Setup playback timer."""
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.on_playback_tick)

    def open_folder(self):
        """Open folder picker and load extraction."""
        from PySide6.QtWidgets import QFileDialog

        folder = QFileDialog.getExistingDirectory(self, "Select sync_extract folder")
        if not folder:
            return

        self.load_extraction(folder)

    def load_extraction(self, folder_path: str):
        """Load extraction data from folder."""
        try:
            output_dir = Path(folder_path)
            manifest_path = output_dir / 'manifest.json'

            if not manifest_path.exists():
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Error", "No manifest.json found in folder")
                return

            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            # Get all camera keys
            all_cameras = [cam['key'] for cam in manifest['cameras']]

            # Create index
            self.index = SyncIndex(folder_path, manifest, all_cameras)
            self.output_dir = output_dir
            self.frames_dir = output_dir / 'frames'

            # Initialize selected cameras (all by default)
            self.selected_cameras = set(all_cameras)

            # Build camera checkboxes
            self.camera_checks.clear()
            while self.camera_scroll_layout.count():
                self.camera_scroll_layout.takeAt(0).widget().deleteLater()

            for cam_key in all_cameras:
                check = QCheckBox(cam_key)
                check.setChecked(True)
                check.stateChanged.connect(lambda state, k=cam_key: self.on_camera_selection_changed(k, state))
                self.camera_checks[cam_key] = check
                self.camera_scroll_layout.addWidget(check)

            self.camera_scroll_layout.addStretch()

            # Initialize slider
            self.current_master_ts = self.index.global_min_ts
            max_steps = int((self.index.global_max_ts - self.index.global_min_ts) / self.STEP_NS)
            self.slider.setMaximum(max_steps)
            self.slider.setValue(0)

            # Build grid
            self.rebuild_grid()

            # Display first frame
            self.update_display()

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def on_camera_selection_changed(self, camera_key: str, state):
        """Handle camera checkbox changes."""
        if state:
            self.selected_cameras.add(camera_key)
        else:
            self.selected_cameras.discard(camera_key)
        self.rebuild_grid()
        self.update_display()

    def select_all_cameras(self):
        """Select all cameras."""
        for check in self.camera_checks.values():
            check.setChecked(True)

    def select_no_cameras(self):
        """Deselect all cameras."""
        for check in self.camera_checks.values():
            check.setChecked(False)

    def rebuild_grid(self):
        """Rebuild camera grid based on selection."""
        # Clear grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self.camera_tiles.clear()

        if not self.selected_cameras or self.index is None:
            return

        # Layout: ceil(sqrt(N)) columns
        n = len(self.selected_cameras)
        cols = max(1, int(np.ceil(np.sqrt(n))))

        for i, cam_key in enumerate(sorted(self.selected_cameras)):
            if self.index is None:
                continue

            tile = CameraTile(cam_key, self.index, self.frames_dir)
            tile.step_clicked.connect(self.on_camera_step)
            self.camera_tiles[cam_key] = tile

            row = i // cols
            col = i % cols
            self.grid_layout.addWidget(tile, row, col)

    def on_camera_step(self, camera_key: str, direction: str):
        """Handle per-camera frame stepping."""
        if self.current_master_ts is None or self.index is None:
            return

        if direction == 'next':
            next_ts = self.index.next_frame_ts(camera_key, self.current_master_ts)
            if next_ts is not None:
                self.set_master_time(next_ts)
        elif direction == 'prev':
            prev_ts = self.index.prev_frame_ts(camera_key, self.current_master_ts)
            if prev_ts is not None:
                self.set_master_time(prev_ts)

    def on_slider_moved(self, value):
        """Handle slider movement."""
        if self.index is None:
            return
        ts = self.index.global_min_ts + value * self.STEP_NS
        self.set_master_time(int(ts))

    def set_master_time(self, ts_ns: int):
        """Set master timeline to timestamp."""
        if self.index is None:
            return

        # Clamp to valid range
        ts_ns = max(self.index.global_min_ts, min(ts_ns, self.index.global_max_ts))

        self.current_master_ts = ts_ns

        # Update slider
        steps = int((ts_ns - self.index.global_min_ts) / self.STEP_NS)
        self.slider.blockSignals(True)
        self.slider.setValue(steps)
        self.slider.blockSignals(False)

        self.update_display()

    def update_display(self):
        """Update all camera tiles and time labels."""
        if self.current_master_ts is None or self.index is None:
            return

        # Update time label
        dt = datetime.fromtimestamp(self.current_master_ts / 1e9, tz=timezone.utc).astimezone()
        self.time_label.setText(f"Time: {dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")

        # Update encoder label
        encoder_dist = self.index.interpolate_encoder_distance_mm(self.current_master_ts)
        if encoder_dist is not None:
            self.encoder_label.setText(f"Encoder Dist: {encoder_dist:.1f} mm")
        else:
            self.encoder_label.setText("Encoder Dist: N/A")

        # Update tiles
        for cam_key, tile in self.camera_tiles.items():
            frame_result = self.index.nearest_frame(cam_key, self.current_master_ts)
            blank = frame_result is None

            if not blank:
                tile.set_frame(self.current_master_ts, blank=False)
            else:
                tile.set_frame(None, blank=True)

    def toggle_playback(self):
        """Toggle play/pause."""
        self.is_playing = not self.is_playing
        self.play_btn.setText("⏸ Pause" if self.is_playing else "▶ Play")

        if self.is_playing:
            self.playback_timer.start(33)  # ~30 FPS
        else:
            self.playback_timer.stop()

    def on_playback_tick(self):
        """Playback timer tick."""
        if self.current_master_ts is None or self.index is None:
            return

        # Advance time
        delta_ns = int(self.STEP_NS * self.playback_speed)
        new_ts = self.current_master_ts + delta_ns

        if new_ts > self.index.global_max_ts:
            # Loop or stop
            self.is_playing = False
            self.play_btn.setText("▶ Play")
            self.playback_timer.stop()
            self.set_master_time(self.index.global_min_ts)
        else:
            self.set_master_time(new_ts)

    def on_speed_changed(self, text: str):
        """Handle speed combo change."""
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1.0x": 1.0, "2.0x": 2.0, "4.0x": 4.0}
        self.playback_speed = speed_map.get(text, 1.0)

    def jump_to_timestamp(self):
        """Jump to specified timestamp."""
        try:
            ts_ns = int(self.ts_input.text())
            self.set_master_time(ts_ns)
        except:
            pass

    def jump_to_encoder(self):
        """Jump to specified encoder distance."""
        if self.index is None:
            return

        try:
            target_dist = float(self.encoder_input.text())
            matches = self.index.find_encoder_matches(target_dist, tolerance=10.0)

            if not matches:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(self, "Not Found", f"No encoder value near {target_dist} mm")
                return

            if len(matches) == 1:
                self.set_master_time(matches[0])
            else:
                # Show picker dialog
                self.show_encoder_picker(matches)
        except:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Invalid Input", "Enter a valid encoder distance (mm)")

    def show_encoder_picker(self, timestamps: List[int]):
        """Show dialog to pick from multiple encoder matches."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Multiple Matches Found")
        dialog.setGeometry(400, 300, 400, 300)

        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel(f"Found {len(timestamps)} matches. Select one:"))

        list_widget = QListWidget()
        for ts in timestamps:
            dt = datetime.fromtimestamp(ts / 1e9, tz=timezone.utc).astimezone()
            item_text = dt.strftime("%Y-%m-%d %H:%M:%S")
            list_widget.addItem(QListWidgetItem(item_text))

        def on_item_selected():
            idx = list_widget.currentRow()
            if idx >= 0:
                self.set_master_time(timestamps[idx])
                dialog.accept()

        list_widget.itemDoubleClicked.connect(on_item_selected)
        layout.addWidget(list_widget)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("Select")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(on_item_selected)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        dialog.exec()


def main():
    app = QApplication(sys.argv)
    viewer = SyncCameraViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
