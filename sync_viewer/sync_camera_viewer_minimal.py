#!/usr/bin/env python3
"""
Minimal Synchronized Camera Viewer
===================================

Step 3: Single-camera viewer for testing extraction -> index -> display pipeline.
Synchronous decode (no threading), minimal UI (one tile + slider).

Usage:
    python3 sync_camera_viewer_minimal.py
    Then use "Open Folder" button to select a sync_extract directory.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
    QPushButton, QLabel, QSlider, QFileDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont

from sync_index import load_all, nearest_frame, interpolate_encoder_distance_mm, get_master_timeline_range


IST_OFFSET = timedelta(hours=5, minutes=30)


class MinimalCameraViewer(QMainWindow):
    """Minimal viewer: one camera, one slider, synchronous decode."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sync Camera Viewer — Minimal (Step 3)")
        self.setGeometry(100, 100, 1200, 800)

        self.sync_index = None
        self.camera_key = 'rail_left'  # Hardcoded for now
        self.current_t_ns = None
        self.playback_speed = 1.0
        self.is_playing = False
        self.playback_timer = None

        # UI
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Toolbar
        toolbar_layout = QHBoxLayout()
        self.open_btn = QPushButton("Open Folder...")
        self.open_btn.clicked.connect(self.on_open_folder)
        toolbar_layout.addWidget(self.open_btn)

        self.play_pause_btn = QPushButton("Play")
        self.play_pause_btn.clicked.connect(self.on_play_pause)
        self.play_pause_btn.setEnabled(False)
        toolbar_layout.addWidget(self.play_pause_btn)

        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        # Camera display (pixel label)
        self.camera_label = QLabel()
        self.camera_label.setMinimumHeight(400)
        self.camera_label.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        layout.addWidget(self.camera_label)

        # Timeline slider
        slider_layout = QHBoxLayout()
        self.time_label = QLabel("Not loaded")
        slider_layout.addWidget(self.time_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.setEnabled(False)
        slider_layout.addWidget(self.slider)

        layout.addLayout(slider_layout)

        # Status bar
        self.status_label = QLabel("Ready. Click 'Open Folder' to load an extraction.")
        layout.addWidget(self.status_label)

    def on_open_folder(self):
        """Load a sync_extract directory."""
        folder = QFileDialog.getExistingDirectory(self, "Select sync_extract folder")
        if not folder:
            return

        try:
            self.sync_index = load_all(Path(folder))
            self.current_t_ns = self.sync_index.global_min_ts

            # Configure slider
            min_ts, max_ts, step_ns, num_steps = get_master_timeline_range(self.sync_index)
            self.slider.setMaximum(num_steps)
            self.slider.setValue(0)
            self.slider.setEnabled(True)

            self.play_pause_btn.setEnabled(True)
            self.status_label.setText(f"Loaded: {Path(folder).name}")

            # Display initial frame
            self.update_display()

        except Exception as e:
            self.status_label.setText(f"Error: {e}")

    def on_slider_moved(self, pos):
        """Slider moved by user."""
        if not self.sync_index:
            return
        from sync_index import slider_pos_to_ts
        self.current_t_ns = slider_pos_to_ts(self.sync_index, pos)
        self.update_display()

    def on_play_pause(self):
        """Toggle playback."""
        if not self.sync_index:
            return

        self.is_playing = not self.is_playing
        self.play_pause_btn.setText("Pause" if self.is_playing else "Play")

        if self.is_playing:
            if self.playback_timer is None:
                self.playback_timer = QTimer()
                self.playback_timer.timeout.connect(self.on_playback_tick)
            self.playback_timer.start(33)  # ~30 fps
        else:
            if self.playback_timer:
                self.playback_timer.stop()

    def on_playback_tick(self):
        """Advance timeline during playback."""
        if not self.sync_index:
            return

        from sync_index import slider_pos_to_ts, ts_to_slider_pos, STEP_NS
        step = int(STEP_NS * self.playback_speed)
        self.current_t_ns += step

        # Wrap or stop at end
        if self.current_t_ns > self.sync_index.global_max_ts:
            self.current_t_ns = self.sync_index.global_max_ts
            self.is_playing = False
            self.play_pause_btn.setText("Play")
            if self.playback_timer:
                self.playback_timer.stop()

        # Update slider without triggering moved signal
        from sync_index import ts_to_slider_pos
        pos = ts_to_slider_pos(self.sync_index, self.current_t_ns)
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)

        self.update_display()

    def update_display(self):
        """Update the camera frame display."""
        if not self.sync_index or not self.current_t_ns:
            return

        # Get nearest frame for this camera
        cam = self.sync_index.cameras.get(self.camera_key)
        if not cam:
            self.status_label.setText(f"Camera {self.camera_key} not in index")
            return

        result = nearest_frame(cam, self.current_t_ns)
        if result is None:
            # Blanked
            pixmap = QPixmap(400, 300)
            pixmap.fill(Qt.gray)
            self.camera_label.setPixmap(pixmap)
            self.update_time_label("No frame (>50ms away)")
            return

        filepath, actual_ts_ns = result

        # Decode frame
        try:
            frame_data = np.fromfile(filepath, dtype=np.uint8)
            bgr = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            if bgr is None:
                self.status_label.setText(f"Failed to decode {Path(filepath).name}")
                return

            # Resize for display (keep aspect)
            h, w = bgr.shape[:2]
            max_w, max_h = 400, 400
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            bgr_resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Convert BGR -> RGB for Qt
            rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qt_img = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qt_img)
            self.camera_label.setPixmap(pixmap)

            # Update time label
            self.update_time_label(actual_ts_ns)

        except Exception as e:
            self.status_label.setText(f"Error decoding frame: {e}")

    def update_time_label(self, ts_ns_or_msg):
        """Update the time display label."""
        if isinstance(ts_ns_or_msg, str):
            self.time_label.setText(ts_ns_or_msg)
        else:
            ts_ns = ts_ns_or_msg
            # Convert to human-readable time
            ts_sec = ts_ns // 1_000_000_000
            ts_us = (ts_ns % 1_000_000_000) // 1_000

            # UTC to IST
            dt_utc = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
            dt_ist = dt_utc.astimezone(timezone(IST_OFFSET))
            time_str = dt_ist.strftime('%Y-%m-%d %H:%M:%S')
            us_str = f"{ts_us:06d}"

            # Encoder distance if available
            dist_mm = interpolate_encoder_distance_mm(self.sync_index, ts_ns)
            if dist_mm is not None:
                dist_str = f"{dist_mm:.1f}mm"
            else:
                dist_str = "N/A"

            label_text = f"{time_str}.{us_str} IST | Encoder: {dist_str}"
            self.time_label.setText(label_text)


def main():
    app = QApplication(sys.argv)
    viewer = MinimalCameraViewer()
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
