"""
Oculomotor Assessment System - Three Visual Tasks
Measures fixation stability, saccade speed, and smooth pursuit accuracy
Designed for Mac embedded camera in a virtual environment
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import math


class EyeTracker:
    """Eye tracking using MediaPipe Face Mesh for landmark detection"""

    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Eye landmark indices for iris tracking
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [33, 133, 160, 159, 158, 157, 173]
        self.RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398]

    def get_gaze_position(self, frame):
        """
        Returns normalized gaze position (X, Y) in screen coordinates and quality score
        X, Y are in range [0, 1] where (0, 0) is top-left, (1, 1) is bottom-right
        Quality score ranges from 0.0 to 1.0
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None, 0.0

        landmarks = results.multi_face_landmarks[0].landmark

        # Get iris centers
        left_iris_x = np.mean([landmarks[i].x for i in self.LEFT_IRIS])
        left_iris_y = np.mean([landmarks[i].y for i in self.LEFT_IRIS])
        right_iris_x = np.mean([landmarks[i].x for i in self.RIGHT_IRIS])
        right_iris_y = np.mean([landmarks[i].y for i in self.RIGHT_IRIS])

        # Average both eyes for gaze position
        gaze_x = (left_iris_x + right_iris_x) / 2
        gaze_y = (left_iris_y + right_iris_y) / 2

        # Calculate quality score based on eye openness
        left_eye_height = abs(landmarks[159].y - landmarks[145].y)
        right_eye_height = abs(landmarks[386].y - landmarks[374].y)
        avg_eye_openness = (left_eye_height + right_eye_height) / 2

        # Quality score between 0 and 1
        quality = min(1.0, avg_eye_openness * 20)
        quality = max(0.3, quality)

        return gaze_x, gaze_y, quality

    def release(self):
        self.face_mesh.close()


class OculomotorExperiment:
    """
    Three-task oculomotor assessment system:
    1. Fixation Stability
    2. Saccade Speed
    3. Smooth Pursuit Accuracy
    """

    def __init__(self):
        self.eye_tracker = EyeTracker()
        self.results = []

        # Screen parameters (will be updated to actual fullscreen size)
        self.screen_width = 1920
        self.screen_height = 1080

        # Calibration points and mapping
        self.calibration_points = []
        self.calibration_data = []
        self.is_calibrated = False

        # Task configuration
        self.fixation_trials = 5  # Number of fixation trials
        self.saccade_trials = 10  # Number of saccade trials
        self.pursuit_trials = 5   # Number of pursuit trials

        # Timing parameters (ms)
        self.fixation_duration = 5000  # 5 seconds per fixation trial
        self.saccade_target_duration = 1500  # Time to make saccade
        self.saccade_iti = 500  # Inter-trial interval
        self.pursuit_duration = 8000  # 8 seconds per pursuit trial

        # Colors (BGR format)
        self.COLOR_BG = (40, 40, 40)
        self.COLOR_TARGET = (0, 255, 255)  # Yellow
        self.COLOR_FIXATION = (0, 255, 0)  # Green
        self.COLOR_TEXT = (255, 255, 255)  # White
        self.COLOR_GAZE = (255, 100, 100)  # Light blue (for feedback)

        # Target parameters
        self.target_radius = 15  # pixels
        self.pursuit_speed = 200  # pixels per second

    def initialize_screen(self, cap):
        """Get actual screen dimensions and set up fullscreen display"""
        cv2.namedWindow('Oculomotor Assessment', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Oculomotor Assessment', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Get actual screen size
        ret, frame = cap.read()
        if ret:
            temp_display = np.zeros((1080, 1920, 3), dtype=np.uint8)
            cv2.imshow('Oculomotor Assessment', temp_display)
            cv2.waitKey(1)

            # Get window size (this will be actual screen size in fullscreen)
            window_name = 'Oculomotor Assessment'
            # Use a default size or get from system
            self.screen_width = 1920
            self.screen_height = 1080

    def show_instruction_screen(self, cap, title, instructions, wait_for_key=True):
        """Display instruction screen with text"""
        display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        display[:] = self.COLOR_BG

        # Title
        cv2.putText(display, title, (self.screen_width // 2 - 300, 150),
                   cv2.FONT_HERSHEY_BOLD, 1.5, self.COLOR_TEXT, 2)

        # Instructions (multi-line)
        y_offset = 250
        for line in instructions:
            cv2.putText(display, line, (self.screen_width // 2 - 400, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 1)
            y_offset += 50

        if wait_for_key:
            cv2.putText(display, "Press ENTER to continue or 'q' to quit",
                       (self.screen_width // 2 - 250, self.screen_height - 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)

        cv2.imshow('Oculomotor Assessment', display)

        if wait_for_key:
            while True:
                key = cv2.waitKey(100)
                if key == 13:  # Enter
                    return True
                elif key == ord('q'):
                    return False
        else:
            cv2.waitKey(2000)
            return True

    def run_calibration(self, cap):
        """
        9-point calibration to map eye features to screen coordinates
        """
        print("\n=== CALIBRATION PHASE ===")

        instructions = [
            "Calibration will help map your gaze to the screen.",
            "",
            "You will see 9 points appear on the screen.",
            "Look directly at each point and press SPACE when ready.",
            "",
            "Keep your head still during calibration."
        ]

        if not self.show_instruction_screen(cap, "Calibration", instructions):
            return False

        # Define 9 calibration points (3x3 grid)
        margin_x = self.screen_width // 6
        margin_y = self.screen_height // 6

        self.calibration_points = [
            (margin_x, margin_y),  # Top-left
            (self.screen_width // 2, margin_y),  # Top-center
            (self.screen_width - margin_x, margin_y),  # Top-right
            (margin_x, self.screen_height // 2),  # Middle-left
            (self.screen_width // 2, self.screen_height // 2),  # Center
            (self.screen_width - margin_x, self.screen_height // 2),  # Middle-right
            (margin_x, self.screen_height - margin_y),  # Bottom-left
            (self.screen_width // 2, self.screen_height - margin_y),  # Bottom-center
            (self.screen_width - margin_x, self.screen_height - margin_y)  # Bottom-right
        ]

        self.calibration_data = []

        for idx, (px, py) in enumerate(self.calibration_points):
            print(f"Calibration point {idx + 1}/9")

            # Show calibration point
            waiting = True
            gaze_samples = []

            while waiting:
                ret, frame = cap.read()
                if not ret:
                    continue

                gaze_x, gaze_y, quality = self.eye_tracker.get_gaze_position(frame)

                # Create display
                display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                display[:] = self.COLOR_BG

                # Draw calibration target
                cv2.circle(display, (px, py), self.target_radius, self.COLOR_TARGET, -1)
                cv2.circle(display, (px, py), self.target_radius + 5, self.COLOR_TARGET, 2)

                # Instructions
                cv2.putText(display, f"Look at the target ({idx + 1}/9)",
                           (self.screen_width // 2 - 150, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 1)
                cv2.putText(display, "Press SPACE when ready",
                           (self.screen_width // 2 - 130, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

                # Quality indicator
                quality_color = self._get_quality_color(quality)
                cv2.rectangle(display, (20, 20), (20 + int(quality * 200), 40), quality_color, -1)

                cv2.imshow('Oculomotor Assessment', display)

                # Collect gaze samples while looking at point
                if gaze_x is not None:
                    gaze_samples.append((gaze_x, gaze_y, quality))

                key = cv2.waitKey(1)
                if key == ord(' ') and len(gaze_samples) > 10:  # Need enough samples
                    # Store average gaze position for this calibration point
                    avg_gaze_x = np.mean([s[0] for s in gaze_samples[-30:]])  # Last 30 samples
                    avg_gaze_y = np.mean([s[1] for s in gaze_samples[-30:]])
                    avg_quality = np.mean([s[2] for s in gaze_samples[-30:]])

                    self.calibration_data.append({
                        'screen_x': px,
                        'screen_y': py,
                        'gaze_x': avg_gaze_x,
                        'gaze_y': avg_gaze_y,
                        'quality': avg_quality
                    })
                    waiting = False
                elif key == ord('q'):
                    return False

        self.is_calibrated = True
        print("Calibration complete!")

        # Show completion message
        self.show_instruction_screen(cap, "Calibration Complete",
                                    ["Calibration successful!",
                                     "Ready to begin tasks."],
                                    wait_for_key=False)
        return True

    def map_gaze_to_screen(self, gaze_x, gaze_y):
        """
        Map raw gaze coordinates to screen coordinates using calibration data
        Uses simple linear interpolation based on calibration points
        """
        if not self.is_calibrated or gaze_x is None or gaze_y is None:
            return None, None

        # Simple linear mapping (can be improved with polynomial fitting)
        # For now, use direct scaling with calibration center point
        center_calib = self.calibration_data[4]  # Center point

        # Calculate offset from calibration center
        gaze_offset_x = gaze_x - center_calib['gaze_x']
        gaze_offset_y = gaze_y - center_calib['gaze_y']

        # Scale to screen coordinates (empirical scaling factor)
        scale_factor_x = self.screen_width * 1.5
        scale_factor_y = self.screen_height * 1.5

        screen_x = center_calib['screen_x'] + gaze_offset_x * scale_factor_x
        screen_y = center_calib['screen_y'] + gaze_offset_y * scale_factor_y

        # Clamp to screen bounds
        screen_x = max(0, min(self.screen_width, screen_x))
        screen_y = max(0, min(self.screen_height, screen_y))

        return int(screen_x), int(screen_y)

    def _get_quality_color(self, quality):
        """Get color based on quality score"""
        if quality >= 0.8:
            return (0, 255, 0)  # Green - excellent
        elif quality >= 0.6:
            return (0, 255, 255)  # Yellow - good
        elif quality >= 0.4:
            return (0, 165, 255)  # Orange - okay
        else:
            return (0, 0, 255)  # Red - poor

    def run_fixation_task(self, cap):
        """
        Task 1: Fixation Stability
        Measures ability to maintain steady gaze on stationary target
        """
        print("\n=== TASK 1: FIXATION STABILITY ===")

        instructions = [
            "TASK 1: Fixation Stability",
            "",
            "A target will appear on the screen.",
            "Keep your gaze steady on the target without moving your eyes.",
            "Try not to blink if possible.",
            "",
            f"You will complete {self.fixation_trials} trials."
        ]

        if not self.show_instruction_screen(cap, "Task 1: Fixation", instructions):
            return False

        # Target positions for fixation trials (varied locations)
        target_positions = [
            (self.screen_width // 2, self.screen_height // 2),  # Center
            (self.screen_width // 3, self.screen_height // 3),  # Upper-left
            (2 * self.screen_width // 3, self.screen_height // 3),  # Upper-right
            (self.screen_width // 3, 2 * self.screen_height // 3),  # Lower-left
            (2 * self.screen_width // 3, 2 * self.screen_height // 3)  # Lower-right
        ]

        for trial_num in range(self.fixation_trials):
            target_x, target_y = target_positions[trial_num]

            print(f"Fixation trial {trial_num + 1}/{self.fixation_trials}")

            # Countdown before trial
            for countdown in range(3, 0, -1):
                display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                display[:] = self.COLOR_BG
                cv2.putText(display, str(countdown),
                           (self.screen_width // 2 - 30, self.screen_height // 2),
                           cv2.FONT_HERSHEY_BOLD, 3, self.COLOR_TEXT, 3)
                cv2.imshow('Oculomotor Assessment', display)
                cv2.waitKey(1000)

            # Run fixation trial
            start_time = time.perf_counter()
            gaze_positions = []
            quality_scores = []

            while (time.perf_counter() - start_time) * 1000 < self.fixation_duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                gaze_x, gaze_y, quality = self.eye_tracker.get_gaze_position(frame)
                screen_gaze_x, screen_gaze_y = self.map_gaze_to_screen(gaze_x, gaze_y)

                # Create display
                display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                display[:] = self.COLOR_BG

                # Draw fixation target
                cv2.circle(display, (target_x, target_y), self.target_radius, self.COLOR_FIXATION, -1)
                cv2.circle(display, (target_x, target_y), 3, (0, 0, 0), -1)  # Center dot

                # Progress indicator
                elapsed = (time.perf_counter() - start_time) * 1000
                progress = elapsed / self.fixation_duration
                bar_width = 300
                cv2.rectangle(display, (self.screen_width // 2 - bar_width // 2, self.screen_height - 50),
                             (self.screen_width // 2 - bar_width // 2 + int(bar_width * progress), self.screen_height - 30),
                             (100, 100, 100), -1)

                # Quality indicator
                quality_color = self._get_quality_color(quality)
                cv2.rectangle(display, (20, 20), (20 + int(quality * 200), 40), quality_color, -1)

                cv2.imshow('Oculomotor Assessment', display)

                # Record data
                if screen_gaze_x is not None:
                    timestamp = (time.perf_counter() - start_time) * 1000
                    gaze_positions.append((timestamp, screen_gaze_x, screen_gaze_y))
                    quality_scores.append(quality)

                if cv2.waitKey(1) == ord('q'):
                    return False

            # Calculate fixation stability metrics
            if len(gaze_positions) > 0:
                gaze_x_coords = [pos[1] for pos in gaze_positions]
                gaze_y_coords = [pos[2] for pos in gaze_positions]

                # Standard deviation (lower = better stability)
                std_x = np.std(gaze_x_coords)
                std_y = np.std(gaze_y_coords)
                fixation_stability = math.sqrt(std_x**2 + std_y**2)  # Overall stability

                # Mean distance from target (accuracy)
                distances = [math.sqrt((gx - target_x)**2 + (gy - target_y)**2)
                            for _, gx, gy in gaze_positions]
                mean_error = np.mean(distances)

                # Store trial results
                self.results.append({
                    'task': 'fixation',
                    'trial_num': trial_num + 1,
                    'target_x': target_x,
                    'target_y': target_y,
                    'fixation_stability_px': fixation_stability,
                    'mean_error_px': mean_error,
                    'std_x': std_x,
                    'std_y': std_y,
                    'avg_quality': np.mean(quality_scores) if quality_scores else 0,
                    'num_samples': len(gaze_positions)
                })

                print(f"  Stability: {fixation_stability:.2f}px, Mean error: {mean_error:.2f}px")

        print("Fixation task complete!")
        return True

    def run_saccade_task(self, cap):
        """
        Task 2: Saccade Speed
        Measures speed of rapid eye movements between targets
        """
        print("\n=== TASK 2: SACCADE SPEED ===")

        instructions = [
            "TASK 2: Saccade Speed",
            "",
            "A target will jump to different locations.",
            "Look at the target as quickly as possible each time it moves.",
            "",
            f"You will complete {self.saccade_trials} trials."
        ]

        if not self.show_instruction_screen(cap, "Task 2: Saccades", instructions):
            return False

        # Generate random target positions (avoiding edges)
        margin = 100
        target_positions = []
        for _ in range(self.saccade_trials):
            x = random.randint(margin, self.screen_width - margin)
            y = random.randint(margin, self.screen_height - margin)
            target_positions.append((x, y))

        prev_target = (self.screen_width // 2, self.screen_height // 2)  # Start at center

        for trial_num in range(self.saccade_trials):
            target_x, target_y = target_positions[trial_num]

            print(f"Saccade trial {trial_num + 1}/{self.saccade_trials}")

            # Show target and wait for saccade
            start_time = time.perf_counter()
            gaze_positions = []
            quality_scores = []
            saccade_detected = False
            saccade_latency = None

            while (time.perf_counter() - start_time) * 1000 < self.saccade_target_duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                gaze_x, gaze_y, quality = self.eye_tracker.get_gaze_position(frame)
                screen_gaze_x, screen_gaze_y = self.map_gaze_to_screen(gaze_x, gaze_y)

                # Create display
                display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                display[:] = self.COLOR_BG

                # Draw new target
                cv2.circle(display, (target_x, target_y), self.target_radius, self.COLOR_TARGET, -1)

                # Draw gaze cursor (for feedback)
                if screen_gaze_x is not None:
                    cv2.circle(display, (screen_gaze_x, screen_gaze_y), 8, self.COLOR_GAZE, 2)

                # Quality indicator
                quality_color = self._get_quality_color(quality)
                cv2.rectangle(display, (20, 20), (20 + int(quality * 200), 40), quality_color, -1)

                cv2.imshow('Oculomotor Assessment', display)

                # Record data
                if screen_gaze_x is not None:
                    timestamp = (time.perf_counter() - start_time) * 1000
                    gaze_positions.append((timestamp, screen_gaze_x, screen_gaze_y))
                    quality_scores.append(quality)

                    # Detect saccade (gaze moves close to target)
                    distance_to_target = math.sqrt((screen_gaze_x - target_x)**2 + (screen_gaze_y - target_y)**2)
                    if not saccade_detected and distance_to_target < 100:  # Within 100px of target
                        saccade_detected = True
                        saccade_latency = timestamp

                if cv2.waitKey(1) == ord('q'):
                    return False

            # Calculate saccade metrics
            if len(gaze_positions) >= 2:
                # Calculate saccade amplitude (distance from previous target to current target)
                amplitude = math.sqrt((target_x - prev_target[0])**2 + (target_y - prev_target[1])**2)

                # Calculate peak velocity (max velocity between consecutive samples)
                velocities = []
                for i in range(1, len(gaze_positions)):
                    t1, x1, y1 = gaze_positions[i-1]
                    t2, x2, y2 = gaze_positions[i]
                    dt = (t2 - t1) / 1000.0  # Convert to seconds
                    if dt > 0:
                        dx = x2 - x1
                        dy = y2 - y1
                        distance = math.sqrt(dx**2 + dy**2)
                        velocity = distance / dt  # pixels per second
                        velocities.append(velocity)

                peak_velocity = max(velocities) if velocities else 0
                mean_velocity = np.mean(velocities) if velocities else 0

                # Final distance from target (accuracy)
                final_gaze_x = gaze_positions[-1][1]
                final_gaze_y = gaze_positions[-1][2]
                final_error = math.sqrt((final_gaze_x - target_x)**2 + (final_gaze_y - target_y)**2)

                # Store trial results
                self.results.append({
                    'task': 'saccade',
                    'trial_num': trial_num + 1,
                    'start_x': prev_target[0],
                    'start_y': prev_target[1],
                    'target_x': target_x,
                    'target_y': target_y,
                    'amplitude_px': amplitude,
                    'saccade_latency_ms': saccade_latency if saccade_latency else None,
                    'peak_velocity_px_per_s': peak_velocity,
                    'mean_velocity_px_per_s': mean_velocity,
                    'final_error_px': final_error,
                    'avg_quality': np.mean(quality_scores) if quality_scores else 0,
                    'num_samples': len(gaze_positions)
                })

                print(f"  Amplitude: {amplitude:.1f}px, Peak velocity: {peak_velocity:.1f}px/s, Error: {final_error:.1f}px")

            # ITI (inter-trial interval)
            time.sleep(self.saccade_iti / 1000.0)
            prev_target = (target_x, target_y)

        print("Saccade task complete!")
        return True

    def run_pursuit_task(self, cap):
        """
        Task 3: Smooth Pursuit
        Measures ability to smoothly track a moving target
        """
        print("\n=== TASK 3: SMOOTH PURSUIT ===")

        instructions = [
            "TASK 3: Smooth Pursuit",
            "",
            "A target will move smoothly across the screen.",
            "Follow the target with your eyes as smoothly as possible.",
            "Try to keep your gaze on the moving target.",
            "",
            f"You will complete {self.pursuit_trials} trials."
        ]

        if not self.show_instruction_screen(cap, "Task 3: Pursuit", instructions):
            return False

        # Define pursuit paths (different motion patterns)
        pursuit_patterns = [
            'horizontal',  # Left to right
            'vertical',    # Top to bottom
            'circular',    # Circular motion
            'horizontal',  # Right to left
            'diagonal'     # Diagonal
        ]

        for trial_num in range(self.pursuit_trials):
            pattern = pursuit_patterns[trial_num % len(pursuit_patterns)]

            print(f"Pursuit trial {trial_num + 1}/{self.pursuit_trials} - Pattern: {pattern}")

            # Countdown
            for countdown in range(3, 0, -1):
                display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                display[:] = self.COLOR_BG
                cv2.putText(display, str(countdown),
                           (self.screen_width // 2 - 30, self.screen_height // 2),
                           cv2.FONT_HERSHEY_BOLD, 3, self.COLOR_TEXT, 3)
                cv2.imshow('Oculomotor Assessment', display)
                cv2.waitKey(1000)

            # Run pursuit trial
            start_time = time.perf_counter()
            gaze_positions = []
            target_positions = []
            quality_scores = []

            while (time.perf_counter() - start_time) * 1000 < self.pursuit_duration:
                ret, frame = cap.read()
                if not ret:
                    continue

                elapsed = (time.perf_counter() - start_time) * 1000

                # Calculate target position based on pattern
                target_x, target_y = self._calculate_pursuit_target(pattern, elapsed)

                gaze_x, gaze_y, quality = self.eye_tracker.get_gaze_position(frame)
                screen_gaze_x, screen_gaze_y = self.map_gaze_to_screen(gaze_x, gaze_y)

                # Create display
                display = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
                display[:] = self.COLOR_BG

                # Draw moving target
                cv2.circle(display, (int(target_x), int(target_y)), self.target_radius, self.COLOR_TARGET, -1)

                # Draw gaze cursor (for feedback)
                if screen_gaze_x is not None:
                    cv2.circle(display, (screen_gaze_x, screen_gaze_y), 8, self.COLOR_GAZE, 2)

                # Progress bar
                progress = elapsed / self.pursuit_duration
                bar_width = 300
                cv2.rectangle(display, (self.screen_width // 2 - bar_width // 2, self.screen_height - 50),
                             (self.screen_width // 2 - bar_width // 2 + int(bar_width * progress), self.screen_height - 30),
                             (100, 100, 100), -1)

                # Quality indicator
                quality_color = self._get_quality_color(quality)
                cv2.rectangle(display, (20, 20), (20 + int(quality * 200), 40), quality_color, -1)

                cv2.imshow('Oculomotor Assessment', display)

                # Record data
                if screen_gaze_x is not None:
                    gaze_positions.append((elapsed, screen_gaze_x, screen_gaze_y))
                    target_positions.append((elapsed, target_x, target_y))
                    quality_scores.append(quality)

                if cv2.waitKey(1) == ord('q'):
                    return False

            # Calculate pursuit metrics
            if len(gaze_positions) > 0 and len(target_positions) > 0:
                # Calculate tracking error (distance between gaze and target over time)
                errors = []
                for (t_gaze, gx, gy), (t_target, tx, ty) in zip(gaze_positions, target_positions):
                    error = math.sqrt((gx - tx)**2 + (gy - ty)**2)
                    errors.append(error)

                mean_error = np.mean(errors)
                std_error = np.std(errors)
                max_error = np.max(errors)

                # Calculate pursuit gain (ratio of eye velocity to target velocity)
                # Simplified: compare displacement over time
                gaze_velocities = []
                target_velocities = []

                for i in range(1, min(len(gaze_positions), len(target_positions))):
                    # Gaze velocity
                    t1, gx1, gy1 = gaze_positions[i-1]
                    t2, gx2, gy2 = gaze_positions[i]
                    dt = (t2 - t1) / 1000.0
                    if dt > 0:
                        gaze_vel = math.sqrt((gx2 - gx1)**2 + (gy2 - gy1)**2) / dt
                        gaze_velocities.append(gaze_vel)

                    # Target velocity
                    t1, tx1, ty1 = target_positions[i-1]
                    t2, tx2, ty2 = target_positions[i]
                    if dt > 0:
                        target_vel = math.sqrt((tx2 - tx1)**2 + (ty2 - ty1)**2) / dt
                        target_velocities.append(target_vel)

                mean_gaze_vel = np.mean(gaze_velocities) if gaze_velocities else 0
                mean_target_vel = np.mean(target_velocities) if target_velocities else 0
                pursuit_gain = mean_gaze_vel / mean_target_vel if mean_target_vel > 0 else 0

                # Store trial results
                self.results.append({
                    'task': 'pursuit',
                    'trial_num': trial_num + 1,
                    'pattern': pattern,
                    'mean_error_px': mean_error,
                    'std_error_px': std_error,
                    'max_error_px': max_error,
                    'pursuit_gain': pursuit_gain,
                    'mean_gaze_velocity': mean_gaze_vel,
                    'mean_target_velocity': mean_target_vel,
                    'avg_quality': np.mean(quality_scores) if quality_scores else 0,
                    'num_samples': len(gaze_positions)
                })

                print(f"  Mean error: {mean_error:.2f}px, Pursuit gain: {pursuit_gain:.2f}")

        print("Pursuit task complete!")
        return True

    def _calculate_pursuit_target(self, pattern, elapsed_ms):
        """Calculate target position for smooth pursuit based on pattern"""
        t = elapsed_ms / 1000.0  # Convert to seconds

        if pattern == 'horizontal':
            # Left to right
            x = (self.screen_width * 0.2) + (self.screen_width * 0.6) * (t / (self.pursuit_duration / 1000.0))
            y = self.screen_height / 2

        elif pattern == 'vertical':
            # Top to bottom
            x = self.screen_width / 2
            y = (self.screen_height * 0.2) + (self.screen_height * 0.6) * (t / (self.pursuit_duration / 1000.0))

        elif pattern == 'circular':
            # Circular motion
            radius = min(self.screen_width, self.screen_height) * 0.3
            angular_velocity = 2 * math.pi / (self.pursuit_duration / 1000.0)  # One full rotation
            angle = angular_velocity * t
            x = self.screen_width / 2 + radius * math.cos(angle)
            y = self.screen_height / 2 + radius * math.sin(angle)

        elif pattern == 'diagonal':
            # Diagonal motion
            progress = t / (self.pursuit_duration / 1000.0)
            x = (self.screen_width * 0.2) + (self.screen_width * 0.6) * progress
            y = (self.screen_height * 0.2) + (self.screen_height * 0.6) * progress

        else:
            x = self.screen_width / 2
            y = self.screen_height / 2

        return x, y

    def save_results(self):
        """Save all task results to CSV"""
        if not self.results:
            print("No results to save!")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"oculomotor_results_{timestamp}.csv"

        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)

        print(f"\nResults saved to: {filename}")
        print(f"Total trials: {len(self.results)}")

        # Print summary statistics
        print("\n=== SUMMARY ===")

        if 'fixation' in df['task'].values:
            fixation_data = df[df['task'] == 'fixation']
            print(f"Fixation Stability: {fixation_data['fixation_stability_px'].mean():.2f}px (mean)")
            print(f"Fixation Accuracy: {fixation_data['mean_error_px'].mean():.2f}px (mean error)")

        if 'saccade' in df['task'].values:
            saccade_data = df[df['task'] == 'saccade']
            print(f"Saccade Peak Velocity: {saccade_data['peak_velocity_px_per_s'].mean():.1f}px/s (mean)")
            print(f"Saccade Accuracy: {saccade_data['final_error_px'].mean():.2f}px (mean error)")

        if 'pursuit' in df['task'].values:
            pursuit_data = df[df['task'] == 'pursuit']
            print(f"Pursuit Tracking Error: {pursuit_data['mean_error_px'].mean():.2f}px (mean)")
            print(f"Pursuit Gain: {pursuit_data['pursuit_gain'].mean():.2f} (mean)")

    def run_experiment(self):
        """Main experiment flow"""
        print("=== OCULOMOTOR ASSESSMENT SYSTEM ===")
        print("Three tasks: Fixation, Saccades, Smooth Pursuit\n")

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        try:
            # Initialize screen
            self.initialize_screen(cap)

            # Welcome screen
            welcome_instructions = [
                "Welcome to the Oculomotor Assessment System",
                "",
                "This assessment consists of three tasks:",
                "  1. Fixation Stability - Hold gaze steady on targets",
                "  2. Saccade Speed - Look quickly at jumping targets",
                "  3. Smooth Pursuit - Follow moving targets",
                "",
                "The entire assessment takes approximately 10-15 minutes.",
                "You may press 'q' at any time to quit."
            ]

            if not self.show_instruction_screen(cap, "Oculomotor Assessment", welcome_instructions):
                return

            # Run calibration
            if not self.run_calibration(cap):
                return

            # Run Task 1: Fixation
            if not self.run_fixation_task(cap):
                return

            # Break between tasks
            self.show_instruction_screen(cap, "Break",
                                        ["Task 1 complete!",
                                         "Take a short break.",
                                         "Press ENTER when ready for Task 2."],
                                        wait_for_key=True)

            # Run Task 2: Saccades
            if not self.run_saccade_task(cap):
                return

            # Break between tasks
            self.show_instruction_screen(cap, "Break",
                                        ["Task 2 complete!",
                                         "Take a short break.",
                                         "Press ENTER when ready for Task 3."],
                                        wait_for_key=True)

            # Run Task 3: Pursuit
            if not self.run_pursuit_task(cap):
                return

            # Completion screen
            self.show_instruction_screen(cap, "Assessment Complete!",
                                        ["Thank you for completing the assessment!",
                                         "Your results have been saved.",
                                         ""],
                                        wait_for_key=False)

            # Save results
            self.save_results()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.eye_tracker.release()
            print("\nExperiment ended.")


if __name__ == "__main__":
    experiment = OculomotorExperiment()
    experiment.run_experiment()
