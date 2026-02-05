"""
Eye-Tracking Experiment: Pro/Antisaccade Task (Voice-Enabled Version)
Designed for Mac embedded camera in a virtual environment
Enhanced with text-to-speech voice narration for better accessibility
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime
import threading
import pyttsx3


class VoiceAssistant:
    """Text-to-speech voice assistant for experiment narration"""

    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed of speech (words per minute)
        self.engine.setProperty('volume', 0.9)  # Volume level (0.0 to 1.0)

        # Try to use a pleasant Mac voice
        voices = self.engine.getProperty('voices')
        for voice in voices:
            # Prefer female voices like "Samantha" or "Victoria" for better clarity
            if 'samantha' in voice.name.lower() or 'victoria' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

        self.is_speaking = False
        self.speech_thread = None

    def speak(self, text, blocking=False):
        """
        Convert text to speech
        blocking: If True, wait for speech to complete before returning
        """
        if blocking:
            self.engine.say(text)
            self.engine.runAndWait()
        else:
            # Run in separate thread to avoid blocking experiment
            self.speech_thread = threading.Thread(target=self._speak_async, args=(text,))
            self.speech_thread.start()

    def _speak_async(self, text):
        """Internal method for async speech"""
        self.is_speaking = True
        self.engine.say(text)
        self.engine.runAndWait()
        self.is_speaking = False

    def stop(self):
        """Stop current speech"""
        self.engine.stop()

    def wait_until_done(self):
        """Wait for current speech to complete"""
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join()


class EyeTracker:
    """Eye tracking using MediaPipe Face Mesh"""

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
        Returns normalized gaze position (X, Y) and quality score
        X, Y are in range [0, 1] where (0.5, 0.5) is center
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None, None, 0.0

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Get iris centers
        left_iris_x = np.mean([landmarks[i].x for i in self.LEFT_IRIS])
        left_iris_y = np.mean([landmarks[i].y for i in self.LEFT_IRIS])
        right_iris_x = np.mean([landmarks[i].x for i in self.RIGHT_IRIS])
        right_iris_y = np.mean([landmarks[i].y for i in self.RIGHT_IRIS])

        # Average both eyes for gaze position
        gaze_x = (left_iris_x + right_iris_x) / 2
        gaze_y = (left_iris_y + right_iris_y) / 2

        # Calculate quality score based on detection confidence
        # Check eye openness and landmark stability
        left_eye_points = [landmarks[i] for i in self.LEFT_EYE]
        right_eye_points = [landmarks[i] for i in self.RIGHT_EYE]

        # Simple quality metric: check if eyes are reasonably open
        left_eye_height = abs(landmarks[159].y - landmarks[145].y)
        right_eye_height = abs(landmarks[386].y - landmarks[374].y)
        avg_eye_openness = (left_eye_height + right_eye_height) / 2

        # Quality score between 0 and 1
        quality = min(1.0, avg_eye_openness * 20)  # Scale factor for normalization
        quality = max(0.3, quality)  # Minimum quality if face detected

        return gaze_x, gaze_y, quality

    def release(self):
        self.face_mesh.close()


class SaccadeExperimentVoice:
    """Voice-enabled Pro/Antisaccade experiment controller"""

    def __init__(self):
        self.eye_tracker = EyeTracker()
        self.voice = VoiceAssistant()
        self.results = []
        self.screen_width = 1280
        self.screen_height = 720

        # Trial configuration
        self.practice_trials = 10
        self.test_trials = 40

        # Timing parameters (ms)
        self.fixation_duration = 1000
        self.target_duration = 1500
        self.iti_duration = 500
        self.countdown_duration = 3000

        # Spatial parameters (normalized coordinates)
        self.fixation_box_size = 0.15  # Central fixation zone
        self.target_position = 0.35  # Distance from center

        # Data collection
        self.current_trial = 0
        self.gaze_history = []

        # Quality thresholds
        self.quality_good = 0.8
        self.quality_okay = 0.6
        self.quality_bad = 0.4

    def create_blank_screen(self, color=(50, 50, 50)):
        """Create blank screen with specified color"""
        return np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * np.array(color, dtype=np.uint8)

    def draw_text_centered(self, screen, text, y_pos, font_scale=1.0, color=(255, 255, 255), thickness=2):
        """Draw centered text at specified y position"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x_pos = (self.screen_width - text_size[0]) // 2
        cv2.putText(screen, text, (x_pos, y_pos), font, font_scale, color, thickness, cv2.LINE_AA)
        return screen

    def draw_text_multiline(self, screen, lines, start_y, font_scale=0.7, color=(255, 255, 255), line_spacing=40):
        """Draw multiple lines of centered text"""
        for i, line in enumerate(lines):
            self.draw_text_centered(screen, line, start_y + i * line_spacing, font_scale, color)
        return screen

    def draw_fixation(self, screen, quality=None):
        """Draw central fixation cross with color-coded quality indicator"""
        center = (self.screen_width // 2, self.screen_height // 2)

        # Color based on quality
        if quality is None or quality < self.quality_bad:
            color = (0, 0, 255)  # Red - poor
        elif quality < self.quality_okay:
            color = (0, 165, 255)  # Orange - okay
        elif quality < self.quality_good:
            color = (0, 255, 255)  # Yellow - good
        else:
            color = (0, 255, 0)  # Green - excellent

        # Draw cross
        cv2.line(screen, (center[0] - 20, center[1]), (center[0] + 20, center[1]), color, 4)
        cv2.line(screen, (center[0], center[1] - 20), (center[0], center[1] + 20), color, 4)

        # Draw quality circle around fixation
        cv2.circle(screen, center, 35, color, 2)

        return screen

    def draw_target(self, screen, side):
        """Draw target circle on left or right side"""
        center_x = self.screen_width // 2
        offset = int(self.screen_width * self.target_position)

        if side == "left":
            target_x = center_x - offset
        else:
            target_x = center_x + offset

        target_y = self.screen_height // 2
        cv2.circle(screen, (target_x, target_y), 35, (0, 255, 0), -1)
        cv2.circle(screen, (target_x, target_y), 37, (255, 255, 255), 2)
        return screen

    def draw_gaze_cursor(self, screen, gaze_x, gaze_y):
        """Draw small cursor showing current gaze position"""
        if gaze_x is None or gaze_y is None:
            return screen

        cursor_x = int(gaze_x * self.screen_width)
        cursor_y = int(gaze_y * self.screen_height)
        cv2.circle(screen, (cursor_x, cursor_y), 8, (255, 0, 255), 2)
        return screen

    def draw_progress_bar(self, screen, current, total, text="Progress"):
        """Draw progress bar showing trial completion"""
        bar_width = 400
        bar_height = 30
        bar_x = (self.screen_width - bar_width) // 2
        bar_y = 50

        # Draw background
        cv2.rectangle(screen, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)

        # Draw progress
        progress = int((current / total) * bar_width)
        cv2.rectangle(screen, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height), (0, 255, 150), -1)

        # Draw border
        cv2.rectangle(screen, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)

        # Draw text
        progress_text = f"{text}: {current}/{total} ({int(current/total*100)}%)"
        self.draw_text_centered(screen, progress_text, bar_y - 10, 0.6, (255, 255, 255), 1)

        return screen

    def draw_quality_indicator(self, screen, quality):
        """Draw quality indicator in corner"""
        x, y = 20, self.screen_height - 60

        # Quality bar
        bar_width = 200
        bar_height = 20
        cv2.rectangle(screen, (x, y), (x + bar_width, y + bar_height), (100, 100, 100), -1)

        # Fill based on quality
        fill_width = int(quality * bar_width)
        if quality >= self.quality_good:
            color = (0, 255, 0)  # Green
        elif quality >= self.quality_okay:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red

        cv2.rectangle(screen, (x, y), (x + fill_width, y + bar_height), color, -1)
        cv2.rectangle(screen, (x, y), (x + bar_width, y + bar_height), (255, 255, 255), 2)

        # Text
        cv2.putText(screen, f"Tracking Quality: {quality:.2f}", (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        return screen

    def show_calibration_screen(self, cap):
        """Show calibration screen with voice guidance"""
        print("Starting calibration...")

        # Voice narration
        self.voice.speak("Welcome to the calibration phase. Please look at the fixation cross in the center. Keep your head stable. The cross color indicates tracking quality.", blocking=False)

        quality_samples = []
        start_time = time.perf_counter()
        calibration_duration = 5.0  # 5 seconds

        while (time.perf_counter() - start_time) < calibration_duration:
            ret, frame = cap.read()
            if not ret:
                continue

            gaze_x, gaze_y, quality = self.eye_tracker.get_gaze_position(frame)
            quality_samples.append(quality)

            screen = self.create_blank_screen()

            # Title
            self.draw_text_centered(screen, "CALIBRATION CHECK", 80, 1.5, (100, 255, 255), 3)

            # Instructions
            lines = [
                "Please look at the fixation cross in the center",
                "Keep your head stable",
                "The cross color indicates tracking quality:"
            ]
            self.draw_text_multiline(screen, lines, 180, 0.8)

            # Color legend
            legend_y = 300
            cv2.putText(screen, "GREEN = Excellent", (self.screen_width // 2 - 150, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(screen, "YELLOW = Good", (self.screen_width // 2 - 150, legend_y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(screen, "ORANGE = Okay", (self.screen_width // 2 - 150, legend_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            cv2.putText(screen, "RED = Poor", (self.screen_width // 2 - 150, legend_y + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Draw fixation with quality
            screen = self.draw_fixation(screen, quality)

            # Draw gaze cursor
            screen = self.draw_gaze_cursor(screen, gaze_x, gaze_y)

            # Quality indicator
            if quality is not None and quality > 0:
                screen = self.draw_quality_indicator(screen, quality)

            # Countdown
            remaining = int(calibration_duration - (time.perf_counter() - start_time))
            self.draw_text_centered(screen, f"Time remaining: {remaining}s",
                                   self.screen_height - 120, 0.8, (255, 255, 255))

            cv2.imshow("Eye Tracking Experiment", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        # Evaluate calibration
        avg_quality = np.mean([q for q in quality_samples if q > 0])

        if avg_quality < self.quality_bad:
            self.show_calibration_warning(cap, avg_quality)
        else:
            self.voice.speak("Calibration successful. Your tracking quality is good.", blocking=True)

        return True

    def show_calibration_warning(self, cap, quality):
        """Show warning with voice narration if calibration quality is poor"""
        self.voice.speak(f"Warning. Low tracking quality detected. Average quality is {quality:.1f}.", blocking=False)

        screen = self.create_blank_screen()

        self.draw_text_centered(screen, "WARNING: Low Tracking Quality", 150, 1.2, (0, 100, 255), 2)

        lines = [
            f"Average quality: {quality:.2f}",
            "",
            "Tips to improve tracking:",
            "- Adjust lighting (avoid backlighting)",
            "- Move closer to the camera",
            "- Remove glasses if possible",
            "- Keep your head stable",
            "",
            "Press ENTER to continue anyway or 'q' to quit"
        ]
        self.draw_text_multiline(screen, lines, 250, 0.7, (255, 255, 255), 35)

        cv2.imshow("Eye Tracking Experiment", screen)

        self.voice.wait_until_done()
        self.voice.speak("Please adjust your setup and press enter to continue, or press Q to quit.", blocking=False)

        while True:
            key = cv2.waitKey(100)
            if key == ord('q'):
                return False
            elif key == 13:  # Enter
                return True

    def show_instruction_screen(self, cap):
        """Show detailed instruction screen with voice narration"""
        instruction_pages = [
            {
                "title": "Welcome to the Eye-Tracking Experiment",
                "lines": [
                    "This experiment measures your eye movements",
                    "in response to visual targets.",
                    "",
                    "You will complete:",
                    "- 10 practice trials (with feedback)",
                    "- 40 test trials",
                    "",
                    "Press ENTER to continue"
                ],
                "voice": "Welcome to the Eye-Tracking Experiment. This experiment measures your eye movements in response to visual targets. You will complete 10 practice trials with feedback, followed by 40 test trials. Press Enter to continue."
            },
            {
                "title": "PROSACCADE Trials",
                "lines": [
                    "When you see 'Prosaccade':",
                    "",
                    "LOOK AT THE TARGET",
                    "",
                    "Move your eyes as quickly as possible",
                    "toward the green circle when it appears.",
                    "",
                    "Press ENTER to continue"
                ],
                "voice": "For Prosaccade trials, when you see a green target appear, look directly at the target. Move your eyes as quickly as possible toward the green circle. Press Enter to continue."
            },
            {
                "title": "ANTISACCADE Trials",
                "lines": [
                    "When you see 'Antisaccade':",
                    "",
                    "LOOK AWAY FROM THE TARGET",
                    "",
                    "Move your eyes as quickly as possible",
                    "to the OPPOSITE side from where the target appears.",
                    "",
                    "Press ENTER to continue"
                ],
                "voice": "For Antisaccade trials, when you see the target, look away from it. Move your eyes as quickly as possible to the opposite side from where the target appears. Press Enter to continue."
            },
            {
                "title": "Important Tips",
                "lines": [
                    "- Keep your head as still as possible",
                    "- Always start by looking at the center cross",
                    "- Respond as quickly and accurately as you can",
                    "- You'll get breaks during the experiment",
                    "- Press 'q' at any time to quit",
                    "",
                    "Press ENTER to begin calibration"
                ],
                "voice": "Important tips. Keep your head as still as possible. Always start by looking at the center cross. Respond as quickly and accurately as you can. You will get breaks during the experiment. Press Q at any time to quit. Press Enter to begin calibration."
            }
        ]

        for page in instruction_pages:
            # Start voice narration
            self.voice.speak(page["voice"], blocking=False)

            screen = self.create_blank_screen()

            # Title
            self.draw_text_centered(screen, page["title"], 100, 1.2, (100, 200, 255), 3)

            # Draw line under title
            title_y = 120
            cv2.line(screen, (200, title_y), (self.screen_width - 200, title_y), (100, 200, 255), 2)

            # Content
            self.draw_text_multiline(screen, page["lines"], 180, 0.8, (255, 255, 255), 40)

            cv2.imshow("Eye Tracking Experiment", screen)

            while True:
                key = cv2.waitKey(100)
                if key == ord('q'):
                    self.voice.stop()
                    return False
                elif key == 13:  # Enter
                    self.voice.stop()
                    break

        return True

    def show_countdown(self, cap, message="Get Ready!", voice_message=None, duration=3):
        """Show countdown with voice narration"""
        if voice_message:
            self.voice.speak(voice_message, blocking=False)

        start_time = time.perf_counter()

        while (time.perf_counter() - start_time) < duration:
            ret, frame = cap.read()
            if not ret:
                continue

            remaining = int(duration - (time.perf_counter() - start_time)) + 1

            screen = self.create_blank_screen()
            self.draw_text_centered(screen, message, 250, 1.5, (100, 255, 255), 3)
            self.draw_text_centered(screen, str(remaining), 400, 4.0, (0, 255, 0), 8)

            cv2.imshow("Eye Tracking Experiment", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        return True

    def show_break_screen(self, cap, current_trial, total_trials):
        """Show break screen with voice encouragement"""
        completion_pct = int(current_trial/total_trials*100)
        self.voice.speak(f"Time for a break. You have completed {current_trial} of {total_trials} trials. That's {completion_pct} percent done. Great job! Take your time, and press Enter when you're ready to continue.", blocking=False)

        screen = self.create_blank_screen()

        self.draw_text_centered(screen, "Take a Break!", 200, 1.5, (100, 255, 100), 3)

        lines = [
            f"You've completed {current_trial} of {total_trials} trials",
            f"That's {completion_pct}% done!",
            "",
            "Feel free to:",
            "- Blink and rest your eyes",
            "- Adjust your position",
            "- Stretch",
            "",
            "Press ENTER when ready to continue",
            "Press 'q' to quit"
        ]
        self.draw_text_multiline(screen, lines, 280, 0.8, (255, 255, 255), 35)

        cv2.imshow("Eye Tracking Experiment", screen)

        while True:
            key = cv2.waitKey(100)
            if key == ord('q'):
                self.voice.stop()
                return False
            elif key == 13:  # Enter
                self.voice.stop()
                return True

    def show_practice_feedback(self, cap, accuracy, rt_ms):
        """Show feedback with voice narration after practice trial"""
        screen = self.create_blank_screen()

        # Feedback based on accuracy
        if accuracy == "Correct":
            feedback_color = (0, 255, 0)
            feedback_text = "CORRECT!"
            emoji = "✓"
            voice_text = f"Correct! Your response time was {int(rt_ms)} milliseconds."
        else:
            feedback_color = (0, 100, 255)
            feedback_text = "INCORRECT"
            emoji = "✗"
            voice_text = f"Incorrect. Try to follow the instruction carefully. Your response time was {int(rt_ms)} milliseconds."

        self.voice.speak(voice_text, blocking=False)

        self.draw_text_centered(screen, emoji + " " + feedback_text + " " + emoji,
                               250, 2.0, feedback_color, 4)
        self.draw_text_centered(screen, f"Response Time: {rt_ms:.0f} ms",
                               350, 1.0, (255, 255, 255), 2)

        cv2.imshow("Eye Tracking Experiment", screen)
        cv2.waitKey(1000)  # Show for 1 second
        self.voice.wait_until_done()

    def is_gaze_outside_fixation(self, gaze_x, gaze_y):
        """Check if gaze has left the central fixation box"""
        if gaze_x is None or gaze_y is None:
            return False

        # Check if gaze is outside the central fixation box
        dist_from_center_x = abs(gaze_x - 0.5)
        dist_from_center_y = abs(gaze_y - 0.5)

        return (dist_from_center_x > self.fixation_box_size / 2 or
                dist_from_center_y > self.fixation_box_size / 2)

    def check_saccade_direction(self, gaze_x, target_side):
        """Determine if saccade went in correct direction"""
        if gaze_x is None:
            return "Error"

        # Check if gaze moved to the correct side
        if target_side == "left":
            return "Correct" if gaze_x < 0.5 else "Error"
        else:
            return "Correct" if gaze_x > 0.5 else "Error"

    def calculate_saccade_velocity(self, gaze_history, rt_ms):
        """Calculate approximate saccade velocity from gaze history"""
        if len(gaze_history) < 2:
            return 0.0

        # Find frames around the saccade
        rt_seconds = rt_ms / 1000.0
        rt_frames = int(rt_seconds * 30)  # Assuming ~30 fps

        if rt_frames < len(gaze_history):
            start_idx = max(0, rt_frames - 3)
            end_idx = min(len(gaze_history) - 1, rt_frames + 3)

            if end_idx > start_idx:
                start_gaze = gaze_history[start_idx]
                end_gaze = gaze_history[end_idx]

                if start_gaze[0] is not None and end_gaze[0] is not None:
                    distance = np.sqrt((end_gaze[0] - start_gaze[0])**2 +
                                     (end_gaze[1] - start_gaze[1])**2)
                    time_diff = (end_idx - start_idx) / 30.0  # seconds
                    velocity = distance / time_diff if time_diff > 0 else 0.0
                    return velocity * 1000  # Convert to units per second

        return 0.0

    def run_trial(self, trial_id, trial_type, target_side, is_practice, total_trials, cap):
        """Run a single trial with voice-enhanced UI"""
        self.gaze_history = []
        rt_ms = None
        accuracy = "Error"
        avg_gaze_quality = 0.0
        saccade_velocity = 0.0
        is_flagged_bad = False

        # Determine instruction text
        if trial_type == "Prosaccade":
            instruction = "LOOK AT THE TARGET"
            instruction_color = (0, 255, 255)
        else:
            instruction = "LOOK AWAY FROM TARGET"
            instruction_color = (255, 100, 255)

        # Phase 1: Fixation with countdown
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < self.fixation_duration / 1000.0:
            ret, frame = cap.read()
            if not ret:
                continue

            gaze_x, gaze_y, quality = self.eye_tracker.get_gaze_position(frame)

            screen = self.create_blank_screen()

            # Progress bar
            screen = self.draw_progress_bar(screen, trial_id - 1, total_trials,
                                           "Practice" if is_practice else "Test Trials")

            # Draw fixation with quality indicator
            screen = self.draw_fixation(screen, quality)

            # Only show gaze cursor in practice trials
            if is_practice:
                screen = self.draw_gaze_cursor(screen, gaze_x, gaze_y)

            # Trial information
            info_y = 130
            self.draw_text_centered(screen, f"Trial {trial_id}", info_y, 0.8, (200, 200, 200))
            self.draw_text_centered(screen, trial_type, info_y + 40, 0.9, (255, 255, 100))

            # Instruction
            self.draw_text_centered(screen, instruction, self.screen_height - 200,
                                   0.9, instruction_color, 2)

            # Quality indicator
            if quality is not None and quality > 0:
                screen = self.draw_quality_indicator(screen, quality)

            cv2.imshow("Eye Tracking Experiment", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None

        # Phase 2: Target presentation and response collection
        target_onset = time.perf_counter()
        saccade_detected = False
        quality_scores = []

        while (time.perf_counter() - target_onset) < self.target_duration / 1000.0:
            ret, frame = cap.read()
            if not ret:
                continue

            gaze_x, gaze_y, quality = self.eye_tracker.get_gaze_position(frame)
            self.gaze_history.append((gaze_x, gaze_y, time.perf_counter()))
            quality_scores.append(quality)

            # Check for saccade (first gaze leaving fixation box)
            if not saccade_detected and self.is_gaze_outside_fixation(gaze_x, gaze_y):
                rt_ms = (time.perf_counter() - target_onset) * 1000
                saccade_detected = True

                # Determine accuracy based on trial type
                if trial_type == "Prosaccade":
                    accuracy = self.check_saccade_direction(gaze_x, target_side)
                else:  # Antisaccade
                    # For antisaccade, correct response is OPPOSITE side
                    opposite_side = "right" if target_side == "left" else "left"
                    accuracy = self.check_saccade_direction(gaze_x, opposite_side)

            # Draw screen
            screen = self.create_blank_screen()

            # Progress bar
            screen = self.draw_progress_bar(screen, trial_id - 1, total_trials,
                                           "Practice" if is_practice else "Test Trials")

            # Draw fixation and target
            screen = self.draw_fixation(screen, quality)
            screen = self.draw_target(screen, target_side)

            # Only show gaze cursor in practice trials
            if is_practice:
                screen = self.draw_gaze_cursor(screen, gaze_x, gaze_y)

            # Trial information
            info_y = 130
            self.draw_text_centered(screen, f"Trial {trial_id}", info_y, 0.8, (200, 200, 200))
            self.draw_text_centered(screen, trial_type, info_y + 40, 0.9, (255, 255, 100))

            # Instruction
            self.draw_text_centered(screen, instruction, self.screen_height - 200,
                                   0.9, instruction_color, 2)

            # Quality indicator
            if quality is not None and quality > 0:
                screen = self.draw_quality_indicator(screen, quality)

            cv2.imshow("Eye Tracking Experiment", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None

        # Calculate metrics
        if rt_ms is None:
            rt_ms = self.target_duration  # No response
            accuracy = "Error"

        avg_gaze_quality = np.mean(quality_scores) if quality_scores else 0.0
        saccade_velocity = self.calculate_saccade_velocity(self.gaze_history, rt_ms)

        # Flag bad trials if quality dropped during critical window
        if saccade_detected and quality_scores:
            rt_frames = int((rt_ms / 1000.0) * 30)
            critical_window = quality_scores[max(0, rt_frames-5):min(len(quality_scores), rt_frames+5)]
            if critical_window and min(critical_window) < 0.7:
                is_flagged_bad = True

        # Show feedback for practice trials
        if is_practice:
            self.show_practice_feedback(cap, accuracy, rt_ms)

        # Phase 3: Inter-trial interval
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < self.iti_duration / 1000.0:
            ret, frame = cap.read()
            if not ret:
                continue

            screen = self.create_blank_screen()
            cv2.imshow("Eye Tracking Experiment", screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None

        # Return trial data
        return {
            'trial_id': trial_id,
            'trial_type': trial_type,
            'target_side': target_side,
            'RT_ms': round(rt_ms, 2),
            'accuracy': accuracy,
            'avg_gaze_quality': round(avg_gaze_quality, 3),
            'saccade_velocity': round(saccade_velocity, 2),
            'is_flagged_bad': is_flagged_bad
        }

    def run_experiment(self):
        """Run the complete experiment with voice assistance"""
        print("Starting Voice-Enabled Eye-Tracking Experiment")

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Create fullscreen window
        cv2.namedWindow("Eye Tracking Experiment", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Eye Tracking Experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Show instruction screens
        if not self.show_instruction_screen(cap):
            cap.release()
            cv2.destroyAllWindows()
            return

        # Run calibration
        if not self.show_calibration_screen(cap):
            cap.release()
            cv2.destroyAllWindows()
            return

        # Countdown before starting
        if not self.show_countdown(cap, "Starting Practice Trials...",
                                   "Get ready. Practice trials will begin in 3, 2, 1.", 3):
            cap.release()
            cv2.destroyAllWindows()
            return

        # Generate trial list
        trial_types = ['Prosaccade', 'Antisaccade']
        sides = ['left', 'right']

        # Practice trials
        practice_list = []
        for i in range(self.practice_trials):
            trial_type = random.choice(trial_types)
            side = random.choice(sides)
            practice_list.append((i + 1, trial_type, side, True))

        # Test trials (balanced)
        test_list = []
        for trial_type in trial_types:
            for side in sides:
                for _ in range(self.test_trials // 4):
                    test_list.append((trial_type, side))

        random.shuffle(test_list)
        test_list = [(i + self.practice_trials + 1, tt, s, False)
                     for i, (tt, s) in enumerate(test_list)]

        all_trials = practice_list + test_list
        total_trials = len(all_trials)

        # Run trials
        for trial_id, trial_type, target_side, is_practice in all_trials:
            # Break after practice trials
            if trial_id == self.practice_trials + 1:
                if not self.show_break_screen(cap, self.practice_trials, total_trials):
                    break
                if not self.show_countdown(cap, "Starting Test Trials...",
                                          "Great job on the practice. Now let's begin the test trials.", 3):
                    break

            # Optional breaks every 10 test trials
            if not is_practice and trial_id > self.practice_trials and (trial_id - self.practice_trials) % 10 == 0:
                if not self.show_break_screen(cap, trial_id - self.practice_trials, self.test_trials):
                    break
                if not self.show_countdown(cap, "Resuming...", "Let's continue.", 2):
                    break

            result = self.run_trial(trial_id, trial_type, target_side, is_practice, total_trials, cap)

            if result is None:  # User quit
                break

            if not is_practice:  # Only save test trials
                self.results.append(result)

            print(f"Trial {trial_id} complete: {trial_type} {target_side} - RT: {result['RT_ms']}ms, Accuracy: {result['accuracy']}")

        # Show completion screen
        self.show_completion_screen(cap)

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.eye_tracker.release()

        # Save results
        self.save_results()

    def show_completion_screen(self, cap):
        """Show experiment completion screen with voice summary"""
        if not self.results:
            return

        df = pd.DataFrame(self.results)
        avg_rt = df['RT_ms'].mean()
        accuracy_pct = (df['accuracy'] == 'Correct').sum() / len(df) * 100

        voice_summary = f"Experiment complete! Thank you for participating. You completed {len(df)} trials with an average response time of {int(avg_rt)} milliseconds and an overall accuracy of {int(accuracy_pct)} percent. Excellent work!"
        self.voice.speak(voice_summary, blocking=False)

        screen = self.create_blank_screen()

        self.draw_text_centered(screen, "Experiment Complete!", 150, 1.8, (100, 255, 100), 3)

        lines = [
            "Thank you for participating!",
            "",
            f"Trials completed: {len(df)}",
            f"Average Response Time: {avg_rt:.0f} ms",
            f"Overall Accuracy: {accuracy_pct:.1f}%",
            "",
            "Your data has been saved.",
            "",
            "Press any key to exit"
        ]
        self.draw_text_multiline(screen, lines, 250, 0.9, (255, 255, 255), 40)

        cv2.imshow("Eye Tracking Experiment", screen)
        cv2.waitKey(0)
        self.voice.wait_until_done()

    def save_results(self):
        """Save results to CSV"""
        if not self.results:
            print("No results to save")
            return

        df = pd.DataFrame(self.results)
        filename = f"eye_tracking_results_DATA.csv"
        df.to_csv(filename, index=False)
        print(f"\nResults saved to {filename}")
        print(f"Total trials completed: {len(self.results)}")
        print(f"Average RT: {df['RT_ms'].mean():.2f}ms")
        print(f"Overall Accuracy: {(df['accuracy'] == 'Correct').sum() / len(df) * 100:.1f}%")
        print(f"Flagged trials: {df['is_flagged_bad'].sum()} ({df['is_flagged_bad'].sum() / len(df) * 100:.1f}%)")


def main():
    """Main entry point"""
    print("=" * 60)
    print("Pro/Antisaccade Eye-Tracking Experiment (Voice-Enabled)")
    print("=" * 60)
    print("\nPress ENTER to begin...")
    input()

    experiment = SaccadeExperimentVoice()
    experiment.run_experiment()

    print("\nExperiment complete! Thank you for participating.")


if __name__ == "__main__":
    main()
