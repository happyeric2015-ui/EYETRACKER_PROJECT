"""
Eye-Tracking Experiment: Pro/Antisaccade Task
Designed for Mac embedded camera in a virtual environment
"""

import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import time
import random
from datetime import datetime


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


class SaccadeExperiment:
    """Pro/Antisaccade experiment controller"""

    def __init__(self):
        self.eye_tracker = EyeTracker()
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

        # Spatial parameters (normalized coordinates)
        self.fixation_box_size = 0.15  # Central fixation zone
        self.target_position = 0.35  # Distance from center

        # Data collection
        self.current_trial = 0
        self.gaze_history = []

    def create_blank_screen(self):
        """Create blank gray screen"""
        return np.ones((self.screen_height, self.screen_width, 3), dtype=np.uint8) * 128

    def draw_fixation(self, screen):
        """Draw central fixation cross"""
        center = (self.screen_width // 2, self.screen_height // 2)
        color = (255, 255, 255)
        cv2.line(screen, (center[0] - 20, center[1]), (center[0] + 20, center[1]), color, 3)
        cv2.line(screen, (center[0], center[1] - 20), (center[0], center[1] + 20), color, 3)
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
        cv2.circle(screen, (target_x, target_y), 30, (0, 255, 0), -1)
        return screen

    def draw_gaze_cursor(self, screen, gaze_x, gaze_y):
        """Draw small cursor showing current gaze position"""
        if gaze_x is None or gaze_y is None:
            return screen

        cursor_x = int(gaze_x * self.screen_width)
        cursor_y = int(gaze_y * self.screen_height)
        cv2.circle(screen, (cursor_x, cursor_y), 10, (0, 0, 255), 2)
        return screen

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

    def run_trial(self, trial_id, trial_type, target_side, is_practice, cap):
        """Run a single trial"""
        self.gaze_history = []
        rt_ms = None
        accuracy = "Error"
        avg_gaze_quality = 0.0
        saccade_velocity = 0.0
        is_flagged_bad = False

        # Phase 1: Fixation
        start_time = time.perf_counter()
        while (time.perf_counter() - start_time) < self.fixation_duration / 1000.0:
            ret, frame = cap.read()
            if not ret:
                continue

            gaze_x, gaze_y, quality = self.eye_tracker.get_gaze_position(frame)

            screen = self.create_blank_screen()
            screen = self.draw_fixation(screen)
            screen = self.draw_gaze_cursor(screen, gaze_x, gaze_y)

            # Display trial info
            if is_practice:
                cv2.putText(screen, "PRACTICE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(screen, f"Trial {trial_id}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(screen, f"Type: {trial_type}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
            screen = self.draw_fixation(screen)
            screen = self.draw_target(screen, target_side)
            screen = self.draw_gaze_cursor(screen, gaze_x, gaze_y)

            # Display trial info
            if is_practice:
                cv2.putText(screen, "PRACTICE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(screen, f"Trial {trial_id}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(screen, f"Type: {trial_type}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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
        """Run the complete experiment"""
        print("Starting Eye-Tracking Experiment")
        print("Press 'q' to quit at any time")

        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        # Create fullscreen window
        cv2.namedWindow("Eye Tracking Experiment", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Eye Tracking Experiment", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

        # Run trials
        for trial_id, trial_type, target_side, is_practice in all_trials:
            result = self.run_trial(trial_id, trial_type, target_side, is_practice, cap)

            if result is None:  # User quit
                break

            if not is_practice:  # Only save test trials
                self.results.append(result)

            print(f"Trial {trial_id} complete: {trial_type} {target_side} - RT: {result['RT_ms']}ms, Accuracy: {result['accuracy']}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.eye_tracker.release()

        # Save results
        self.save_results()

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
    print("Pro/Antisaccade Eye-Tracking Experiment")
    print("=" * 60)
    print("\nInstructions:")
    print("- Keep your head stable and look at the fixation cross")
    print("- PROSACCADE: Look at the target when it appears")
    print("- ANTISACCADE: Look AWAY from the target (opposite side)")
    print("- You will complete 10 practice trials, then 40 test trials")
    print("- Press 'q' to quit at any time")
    print("\nPress ENTER to begin...")
    input()

    experiment = SaccadeExperiment()
    experiment.run_experiment()

    print("\nExperiment complete! Thank you for participating.")


if __name__ == "__main__":
    main()
