# Eye Experiment Enhanced - Detailed Documentation

**File:** `eye_experiment_enhanced.py`
**Purpose:** Pro/Antisaccade cognitive task with enhanced user interface, calibration, and real-time feedback
**Version:** Enhanced (Legacy System)

---

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Class: EyeTracker](#class-eyetracker)
4. [Class: SaccadeExperimentEnhanced](#class-saccadeexperimentenhanced)
5. [Parameter Configuration](#parameter-configuration)
6. [Function Reference](#function-reference)
7. [Experiment Flow](#experiment-flow)
8. [Data Output](#data-output)

---

## Overview

### Purpose
This program implements a Pro/Antisaccade eye-tracking experiment designed to measure executive function and inhibitory control. Participants must either look toward a target (Prosaccade) or away from it (Antisaccade), testing their ability to inhibit reflexive eye movements.

### Key Features
- **Enhanced User Experience**: Visual instruction screens, progress tracking, quality indicators
- **Real-time Feedback**: Color-coded fixation cross, gaze cursor (practice trials), immediate accuracy feedback
- **Quality Control**: Pre-experiment calibration, real-time quality monitoring, post-hoc trial flagging
- **User-Friendly Design**: Automatic breaks, countdown timers, completion summary
- **Practice & Test Phases**: 10 practice trials with feedback + 40 balanced test trials

### Technical Stack
- **Eye Tracking**: MediaPipe Face Mesh (iris landmark detection)
- **Display**: OpenCV (fullscreen rendering)
- **Data Processing**: NumPy, Pandas
- **Camera**: Standard webcam (30 fps assumed)

---

## System Architecture

### Data Flow
```
Camera Input → MediaPipe Face Mesh → Gaze Coordinates → Trial Logic → CSV Output
                                    ↓
                              Quality Monitoring
                                    ↓
                              Visual Feedback
```

### Trial Structure
```
Fixation Phase (1000ms) → Target Phase (1500ms) → ITI (500ms)
        ↓                         ↓                    ↓
   Show fixation cross    Show target + fixation   Blank screen
   Monitor quality        Detect saccade           Rest period
                          Calculate RT
```

---

## Class: EyeTracker

### Purpose
Wraps MediaPipe Face Mesh for iris tracking and gaze position estimation.

### Initialization (`__init__`)

```python
def __init__(self):
    self.mp_face_mesh = mp.solutions.face_mesh
    self.face_mesh = self.mp_face_mesh.FaceMesh(
        max_num_faces=1,              # Track only one face
        refine_landmarks=True,         # Enable iris tracking
        min_detection_confidence=0.5,  # Threshold for face detection
        min_tracking_confidence=0.5    # Threshold for tracking
    )
```

**Landmark Indices:**
- `LEFT_IRIS`: [474, 475, 476, 477] - Four points defining left iris boundary
- `RIGHT_IRIS`: [469, 470, 471, 472] - Four points defining right iris boundary
- `LEFT_EYE`: [33, 133, 160, 159, 158, 157, 173] - Left eye contour for openness calculation
- `RIGHT_EYE`: [362, 263, 387, 386, 385, 384, 398] - Right eye contour

### Method: `get_gaze_position(frame)`

**Purpose:** Process camera frame and extract gaze position with quality score.

**Algorithm:**
1. Convert BGR frame to RGB (MediaPipe requirement)
2. Process frame through Face Mesh model
3. Extract iris landmarks for both eyes
4. Calculate iris centers (mean of 4 iris points per eye)
5. Average both eyes for final gaze position
6. Calculate quality score based on eye openness

**Parameters:**
- `frame`: OpenCV image (BGR format)

**Returns:**
- `gaze_x`: Normalized X coordinate [0, 1], where 0.5 = center
- `gaze_y`: Normalized Y coordinate [0, 1], where 0.5 = center
- `quality`: Quality score [0, 1], based on eye openness

**Quality Calculation:**
```python
# Eye openness = vertical distance between eyelids
left_eye_height = abs(landmarks[159].y - landmarks[145].y)
right_eye_height = abs(landmarks[386].y - landmarks[374].y)
avg_eye_openness = (left_eye_height + right_eye_height) / 2

# Scale to [0, 1] range
quality = min(1.0, avg_eye_openness * 20)  # Scale factor = 20
quality = max(0.3, quality)  # Minimum = 0.3 if face detected
```

**Edge Cases:**
- No face detected → Returns `(None, None, 0.0)`
- Eye closed/partially closed → Lower quality score
- Head movement → Iris position changes, quality may decrease

---

## Class: SaccadeExperimentEnhanced

### Purpose
Controls the Pro/Antisaccade experiment flow with enhanced UI/UX features.

---

## Parameter Configuration

### Initialization Parameters (`__init__`)

#### Display Parameters
```python
self.screen_width = 1280   # Screen width in pixels
self.screen_height = 720   # Screen height in pixels
```
**Note:** These are default values. The actual screen size is determined by OpenCV's fullscreen mode.

#### Trial Configuration
```python
self.practice_trials = 10  # Number of practice trials with feedback
self.test_trials = 40      # Number of test trials (data collection)
```
**Total experiment duration:** ~8-12 minutes
- Practice: ~10 trials × 3 seconds/trial = ~30 seconds
- Test: 40 trials × 3 seconds/trial = ~2 minutes
- Breaks + instructions: ~2-3 minutes

#### Timing Parameters (milliseconds)
```python
self.fixation_duration = 1000     # Time to look at fixation cross before target
self.target_duration = 1500       # Maximum time to respond after target appears
self.iti_duration = 500           # Inter-trial interval (blank screen)
self.countdown_duration = 3000    # Countdown before trial blocks
```

**Timing Rationale:**
- **Fixation (1000ms):** Ensures gaze is centered before target onset; typical in saccade research
- **Target (1500ms):** Sufficient time for saccade execution (typical RT: 200-400ms) plus tracking
- **ITI (500ms):** Brief rest between trials to reduce eye fatigue

#### Spatial Parameters (normalized coordinates)
```python
self.fixation_box_size = 0.15  # Size of central fixation zone [0-1]
self.target_position = 0.35    # Distance from center to target [0-1]
```

**Coordinate System:**
- Normalized coordinates: [0, 1] range
- Center of screen: (0.5, 0.5)
- `fixation_box_size = 0.15` → 15% of screen width/height
- `target_position = 0.35` → 35% of screen width from center

**Example (1920×1080 screen):**
- Fixation box: 288×162 pixels (centered)
- Target offset: 672 pixels from center (left or right)

#### Quality Thresholds
```python
self.quality_good = 0.8   # Green indicator (excellent tracking)
self.quality_okay = 0.6   # Yellow indicator (good tracking)
self.quality_bad = 0.4    # Red indicator (poor tracking)
```

**Color Coding:**
- **Green (≥0.8):** Optimal eye openness, stable tracking
- **Yellow (0.6-0.8):** Good tracking, minor quality issues
- **Orange (0.4-0.6):** Acceptable tracking, some concerns
- **Red (<0.4):** Poor tracking, consider recalibrating

---

## Function Reference

### Utility Functions

#### `create_blank_screen(color=(50, 50, 50))`
**Purpose:** Create uniform background screen.

**Parameters:**
- `color`: RGB tuple (default: dark gray)

**Returns:** NumPy array (screen_height × screen_width × 3)

**Usage:**
```python
screen = self.create_blank_screen()  # Dark gray background
screen = self.create_blank_screen((0, 0, 0))  # Black background
```

---

#### `draw_text_centered(screen, text, y_pos, font_scale=1.0, color=(255,255,255), thickness=2)`
**Purpose:** Draw horizontally centered text at specified vertical position.

**Parameters:**
- `screen`: Target image array
- `text`: String to display
- `y_pos`: Y coordinate (pixels from top)
- `font_scale`: Text size multiplier (default: 1.0)
- `color`: RGB tuple (default: white)
- `thickness`: Line thickness (default: 2)

**Algorithm:**
1. Calculate text width using `cv2.getTextSize()`
2. Compute X position: `(screen_width - text_width) / 2`
3. Draw text using `cv2.putText()` with anti-aliasing

**Example:**
```python
# Draw title at top of screen
self.draw_text_centered(screen, "Welcome!", 100, font_scale=1.5, color=(100, 255, 255))
```

---

#### `draw_text_multiline(screen, lines, start_y, font_scale=0.7, color=(255,255,255), line_spacing=40)`
**Purpose:** Draw multiple lines of centered text with consistent spacing.

**Parameters:**
- `lines`: List of strings
- `start_y`: Starting Y position for first line
- `line_spacing`: Pixels between lines (default: 40)

**Example:**
```python
instructions = [
    "Line 1: Instructions",
    "Line 2: More details",
    "Line 3: Press ENTER"
]
self.draw_text_multiline(screen, instructions, 200)
```

---

### Visual Feedback Functions

#### `draw_fixation(screen, quality=None)`
**Purpose:** Draw central fixation cross with color-coded quality indicator.

**Algorithm:**
1. Determine color based on quality score (green/yellow/orange/red)
2. Draw horizontal line: center ± 20 pixels
3. Draw vertical line: center ± 20 pixels
4. Draw quality indicator circle (radius: 35 pixels)

**Color Mapping:**
```python
quality >= 0.8  → Green (0, 255, 0)      # Excellent
quality >= 0.6  → Yellow (0, 255, 255)   # Good
quality >= 0.4  → Orange (0, 165, 255)   # Okay
quality < 0.4   → Red (0, 0, 255)        # Poor
```

**Visual Design:**
```
        |
    ----+----  ← Fixation cross (4px thick)
        |

    ( cross )  ← Quality circle (2px thick)
```

---

#### `draw_target(screen, side)`
**Purpose:** Draw target stimulus on left or right side of screen.

**Parameters:**
- `side`: "left" or "right"

**Algorithm:**
1. Calculate target X position:
   - Left: `center_x - (screen_width × 0.35)`
   - Right: `center_x + (screen_width × 0.35)`
2. Draw filled green circle (radius: 35px)
3. Draw white border circle (radius: 37px, thickness: 2px)

**Visual Design:**
```
    ●  ← Target (green filled circle with white border)
```

---

#### `draw_gaze_cursor(screen, gaze_x, gaze_y)`
**Purpose:** Visualize current gaze position (used in practice trials only).

**Algorithm:**
1. Convert normalized coordinates to screen pixels:
   - `cursor_x = gaze_x × screen_width`
   - `cursor_y = gaze_y × screen_height`
2. Draw magenta circle (radius: 8px, thickness: 2px)

**Color:** Magenta (255, 0, 255) - distinct from other UI elements

**Note:** Only shown during practice trials to avoid distracting participants during test trials.

---

#### `draw_progress_bar(screen, current, total, text="Progress")`
**Purpose:** Display trial completion progress with percentage.

**Parameters:**
- `current`: Completed trials
- `total`: Total trials
- `text`: Label text (e.g., "Practice", "Test Trials")

**Layout:**
```
                Progress: 15/40 (38%)
        ┌─────────────────────────────────┐
        │███████████████░░░░░░░░░░░░░░░░░│
        └─────────────────────────────────┘
```

**Colors:**
- Background: Gray (100, 100, 100)
- Progress: Cyan (0, 255, 150)
- Border: White (255, 255, 255)

**Dimensions:**
- Width: 400 pixels (centered horizontally)
- Height: 30 pixels
- Position: 50 pixels from top

---

#### `draw_quality_indicator(screen, quality)`
**Purpose:** Show real-time tracking quality in corner of screen.

**Layout:**
```
Tracking Quality: 0.87
┌────────────────────┐
│████████████████░░░│  ← Green fill (87% quality)
└────────────────────┘
```

**Position:** Bottom-left corner (20px from left, 60px from bottom)

**Colors:**
- Green (≥0.8): Excellent
- Yellow (0.6-0.8): Good
- Red (<0.6): Poor

**Dimensions:**
- Width: 200 pixels
- Height: 20 pixels

---

### Screen Display Functions

#### `show_calibration_screen(cap)`
**Purpose:** Pre-experiment quality check with real-time feedback (5 seconds).

**Process:**
1. Display instructions and color legend
2. Show fixation cross with color-coded quality
3. Display gaze cursor for visual feedback
4. Collect quality samples over 5 seconds
5. Calculate average quality
6. Show warning if quality < 0.4

**User Actions:**
- Watch fixation cross
- Press 'q' to quit

**Quality Legend Displayed:**
- GREEN = Excellent
- YELLOW = Good
- ORANGE = Okay
- RED = Poor

**Output:** `True` if calibration completed, `False` if user quit

---

#### `show_calibration_warning(cap, quality)`
**Purpose:** Alert user if tracking quality is insufficient.

**Trigger:** Average quality < 0.4 during calibration

**Improvement Tips Displayed:**
- Adjust lighting (avoid backlighting)
- Move closer to camera
- Remove glasses if possible
- Keep head stable

**User Actions:**
- Press ENTER to continue anyway
- Press 'q' to quit

---

#### `show_instruction_screen(cap)`
**Purpose:** Multi-page visual instructions explaining the experiment.

**Pages:**

**Page 1: Welcome**
- Experiment overview
- Trial counts (10 practice + 40 test)
- Navigation instructions

**Page 2: Prosaccade Explanation**
- Task description: "LOOK AT THE TARGET"
- Behavioral instruction: Move eyes toward green circle
- Visual emphasis on speed

**Page 3: Antisaccade Explanation**
- Task description: "LOOK AWAY FROM TARGET"
- Behavioral instruction: Move eyes to opposite side
- Cognitive challenge emphasis

**Page 4: Important Tips**
- Head stability
- Starting position (center cross)
- Speed-accuracy tradeoff
- Break information
- Quit option ('q')

**Navigation:**
- Press ENTER to advance pages
- Press 'q' to quit

---

#### `show_countdown(cap, message="Get Ready!", duration=3)`
**Purpose:** Display countdown timer before trial blocks.

**Visual Design:**
```
        Get Ready!

           3     ← Large countdown number (decreases each second)
```

**Parameters:**
- `message`: Text displayed above countdown
- `duration`: Countdown duration in seconds (default: 3)

**Colors:**
- Message: Cyan (100, 255, 255)
- Countdown: Green (0, 255, 0)

**Usage Examples:**
```python
self.show_countdown(cap, "Starting Practice Trials...", 3)
self.show_countdown(cap, "Resuming...", 2)
```

---

#### `show_break_screen(cap, current_trial, total_trials)`
**Purpose:** Provide rest opportunity between trial blocks.

**Display Information:**
- Trials completed
- Percentage complete
- Suggestions: blink, adjust position, stretch

**Frequency:**
- After practice trials (mandatory)
- Every 10 test trials (optional)

**User Actions:**
- Press ENTER to resume
- Press 'q' to quit

**Example Display:**
```
            Take a Break!

    You've completed 20 of 40 trials
            That's 50% done!

            Feel free to:
            - Blink and rest your eyes
            - Adjust your position
            - Stretch
```

---

#### `show_practice_feedback(cap, accuracy, rt_ms)`
**Purpose:** Immediate feedback after each practice trial (1 second display).

**Correct Response:**
```
        ✓ CORRECT! ✓     ← Green text, large font

    Response Time: 287 ms
```

**Incorrect Response:**
```
        ✗ INCORRECT ✗    ← Orange text, large font

    Response Time: 412 ms
```

**Duration:** 1000ms (1 second)

**Note:** Only shown during practice trials, not test trials.

---

#### `show_completion_screen(cap)`
**Purpose:** Display experiment summary and thank participant.

**Summary Statistics:**
- Total trials completed
- Average response time
- Overall accuracy percentage

**Example:**
```
        Experiment Complete!

        Thank you for participating!

        Trials completed: 40
        Average Response Time: 324 ms
        Overall Accuracy: 87.5%

        Your data has been saved.

        Press any key to exit
```

**Waits For:** Any key press to exit

---

### Trial Logic Functions

#### `is_gaze_outside_fixation(gaze_x, gaze_y)`
**Purpose:** Detect when participant's gaze leaves central fixation zone.

**Algorithm:**
```python
# Calculate distance from screen center (0.5, 0.5)
dist_from_center_x = abs(gaze_x - 0.5)
dist_from_center_y = abs(gaze_y - 0.5)

# Check if outside fixation box (default: 0.15 / 2 = 0.075)
outside = (dist_from_center_x > self.fixation_box_size / 2) or
          (dist_from_center_y > self.fixation_box_size / 2)
```

**Returns:** `True` if gaze has left fixation zone, `False` otherwise

**Use Case:** Saccade detection - first gaze departure triggers RT measurement

**Edge Case:** Returns `False` if gaze is `None` (face not detected)

---

#### `check_saccade_direction(gaze_x, target_side)`
**Purpose:** Determine if saccade was made in correct direction.

**Algorithm:**
```python
if target_side == "left":
    correct = gaze_x < 0.5  # Gaze moved to left side of screen
else:  # "right"
    correct = gaze_x > 0.5  # Gaze moved to right side of screen

return "Correct" if correct else "Error"
```

**Parameters:**
- `gaze_x`: Horizontal gaze position [0, 1]
- `target_side`: "left" or "right"

**Returns:** "Correct" or "Error"

**Note:** This checks **direction only**, not endpoint accuracy.

---

#### `calculate_saccade_velocity(gaze_history, rt_ms)`
**Purpose:** Estimate saccade velocity from gaze trajectory.

**Algorithm:**
1. Convert RT (milliseconds) to frame index (assuming 30 fps)
2. Extract gaze samples around saccade time (±3 frames)
3. Calculate Euclidean distance traveled
4. Divide by time interval to get velocity

**Formula:**
```
velocity = sqrt((x2 - x1)² + (y2 - y1)²) / time_diff
velocity_scaled = velocity × 1000  (units per second)
```

**Parameters:**
- `gaze_history`: List of (gaze_x, gaze_y, timestamp) tuples
- `rt_ms`: Reaction time in milliseconds

**Returns:** Velocity in normalized units per second

**Assumptions:**
- Camera frame rate: 30 fps
- Saccade occurs near RT timestamp

**Edge Cases:**
- Insufficient gaze history → Returns 0.0
- Missing gaze data → Returns 0.0

---

### Main Trial Function

#### `run_trial(trial_id, trial_type, target_side, is_practice, total_trials, cap)`
**Purpose:** Execute a single trial with all phases and data collection.

**Parameters:**
- `trial_id`: Sequential trial number (1-50)
- `trial_type`: "Prosaccade" or "Antisaccade"
- `target_side`: "left" or "right"
- `is_practice`: Boolean (practice vs. test trial)
- `total_trials`: Total trial count (for progress bar)
- `cap`: OpenCV camera capture object

**Returns:** Dictionary with trial data, or `None` if user quit

**Trial Phases:**

**Phase 1: Fixation (1000ms)**
- Display fixation cross with quality indicator
- Show trial type and instructions
- Display progress bar
- Show gaze cursor (practice only)
- Monitor quality continuously

**Phase 2: Target & Response (1500ms)**
- Add target stimulus to display
- Detect saccade (first gaze departure from fixation)
- Record reaction time
- Determine accuracy:
  - **Prosaccade:** Correct if gaze moves toward target side
  - **Antisaccade:** Correct if gaze moves opposite to target side
- Collect gaze history and quality scores

**Phase 3: Inter-Trial Interval (500ms)**
- Display blank screen
- Brief rest period

**Accuracy Logic:**
```python
if trial_type == "Prosaccade":
    accuracy = check_saccade_direction(gaze_x, target_side)
else:  # Antisaccade
    opposite_side = "right" if target_side == "left" else "left"
    accuracy = check_saccade_direction(gaze_x, opposite_side)
```

**Quality Control:**
- Calculate average quality during target phase
- Flag trial as bad if quality < 0.7 in critical window (±5 frames around RT)

**Practice Trial Feedback:**
- Show accuracy and RT for 1 second
- Only displayed during practice trials

**Output Data Dictionary:**
```python
{
    'trial_id': int,           # 1-50
    'trial_type': str,         # "Prosaccade" or "Antisaccade"
    'target_side': str,        # "left" or "right"
    'RT_ms': float,           # Reaction time (milliseconds)
    'accuracy': str,          # "Correct" or "Error"
    'avg_gaze_quality': float, # Mean quality [0-1]
    'saccade_velocity': float, # Estimated velocity
    'is_flagged_bad': bool    # Quality flag
}
```

---

### Experiment Controller

#### `run_experiment()`
**Purpose:** Main experiment controller - orchestrates entire session.

**Flow Diagram:**
```
1. Initialize camera
2. Create fullscreen window
3. Show instruction screens (4 pages)
4. Run calibration check (5 seconds)
5. Countdown to practice trials (3 seconds)
6. Generate trial lists:
   - Practice: 10 randomized trials
   - Test: 40 balanced trials (10 per condition)
7. Run practice trials (with feedback)
8. Break screen
9. Countdown to test trials (3 seconds)
10. Run test trials (with optional breaks every 10 trials)
11. Show completion screen
12. Save results to CSV
13. Cleanup (release camera, close windows)
```

**Trial Generation:**

**Practice Trials (10 total):**
- Fully randomized
- Equal distribution not guaranteed
- Feedback shown after each trial

**Test Trials (40 total, balanced):**
```
Prosaccade + Left:  10 trials
Prosaccade + Right: 10 trials
Antisaccade + Left:  10 trials
Antisaccade + Right: 10 trials
Total: 40 trials (randomly shuffled)
```

**Break Points:**
- After trial 10 (end of practice, mandatory)
- After trials 20, 30, 40 (optional, every 10 test trials)

**User Exit Options:**
- Press 'q' during any screen or trial
- Partial data is saved if experiment is interrupted

**Error Handling:**
- Camera not available → Exit with error message
- User quit → Graceful shutdown, save collected data

---

### Data Management

#### `save_results()`
**Purpose:** Export trial data to CSV file.

**Filename:** `eye_tracking_results_DATA.csv`

**CSV Schema:**
```csv
trial_id,trial_type,target_side,RT_ms,accuracy,avg_gaze_quality,saccade_velocity,is_flagged_bad
1,Prosaccade,left,287.45,Correct,0.823,12.45,False
2,Antisaccade,right,412.18,Error,0.765,8.92,False
...
```

**Terminal Output:**
```
Results saved to eye_tracking_results_DATA.csv
Total trials completed: 40
Average RT: 324.56ms
Overall Accuracy: 87.5%
Flagged trials: 3 (7.5%)
```

**Data Included:**
- Only test trials (practice trials excluded)
- All trials, including flagged bad trials (for post-hoc analysis)
- Rounded values for readability (RT: 2 decimals, quality: 3 decimals)

---

## Experiment Flow

### Complete Session Timeline

```
┌─────────────────────────────────────────────────────────────┐
│ 1. SETUP PHASE (1-2 minutes)                                │
├─────────────────────────────────────────────────────────────┤
│   ├─ Launch program                                         │
│   ├─ Press ENTER to begin                                   │
│   ├─ Instruction screens (4 pages)                          │
│   │   └─ Press ENTER to advance each page                   │
│   └─ Calibration check (5 seconds)                          │
│       └─ Look at fixation cross                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 2. PRACTICE PHASE (1-2 minutes)                             │
├─────────────────────────────────────────────────────────────┤
│   ├─ Countdown (3 seconds)                                  │
│   ├─ 10 practice trials                                     │
│   │   ├─ Each trial: ~3 seconds                             │
│   │   ├─ Fixation → Target → Feedback → ITI                │
│   │   └─ Gaze cursor visible                                │
│   └─ Mandatory break                                        │
│       └─ Press ENTER to continue                            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 3. TEST PHASE (3-5 minutes)                                 │
├─────────────────────────────────────────────────────────────┤
│   ├─ Countdown (3 seconds)                                  │
│   ├─ 40 test trials (balanced)                              │
│   │   ├─ Each trial: ~3 seconds                             │
│   │   ├─ Fixation → Target → ITI                            │
│   │   ├─ No feedback shown                                  │
│   │   └─ Optional breaks every 10 trials                    │
│   └─ Data continuously saved                                │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 4. COMPLETION (10 seconds)                                  │
├─────────────────────────────────────────────────────────────┤
│   ├─ Summary screen                                         │
│   │   ├─ Total trials                                       │
│   │   ├─ Average RT                                         │
│   │   └─ Overall accuracy                                   │
│   ├─ Press any key to exit                                  │
│   └─ CSV file saved                                         │
└─────────────────────────────────────────────────────────────┘

Total Duration: ~8-12 minutes
```

---

## Data Output

### CSV File Structure

**Filename:** `eye_tracking_results_DATA.csv`

**Columns:**

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `trial_id` | int | 11-50 | Sequential trial number (practice trials 1-10 excluded) |
| `trial_type` | str | "Prosaccade"<br>"Antisaccade" | Type of cognitive task |
| `target_side` | str | "left"<br>"right" | Screen side where target appeared |
| `RT_ms` | float | 0-1500 | Reaction time in milliseconds (time to first saccade) |
| `accuracy` | str | "Correct"<br>"Error" | Whether response matched task requirements |
| `avg_gaze_quality` | float | 0-1 | Mean quality score during target phase |
| `saccade_velocity` | float | 0-∞ | Estimated saccade velocity (normalized units/sec) |
| `is_flagged_bad` | bool | True/False | Quality flag (True if quality < 0.7 in critical window) |

### Example Data

```csv
trial_id,trial_type,target_side,RT_ms,accuracy,avg_gaze_quality,saccade_velocity,is_flagged_bad
11,Prosaccade,left,287.45,Correct,0.823,12.45,False
12,Antisaccade,right,412.18,Correct,0.765,8.92,False
13,Prosaccade,right,245.67,Correct,0.891,15.34,False
14,Antisaccade,left,523.89,Error,0.634,6.78,True
15,Prosaccade,left,301.23,Correct,0.798,11.23,False
```

### Data Quality Indicators

**Good Trial:**
- `RT_ms`: 200-500 ms (typical saccade RT)
- `avg_gaze_quality`: > 0.7
- `is_flagged_bad`: False

**Questionable Trial:**
- `RT_ms`: < 150 ms (anticipatory) or > 800 ms (delayed)
- `avg_gaze_quality`: 0.4-0.7
- `is_flagged_bad`: True

**Invalid Trial:**
- `RT_ms`: 1500 ms (no response, timeout)
- `avg_gaze_quality`: < 0.4
- `accuracy`: "Error"

### Post-Processing Recommendations

1. **Filter Flagged Trials:** Remove rows where `is_flagged_bad == True`
2. **RT Outlier Removal:** Exclude trials with RT < 100ms or > 1000ms
3. **Quality Threshold:** Keep only trials with `avg_gaze_quality > 0.6`
4. **Accuracy Analysis:** Compare Prosaccade vs. Antisaccade error rates
5. **RT Analysis:** Compare median RT across conditions

---

## Troubleshooting

### Common Issues

**1. Camera Not Opening**
- **Error:** "Error: Could not open camera"
- **Solution:**
  - Grant camera permissions in System Settings → Privacy & Security → Camera
  - Restart terminal after granting permissions
  - Check if camera is already in use by another application

**2. Poor Tracking Quality (Red Fixation Cross)**
- **Causes:**
  - Backlighting (window behind participant)
  - Too far from camera
  - Glasses reflecting light
  - Unstable head position
- **Solutions:**
  - Adjust lighting (face illumination from front/side)
  - Move closer to camera (arm's length)
  - Remove glasses if possible
  - Use head rest or chin rest for stability

**3. Incorrect Gaze Detection**
- **Symptoms:** Gaze cursor not tracking eye movements accurately
- **Solutions:**
  - Re-run calibration check
  - Ensure eyes are fully visible to camera
  - Check for hair obscuring eyes
  - Ensure proper camera angle (eye level)

**4. Fullscreen Not Working**
- **Symptoms:** Window not occupying full screen
- **Solution:**
  - Press F11 or fn+F11 to toggle fullscreen
  - Check OpenCV installation
  - Try different display if using multiple monitors

---

## Technical Notes

### Coordinate Systems

**Gaze Coordinates (Normalized):**
- Origin: Top-left corner
- Range: [0, 1] for both X and Y
- Center: (0.5, 0.5)
- Right edge: X = 1.0
- Bottom edge: Y = 1.0

**Screen Coordinates (Pixels):**
- Origin: Top-left corner
- Range: [0, screen_width] and [0, screen_height]
- Conversion: `pixel_x = gaze_x × screen_width`

### Timing Precision

**System Clock:**
- Uses `time.perf_counter()` for sub-millisecond precision
- Resolution: Typically ~1 microsecond on modern systems
- Monotonic: Not affected by system time adjustments

**Frame Rate Assumptions:**
- Camera: 30 fps (33.33 ms per frame)
- Display: Varies (60 Hz typical)
- Timing error: ±16-33 ms due to frame sampling

### Performance Optimization

**Recommendations:**
- Close unnecessary applications during experiment
- Disable screen savers and notifications
- Use solid-state storage for faster I/O
- Monitor CPU usage (MediaPipe is computationally intensive)

---

## Research Applications

### Suitable Research Questions

1. **Cognitive Development:** Age-related changes in inhibitory control
2. **Clinical Assessment:** ADHD, Parkinson's, schizophrenia biomarkers
3. **Neurodegenerative Disease:** Early detection of cognitive decline
4. **Executive Function:** Individual differences in cognitive control
5. **Training Effects:** Plasticity of inhibitory mechanisms

### Experimental Design Considerations

**Within-Subjects Design:**
- All participants complete both Prosaccade and Antisaccade trials
- Balanced trial counts (10 per condition)
- Randomized trial order

**Dependent Variables:**
- Primary: Reaction Time (RT), Accuracy
- Secondary: Saccade velocity, quality metrics

**Independent Variables:**
- Trial Type (Prosaccade vs. Antisaccade)
- Target Side (left vs. right)
- Between-subjects factors (e.g., age group, clinical status)

### Statistical Analysis

**Recommended Tests:**
- Paired t-test: Prosaccade vs. Antisaccade RT
- ANOVA: Age group × trial type interaction
- Chi-square: Accuracy rates across conditions
- Linear regression: Quality predictors of RT

**Effect Size Measures:**
- Cohen's d for RT differences
- Odds ratio for accuracy
- Partial eta-squared for ANOVA effects

---

## Version History

**Current Version:** Enhanced (Legacy)
- Full-featured Pro/Antisaccade experiment
- Enhanced UI with real-time feedback
- Quality monitoring and calibration
- Practice trials with feedback

**Related Versions:**
- `eye_experiment.py`: Basic version (minimal UI)
- `eye_experiment_voice.py`: Voice-enabled version (accessibility)
- `eye_oculomotor_tasks.py`: NEW oculomotor assessment system (3 tasks)

---

## References

### MediaPipe Face Mesh
- Google Research: https://google.github.io/mediapipe/solutions/face_mesh
- Landmark documentation: 478 face landmarks including iris tracking
- Model: BlazeFace detector + MediaPipe Face Mesh

### Pro/Antisaccade Literature
- Hallett, P. E. (1978). Primary and secondary saccades to goals defined by instructions. *Vision Research, 18*(10), 1279-1296.
- Munoz, D. P., & Everling, S. (2004). Look away: the anti-saccade task and the voluntary control of eye movement. *Nature Reviews Neuroscience, 5*(3), 218-228.

---

## Contact & Support

For questions, bug reports, or feature requests related to this documentation, please refer to the project repository or contact the development team.

**Last Updated:** February 5, 2026
**Documentation Version:** 1.0
