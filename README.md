# Eye-Tracking Experiment: Pro/Antisaccade Task

A Python-based webcam eye-tracking system for conducting cognitive experiments using the Pro/Antisaccade paradigm. This project provides real-time gaze tracking, experiment administration, and comprehensive data analysis tools.

## Overview

This project implements a complete eye-tracking experiment pipeline:
1. **Webcam-based eye tracking** using MediaPipe Face Mesh
2. **Pro/Antisaccade cognitive task** with calibration and practice trials
3. **Automated data collection** with quality control metrics
4. **Statistical analysis and visualization** of reaction times and accuracy

## Features

- Real-time eye tracking using webcam (no special hardware required)
- Two experiment versions: basic and enhanced with user-friendly interface
- Automatic quality control and trial flagging
- Practice trials with immediate feedback
- Calibration system with visual quality indicators
- Automated rest breaks during long experiments
- Comprehensive statistical analysis and visualization

## Quick Start

### Prerequisites

- Python 3.8-3.12 (MediaPipe does not support Python 3.13+)
- Mac with embedded camera (or any system with webcam)
- ~10 minutes for full experiment

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd EyeTracker_Project
```

2. Create and activate virtual environment:
```bash
# If you have Python 3.13+, use Python 3.12
brew install python@3.12
python3.12 -m venv venv

# Or use your default Python (if 3.8-3.12)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Experiment

#### Enhanced Version (Recommended)
```bash
python3.12 eye_experiment_enhanced.py
```

Features:
- Pre-experiment calibration with quality validation
- Visual instruction screens with task demonstrations
- Practice trial feedback (Correct/Incorrect)
- Progress tracking and completion percentage
- Automatic rest breaks every 10 trials
- Color-coded quality indicators

#### Basic Version
```bash
python3.12 eye_experiment.py
```

Minimal interface for experienced users or research purposes.

**Output**: `eye_tracking_results_DATA.csv`

### Analyzing Data

```bash
python3.12 eye_analysis_dashboard.py
```

**Input**: `eye_tracking_results_DATA.csv`
**Output**: `eye_tracking_analysis_dashboard.png` (3-panel visualization)

## Project Structure

```
EyeTracker_Project/
├── eye_experiment.py              # Basic experiment implementation
├── eye_experiment_enhanced.py     # Enhanced version with better UX
├── eye_analysis_dashboard.py      # Data analysis and visualization
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── eye_tracking_results_DATA.csv # Experiment output (generated)
```

## Understanding the Task

### Prosaccade Trials
- **Instruction**: "Look toward the target when it appears"
- **Measures**: Basic visual-motor reaction time

### Antisaccade Trials
- **Instruction**: "Look away from the target (opposite direction)"
- **Measures**: Inhibitory control and executive function

### Trial Structure
1. **Fixation** (1000ms): Maintain gaze at center cross
2. **Target** (1500ms): Target appears on left or right
3. **Inter-trial Interval** (500ms): Brief pause

### Experiment Structure
- 10 practice trials with feedback
- 40 test trials (10 per condition: 2 trial types × 2 target sides)
- Optional rest breaks every 10 trials (enhanced version)

## Data Output

The experiment generates `eye_tracking_results_DATA.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `trial_id` | Sequential trial number |
| `trial_type` | "Prosaccade" or "Antisaccade" |
| `target_side` | "left" or "right" |
| `RT_ms` | Reaction time in milliseconds |
| `accuracy` | "Correct" or "Error" |
| `avg_gaze_quality` | Mean quality score (0-1) |
| `saccade_velocity` | Eye movement velocity |
| `is_flagged_bad` | Quality control flag |

## Analysis Dashboard

The analysis script generates three visualizations:

1. **Mean Reaction Time**: Comparison across conditions
2. **Mean Accuracy**: Performance metrics by trial type
3. **Quality Control**: RT vs. velocity scatter plot with flagged trials

The analyzer includes age group simulation (Teenager vs. Older Adult) to demonstrate expected developmental differences.

## Requirements

```
opencv-python>=4.8.0    # Camera capture and display
mediapipe>=0.10.0       # Face mesh and iris tracking
pandas>=2.0.0           # Data manipulation
numpy>=1.24.0           # Numerical operations
matplotlib>=3.7.0       # Plotting
seaborn>=0.12.0         # Statistical visualizations
```

## Technical Details

### Eye Tracking System
- **Technology**: MediaPipe Face Mesh with iris landmark detection
- **Landmarks**: LEFT_IRIS [474-477], RIGHT_IRIS [469-472]
- **Coordinate System**: Normalized (0-1), center = (0.5, 0.5)
- **Quality Metric**: Based on eye openness and detection confidence

### Timing Precision
- Uses `time.perf_counter()` for high-resolution timing
- Assumes ~30 fps camera frame rate
- RT calculated from target onset to first gaze departure

### Quality Control
- Trials flagged if gaze quality < 0.7 during saccade execution
- Quality checked within ±5 frames of saccade onset
- Flagged trials excluded from analysis

## Customization

### Changing Trial Parameters

Edit experiment initialization (both versions):
```python
self.practice_trials = 10      # Number of practice trials
self.test_trials = 40          # Number of test trials
self.fixation_duration = 1000  # Milliseconds
self.target_duration = 1500    # Milliseconds
self.fixation_box_size = 0.15  # Normalized (0-1)
self.target_position = 0.35    # Distance from center
```

### Adjusting Quality Threshold

Modify in experiment files:
```python
if critical_window and min(critical_window) < 0.7:  # Change 0.7
    is_flagged_bad = True
```

## Troubleshooting

### Camera Not Detected
- Ensure camera permissions are granted to Terminal
- Try restarting the application
- Check `cv2.VideoCapture(0)` captures correctly

### MediaPipe Import Error
- Verify Python version is 3.8-3.12 (not 3.13+)
- Reinstall dependencies: `pip install -r requirements.txt`

### Poor Tracking Quality
- Ensure good lighting conditions
- Position face 1-2 feet from camera
- Run calibration phase carefully (enhanced version)
- Minimize head movement during trials

### Low Frame Rate
- Close other applications using camera
- Ensure sufficient system resources
- Consider reducing resolution if needed

## Use Cases

- Cognitive psychology research
- Developmental studies (attention and inhibitory control)
- Clinical assessments (ADHD, executive function)
- Educational demonstrations of eye-tracking methods
- Human-computer interaction research

## Contributing


## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```
[Add citation information here]
```

## Contact

[Eric Chen happyeric2015@gmail.com]

## Acknowledgments

- MediaPipe by Google for face mesh tracking
- OpenCV for computer vision tools
- Participants who helped test and validate the system
