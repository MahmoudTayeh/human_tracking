# Human Tracking with Face Recognition

Real-time human detection, tracking, and face recognition system using YOLO, DeepSort, and InsightFace for security and surveillance applications.


![insightface_logo jpg_320x320](https://github.com/user-attachments/assets/6508d2de-658b-4059-a8e9-f7064d108a46)

## 🎯 Features

- ✅ **Person Detection** - YOLOv8 for accurate person detection in crowded environments
- ✅ **Multi-Object Tracking** - DeepSort for robust tracking across frames
- ✅ **Face Recognition** - InsightFace for identifying specific individuals
- ✅ **Trajectory Tracking** - Track and visualize movement patterns
- ✅ **Statistics Generation** - Detailed analytics and heatmaps
- ✅ **GPU Accelerated** - CUDA support for real-time performance

## 📊 Demo

### Face Recognition in Action
The system successfully identifies and tracks individuals across video frames:

##### Person Tracking Example 1 !

*Person ID: 6 - Mahmoud tracked with green bounding box*

<img width="336" height="675" alt="image" src="https://github.com/user-attachments/assets/fb8c7fa0-3e11-4d8e-bda2-36acbb8edc18" />

---

##### Person Tracking Example 2 !

*Person ID: 3 - Ahmed tracked and identified*

<img width="357" height="702" alt="image" src="https://github.com/user-attachments/assets/1734adde-a292-46ab-86d3-cf05d9507bd5" />

---

### Technology Stack

![YOLO Detection](docs/yolo_detection.png)
*YOLOv8 detects people with high accuracy*

![InsightFace Recognition](docs/insightface_recognition.png)
*InsightFace provides state-of-the-art face recognition*

## 🔒 Security Importance

This project addresses critical security and safety challenges in modern surveillance systems:

### 🏢 Access Control & Building Security
- **Automated Entry Monitoring**: Identify authorized personnel vs. unauthorized individuals
- **Restricted Area Surveillance**: Alert security when unauthorized persons enter sensitive zones
- **Visitor Tracking**: Monitor and track visitor movements throughout facilities
- **Tailgating Detection**: Identify when multiple people enter using single authorization
- **Real-time Alerts**: Immediate notification when unknown individuals are detected

### 👮 Public Safety & Law Enforcement
- **Suspect Identification**: Automatically identify persons of interest in public spaces
- **Missing Person Search**: Track and locate missing individuals across multiple camera feeds
- **Crowd Monitoring**: Identify VIPs or threats in large gatherings
- **Investigation Support**: Review footage to track suspect movements and identify associates
- **Proactive Threat Detection**: Identify known threats before incidents occur

### 🏪 Retail & Commercial Security
- **Shoplifting Prevention**: Track known shoplifters entering stores
- **VIP Customer Recognition**: Identify high-value customers for personalized service
- **Employee Monitoring**: Track employee movements in restricted areas
- **Loss Prevention**: Monitor high-theft areas and identify repeat offenders
- **Customer Analytics**: Understand customer flow and behavior patterns

### 🏠 Smart Home & Residential Security
- **Family Member Recognition**: Distinguish between family members and strangers
- **Elderly Care Monitoring**: Track elderly family members for safety
- **Child Safety**: Alert when unknown individuals approach children
- **Package Theft Prevention**: Identify delivery persons vs. potential thieves
- **Neighborhood Watch**: Share alerts about suspicious individuals across community

### 🎓 Educational Institution Security
- **Campus Safety**: Monitor who enters school grounds
- **Student Attendance**: Automated attendance tracking via face recognition
- **Unauthorized Visitor Detection**: Alert when non-students/staff enter premises
- **Emergency Response**: Quickly locate specific individuals during emergencies
- **Anti-bullying Monitoring**: Track interactions in common areas

### 🏥 Healthcare Facility Security
- **Patient Elopement Prevention**: Alert when at-risk patients leave designated areas
- **Infant Abduction Prevention**: Track authorized vs. unauthorized individuals in maternity wards
- **Controlled Substance Areas**: Monitor who accesses medication storage areas
- **Visitor Management**: Track visitor movements in sensitive patient areas
- **Staff Accountability**: Monitor staff presence in critical care areas

### 🏭 Industrial & Workplace Safety
- **Safety Compliance**: Ensure only trained personnel enter hazardous areas
- **Accident Investigation**: Review who was present during workplace incidents
- **Contractor Management**: Track contractor movements in secure facilities
- **Time & Attendance**: Automated employee tracking and attendance
- **Workplace Violence Prevention**: Identify banned individuals attempting entry

### 🚨 Key Security Benefits

1. **Proactive Threat Detection**
   - Identify potential threats before incidents occur
   - Real-time alerts when flagged individuals are detected
   - Reduce response time from minutes to seconds

2. **Enhanced Situational Awareness**
   - Know who is where at any given time
   - Track movement patterns to identify suspicious behavior
   - Multi-camera coordination for comprehensive coverage

3. **Investigation Efficiency**
   - Quickly search hours of footage for specific individuals
   - Track suspect movements across multiple locations
   - Generate reports on person's activities and associates

4. **Access Control Automation**
   - Reduce reliance on physical access cards
   - Prevent unauthorized access through tailgating
   - Automatically revoke access for terminated employees

5. **Data-Driven Insights**
   - Understand traffic patterns and peak times
   - Identify security vulnerabilities
   - Optimize security resource allocation

6. **Privacy Protection**
   - Only store face embeddings, not actual photos
   - Configurable data retention periods
   - Audit trails for compliance

### ⚠️ Ethical Considerations

This technology must be used responsibly:

- ✅ **Transparency**: Inform people when face recognition is in use
- ✅ **Consent**: Obtain consent where legally required
- ✅ **Data Protection**: Secure storage and encryption of biometric data
- ✅ **Limited Use**: Use only for stated security purposes
- ✅ **Accuracy**: Regularly test and validate recognition accuracy
- ✅ **Bias Mitigation**: Ensure system works equally across demographics
- ✅ **Oversight**: Human review of automated decisions
- ✅ **Compliance**: Follow GDPR, CCPA, and local privacy laws

### 🎯 Use Case Examples

**Scenario 1: Corporate Office Security**
```
Problem: Unauthorized access to server room
Solution: System alerts when non-IT personnel enter server area
Result: 100% prevention of unauthorized access attempts
```

**Scenario 2: School Campus Safety**
```
Problem: Stranger entered school grounds
Solution: System identified unknown individual, alerted security
Result: Security responded in 30 seconds vs. 15 minutes
```

**Scenario 3: Retail Loss Prevention**
```
Problem: Known shoplifter entered store
Solution: System recognized individual from database, alerted staff
Result: Prevented $500 theft, led to arrest
```

**Scenario 4: Hospital Patient Safety**
```
Problem: Dementia patient wandered from secure ward
Solution: System tracked patient movement, alerted staff
Result: Patient safely returned before reaching exit
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)
- Webcam (for face registration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mahmoudfarajtayeh/human-tracking.git
   cd human-tracking
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_insightface.txt
   ```

4. **Download YOLO model** (auto-downloads on first run)
   ```bash
   # The model will download automatically when you run the program
   # Or manually place yolov8x.pt in data/models/
   ```

### Setup Face Recognition

1. **Create directories**
   ```bash
   mkdir -p data/known_faces
   mkdir -p data/videos
   mkdir -p outputs/videos
   ```

2. **Register authorized personnel**
   ```bash
   # From webcam (recommended)
   python register_faces.py --mode webcam --name "Your Name" --num-photos 5
   
   # Or from images
   python register_faces.py --mode images --name "Your Name" --images data/known_faces/your_name/
   ```

3. **Run tracking**
   ```bash
   python main.py --config configs/config.yaml
   ```

## 📁 Project Structure

```
human-tracking/
├── configs/
│   └── config.yaml              # Configuration settings
├── data/
│   ├── models/                  # YOLO models (auto-downloaded)
│   ├── videos/                  # Input videos (add your own)
│   └── known_faces/             # Authorized personnel photos
├── outputs/
│   ├── videos/                  # Processed output videos
│   ├── statistics/              # Generated statistics
│   └── visualizations/          # Heatmaps and trajectories
├── src/
│   ├── __init__.py
│   ├── detector.py              # YOLO person detector
│   ├── tracker.py               # DeepSort tracker with face recognition
│   ├── face_recognizer.py       # InsightFace recognition engine
│   ├── visualizer.py            # Visualization utilities
│   └── utils.py                 # Helper functions
├── main.py                      # Main application
├── register_faces.py            # Face registration tool
├── requirements_insightface.txt # Dependencies
└── README.md                    # This file
```

## ⚙️ Configuration

Edit `configs/config.yaml` to customize:

```yaml
# YOLO Detection
model:
  yolo_model: "data/models/yolov8x.pt"
  confidence_threshold: 0.5
  device: "cuda"  # or "cpu"

# DeepSort Tracking
tracker:
  max_age: 30
  n_init: 3

# Face Recognition
face_recognition:
  enabled: true
  similarity_threshold: 0.4      # Lower = stricter (0.3-0.5)
  recognition_interval: 10       # Run every N frames
```

## 🎬 Usage

### Basic Tracking
```bash
python main.py --config configs/config.yaml
```

### Register New Person
```bash
# From webcam
python register_faces.py --mode webcam --name "John Doe" --num-photos 5

# From images
python register_faces.py --mode images --name "John Doe" --images data/known_faces/john/
```

### List Registered People
```bash
python register_faces.py --mode list
```

### Remove Person
```bash
python register_faces.py --mode remove --name "John Doe"
```

## 📊 Output

The system generates:
- **Tracked videos** - Videos with bounding boxes and names
- **Statistics** - Duration, distance traveled, speed per person
- **Heatmaps** - Movement density visualization
- **Trajectories** - Path visualization for each tracked person

## 🔧 Troubleshooting

### CUDA Not Available
If you see "CUDAExecutionProvider not available":
```bash
pip install onnxruntime-gpu
```

### Face Not Recognized
- Add more training photos (10+ recommended)
- Adjust `similarity_threshold` in config (try 0.45)
- Ensure good lighting during registration

### Slow Performance
- Increase `recognition_interval` to 20
- Use smaller YOLO model (yolov8n.pt instead of yolov8x.pt)
- Reduce video resolution

## 📈 Performance

**With NVIDIA GPU:**
- Detection + Tracking + Face Recognition: ~25 FPS
- 1080p video processing

**CPU Only:**
- Detection + Tracking + Face Recognition: ~7 FPS
- 720p video recommended

## 🛠️ Tech Stack

- **Detection:** YOLOv8 (Ultralytics)
- **Tracking:** DeepSort
- **Face Recognition:** InsightFace (buffalo_l model)
- **Computer Vision:** OpenCV
- **Deep Learning:** PyTorch, ONNX Runtime

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

**Mahmoud Faraj Tayeh**  
Email: mahmodtayh2003@gmail.com

Project Link: [https://github.com/mahmoudfarajtayeh/human-tracking](https://github.com/mahmoudfarajtayeh/human-tracking)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - State-of-the-art object detection
- [DeepSort](https://github.com/nwojke/deep_sort) - Simple Online and Realtime Tracking
- [InsightFace](https://github.com/deepinsight/insightface) - 2D and 3D Face Analysis Project

---

**⚠️ Responsible Use Notice:** This technology should be used in compliance with local laws and regulations regarding surveillance and biometric data. Always respect individual privacy rights and obtain necessary permissions before deployment.
