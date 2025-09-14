# Vitruvian Whole-Body Creative Action

**A Proposal for the Struck-String Interaction Framework**

*By Sutirtha Chakraborty, Damián Keller, Joseph Timoney*

---

## 🎭 Overview

This repository contains the implementation of **Vitruvian Whole-Body Creative Action**, an innovative musical interface that transforms human body movements into expressive piano-like sounds. Inspired by Leonardo da Vinci's *The Vitruvian Man*, our system uses computer vision and pose detection to create an accessible, browser-based musical experience that requires no physical instruments or specialized hardware.

**Conference Presentation**: [UbiMus 2025](https://www.th-brandenburg.de/forschung-kooperation/projekte/ubimus/) - The Ubiquitous Music Symposium  
**Dates**: September 15-17, 2025  
**Location**: Brandenburg University of Applied Sciences (THB), Germany

## 🚀 Key Features

- **🎹 Struck-String Interaction Framework**: Revolutionary approach to piano sound control without traditional keyboards
- **📹 Camera-Based Pose Detection**: Uses MediaPipe for real-time full-body tracking via webcam
- **🎵 Multi-Zone Musical Mapping**: Vitruvian-inspired geometric zones for different musical expressions
- **🌐 Cross-Platform Accessibility**: Browser-based (HTML/JS) and Python implementations
- **🎨 Real-Time Visual Feedback**: Dynamic geometric overlays showing interaction zones
- **🔄 Horizontal Camera Mirroring**: Natural, mirror-like interaction experience

## 🏛️ The Vitruvian Concept

Drawing inspiration from Da Vinci's masterpiece, our system creates a **Vitruvian Creative-Action Metaphor** where:

- **Square Container**: Represents the structural framework of musical interaction
- **Inner Circle Zone**: Triggers discrete piano strums and close-range interactions
- **Outer Circle Zone**: Enables continuous synth pads and sustained musical expressions
- **Body Proportions**: Dynamically adapt the interface to each performer's physique

## 🎵 Musical Interaction Zones

### 1. **Arms & Wrists** → Synth/Warm Pads (Channel 1)
- Movement triggers expressive synthesizer sounds
- Velocity-sensitive response based on movement speed
- Continuous modulation available in outer zones

### 2. **Feet** → Piano Strums (Channel 2)
- Foot movements trigger piano-like percussive sounds
- Direction changes create musical note onsets
- Supports both left and right foot tracking

### 3. **Right Index Finger** → Multi-Modal Control
- **Far Position (Outside Red Circle)**: Continuous synth notes
- **Close Position (Inside Green Circle)**: Piano strum triggers
- **Middle Position (Between Circles)**: Piano arpeggios
- **Body Line Crossing**: Triggers based on crossing imaginary line from shoulder to hip

## 📁 Repository Structure

```
├── main.py                      # Python implementation with MIDI output
├── index.html                   # Alternative web interface
├── paper.txt                    # Research paper content
└── README.md                    # This file
```

## 🌐 Web Implementation (Recommended)

### Features
- **Zero Installation**: Works directly in modern web browsers
- **Web Audio API**: Real-time audio synthesis without external dependencies
- **MediaPipe Integration**: Advanced pose detection via CDN
- **Responsive Design**: Professional UI with glass morphism effects
- **Cross-Platform**: Works on desktop, tablet, and mobile devices

### Quick Start
```bash
# Simply open in a web browser
open vitruvian-performer.html
# or serve via local server for HTTPS (required for camera access)
python -m http.server 8000
# Then visit: http://localhost:8000/vitruvian-performer.html
```

### Browser Requirements
- Modern browser with WebRTC support (Chrome, Firefox, Safari, Edge)
- Camera permissions for pose detection
- Microphone permissions for audio output

## 🐍 Python Implementation

### Features
- **MIDI Output**: Professional MIDI integration using `mido` library
- **Virtual MIDI Ports**: Compatible with DAWs and external synthesizers
- **OpenCV Processing**: Direct camera feed processing
- **Multi-Channel Routing**: Separate MIDI channels for different instrument types

### Installation
```bash
# Install Python dependencies
pip install -r requirements.txt

# For macOS: Enable IAC Driver in Audio MIDI Setup
# For Windows: Install loopMIDI virtual MIDI driver
# For Linux: Use ALSA/JACK MIDI bridge
```

### Dependencies
```python
opencv-python>=4.5.0
mediapipe>=0.8.0
mido>=1.2.0
numpy>=1.21.0
```

### Usage
```bash
python main.py
```

### MIDI Configuration
The system creates a virtual MIDI port "PythonMIDI" with the following channel mapping:
- **Channel 1**: Synth/Warm Pads (Program 90)
- **Channel 2**: Piano Strums (Program 1)
- **Channel 3**: Piano Arpeggios (Program 1)

## 🎼 Musical Scales & Chords

The system includes pre-configured musical scales and chord progressions:

### Available Scales
- **Am (A Minor)**: [A3, C4, D4, E4, F4, G4, A4, C5]
- **C (C Major)**: [C4, D4, E4, F4, G4, A4, B4, C5]
- **G (G Major)**: [G3, A3, B3, C4, D4, E4, F#4, G4]
- **F (F Major)**: [F3, G3, A3, Bb3, C4, D4, E4, F4]
- **Em (E Minor)**: [E3, G3, A3, B3, C4, D4, E4, G4]

### Chord Progressions
- **Am**: [C4, E4, G4]
- **C**: [C4, E4, A4]
- **G**: [G3, B3, D4]
- **F**: [F3, A3, C4]
- **Em**: [E3, G3, B3]

## 🔧 Technical Architecture

### Real-Time Processing Pipeline
1. **Video Input**: Webcam capture at 640x480 resolution
2. **Pose Detection**: MediaPipe Holistic model with 0.6 confidence threshold
3. **Movement Tracking**: Custom `MovementTracker` class with circular buffers
4. **Gesture Recognition**: Direction change detection and velocity calculation
5. **Musical Mapping**: MIDI/Audio event generation based on body positions
6. **Visual Feedback**: Real-time overlay graphics and color-coded zones

### Performance Optimizations
- **Smoothed Movement**: 10-15 frame buffers for noise reduction
- **Trigger Debouncing**: Minimum 250ms intervals between note triggers
- **Velocity Mapping**: Dynamic scaling based on movement speed
- **Zone Calculations**: Efficient geometric computations for real-time response

## 🎨 Visual Design

### Professional UI Elements
- **Glass Morphism**: Advanced backdrop blur effects
- **Gradient Typography**: Gold gradient text with professional shadows
- **Interactive Elements**: Hover effects and smooth transitions
- **Responsive Layout**: Mobile-friendly design with flexible controls
- **Status Panels**: Real-time feedback on detection and musical activity

### Color Coding System
- **Red Zones**: Outer circle boundaries for continuous control
- **Green Zones**: Inner circle boundaries for discrete triggers
- **Gold Accents**: UI highlights and active elements
- **White Overlays**: Pose landmarks and connection lines

## 📚 Research Context

This work contributes to the **Ubiquitous Music (UbiMus)** research field, specifically addressing:

### Struck-String Interaction Framework
- Decoupling piano sounds from traditional keyboard interfaces
- Reducing cognitive load and temporal investment for musical expression
- Enabling parametric control of piano-like sonic resources
- Supporting both local and remote musical experiences

### Accessibility & Inclusion
- **Zero Hardware Requirements**: Works with standard webcam
- **Intuitive Interaction**: Natural body movements as musical input
- **Scalable Complexity**: Simple gestures to complex musical expressions
- **Cross-Platform Availability**: Browser-based deployment

### Technical Innovation
- **Real-Time Pose Detection**: MediaPipe integration for robust tracking
- **Adaptive Proportions**: Dynamic scaling based on body measurements
- **Multi-Modal Feedback**: Visual, auditory, and proprioceptive cues
- **Sustainable Infrastructure**: Browser-native technologies

## 🎯 Getting Started

### For Performers
1. Open `vitruvian-performer.html` in a modern web browser
2. Allow camera permissions when prompted
3. Click "Start Performance" to begin tracking
4. Move your body within the camera frame to create music
5. Experiment with different zones and movement styles

### For Developers
1. Clone this repository
2. Choose your preferred implementation (Web or Python)
3. Follow the installation instructions above
4. Explore the code to understand the interaction mappings
5. Customize scales, chords, or visual elements as needed

### For Researchers
1. Review the `paper.txt` for detailed technical documentation
2. Examine the movement tracking algorithms in the source code
3. Consider adaptations for your specific research context
4. Reference our work in related ubiquitous music research

## 🔬 Research Applications

### Potential Extensions
- **Machine Learning Integration**: Adaptive gesture recognition
- **Collaborative Performance**: Multi-user simultaneous interaction
- **Accessibility Features**: Support for users with different physical abilities
- **VR/AR Integration**: Immersive three-dimensional interaction spaces
- **Educational Tools**: Music therapy and pedagogical applications

### Evaluation Metrics
- **Latency Measurements**: End-to-end response time analysis
- **Accuracy Assessment**: Pose detection reliability studies
- **User Experience**: Qualitative feedback on interaction design
- **Musical Expression**: Quantitative analysis of generated performances

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{chakraborty2025vitruvian,
  title={Vitruvian Whole-Body Creative Action: A Proposal for the Struck-String Interaction Framework},
  author={Chakraborty, Sutirtha and Keller, Damián and Timoney, Joseph},
  booktitle={Proceedings of the Ubiquitous Music Symposium 2025},
  year={2025},
  organization={Brandenburg University of Applied Sciences}
}
```

## 🤝 Contributing

We welcome contributions from the ubiquitous music community:

1. **Bug Reports**: Use GitHub issues for technical problems
2. **Feature Requests**: Suggest new interaction modalities or improvements
3. **Code Contributions**: Submit pull requests with enhancements
4. **Documentation**: Help improve setup guides and tutorials
5. **Research Collaboration**: Contact authors for academic partnerships

## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- **UbiMus Community**: For fostering innovation in ubiquitous music research
- **MediaPipe Team**: For providing robust pose detection capabilities
- **Brandenburg University**: For hosting the UbiMus 2025 symposium
- **Da Vinci**: For the timeless inspiration of human proportions and creativity

## 📞 Contact

**Sutirtha Chakraborty**  
📧 [sutirtha38@gmail.com]  


---

### 🎵 "Where human movement meets musical expression, the Vitruvian performer emerges."

*Presented at UbiMus 2025 - The Ubiquitous Music Symposium*  
*September 15-17, 2025 | Brandenburg, Germany*
