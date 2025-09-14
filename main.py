"""
Vitruvian Body-Touch and Expressive MIDI Performer
----------------------------------------------------
This program uses a webcam and MediaPipe to track your full‐body pose.
It triggers different MIDI instruments based on your movements:
  • Arms/wrists trigger synth/warm pad sounds on Channel 1.
  • Feet trigger piano “strums” on Channel 2.
  • When your right‐hand index finger crosses the imaginary line from your
    right shoulder to your right hip, its distance from your body center
    selects among three modes:
      - If the finger is far (outside the outer circle): a continuous
        synth/warm pad note on Channel 1.
      - If the finger is very close (inside the inner circle): a piano strum
        on Channel 2.
      - If the finger is at an intermediate distance (between inner and outer
        circles): a piano arpeggio on Channel 3.
Visual feedback is provided by drawing a Vitruvian “container” (a square) and
two circles (inner and outer zones) plus on‐screen labels—including a color
panel that changes based on the octave of the triggered note.
"""

import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque
import mido
from mido import Message
from threading import Thread
import random
import time

# -------------------------
# MIDI Setup
# -------------------------
try:
    midi_out = mido.open_output("PythonMIDI", virtual=True)
    print(f"MIDI Output opened: {midi_out.name}")
except IOError:
    print("Error: Could not open MIDI port 'PythonMIDI'")
    print("For macOS: Use IAC Driver | Windows: Use loopMIDI")
    exit(1)

# NOTE: mido channels are 0-indexed.
# Channel mapping:
#   Channel 1 (mido channel 0) => Synth / Warm Pads
#   Channel 2 (mido channel 1) => Piano Strums
#   Channel 3 (mido channel 2) => Piano Arpeggios
midi_out.send(
    Message("program_change", program=90, channel=0)
)  # Synth/warm pads on Ch 1
midi_out.send(Message("program_change", program=1, channel=1))  # Piano (strums) on Ch 2
midi_out.send(
    Message("program_change", program=1, channel=2)
)  # Piano (arpeggios) on Ch 3

# -------------------------
# Musical scales and chords definitions
# -------------------------
SCALES = {
    "Am": [57, 60, 62, 64, 65, 67, 69, 72],
    "C": [60, 62, 64, 65, 67, 69, 71, 72],
    "G": [55, 57, 59, 60, 62, 64, 66, 67],
    "F": [53, 55, 57, 58, 60, 62, 64, 65],
    "Em": [52, 55, 57, 59, 60, 62, 64, 67],
}
CHORDS = {
    "Am": [60, 64, 67],
    "C": [60, 64, 69],
    "G": [55, 59, 62],
    "F": [53, 57, 60],
    "Em": [52, 55, 59],
}
current_chord = "Am"  # Changeable via keyboard input

# -------------------------
# Global variables for note tracking (for color panels & continuous note)
# -------------------------
continuous_synth_active = False
continuous_synth_note = None
last_piano_strum_note = None
last_arpeggio_notes = None


# -------------------------
# Helper: Map a MIDI note to a color (B,G,R)
# -------------------------
def get_color_for_note(note):
    octave = note // 12
    if octave < 4:
        return (255, 0, 0)  # Blue-ish for lower octaves
    elif octave == 4:
        return (0, 255, 0)  # Green
    elif octave == 5:
        return (0, 0, 255)  # Red
    else:
        return (0, 255, 255)  # Yellowish for high octaves


# -------------------------
# MIDI utility functions
# -------------------------
def send_midi_note_on(note, velocity, channel=0):
    midi_out.send(Message("note_on", note=note, velocity=velocity, channel=channel))


def send_midi_note_off(note, velocity=0, channel=0):
    midi_out.send(Message("note_off", note=note, velocity=velocity, channel=channel))


def send_midi_cc(controller, value, channel=0):
    midi_out.send(
        Message("control_change", channel=channel, control=controller, value=value)
    )


def stop_all_midi_notes():
    for channel in range(3):  # channels 0,1,2
        for note in range(128):
            send_midi_note_off(note, channel=channel)


def play_note_with_articulation(note, velocity, channel, duration, extra_ccs=None):
    """Send a note on then off after a duration, with optional extra CC messages."""
    send_midi_note_on(note, velocity, channel=channel)
    if extra_ccs:
        for ctrl, value in extra_ccs.items():
            send_midi_cc(ctrl, value, channel=channel)

    def note_off_after_delay():
        time.sleep(duration)
        send_midi_note_off(note, channel=channel)

    Thread(target=note_off_after_delay).start()


# -------------------------
# Trigger functions for discrete events
# -------------------------
def trigger_movement_note(velocity, vitruvian_size, channel, octave_shift=0):
    """Choose a random note from the current scale, apply an octave shift,
    and play it with extra CC messages. Returns (note, velocity, channel)."""
    note = random.choice(SCALES[current_chord])
    note += 12 * octave_shift
    note = max(0, min(127, note))
    reverb_amount = int(min(max(vitruvian_size * 127, 20), 127))
    modulation = int(velocity * 0.7)
    extra_ccs = {91: reverb_amount, 1: modulation}  # CC 91: Reverb, CC 1: Modulation
    duration = 0.2 + (vitruvian_size * 0.5)
    play_note_with_articulation(note, velocity, channel, duration, extra_ccs)
    return note, velocity, channel


def play_arpeggio(chord_name, velocity, channel, vitruvian_size):
    """Play an arpeggio built from the current chord on the given channel.
    Returns (chord_notes, velocity, channel)."""
    chord_notes = CHORDS.get(chord_name, [60, 64, 67])
    reverb_amount = int(min(max(vitruvian_size * 127, 20), 127))
    modulation = int(velocity * 0.7)
    extra_ccs = {91: reverb_amount, 1: modulation}

    def arpeggio_thread():
        for note in chord_notes:
            play_note_with_articulation(
                note, velocity, channel, duration=0.15, extra_ccs=extra_ccs
            )
            time.sleep(0.1)

    Thread(target=arpeggio_thread).start()
    return chord_notes, velocity, channel


# -------------------------
# Movement Tracker Class
# -------------------------
class MovementTracker:
    def __init__(self, label="", buffer_size=10):
        self.positions = deque(maxlen=buffer_size)
        self.last_direction = None
        self.last_trigger_time = 0
        self.min_trigger_interval = 0.25  # seconds
        self.direction_threshold = 0.01
        self.label = label

    def update(self, position):
        self.positions.append(position)

    def get_smoothed_speed(self):
        if len(self.positions) < 2:
            return 0
        diffs = [
            abs(self.positions[i + 1].y - self.positions[i].y)
            for i in range(len(self.positions) - 1)
        ]
        return sum(diffs) / len(diffs)

    def get_direction(self):
        if len(self.positions) < 3:
            return None
        recent = list(self.positions)[-3:]
        movement = (recent[-1].y - recent[0].y) / 2
        if abs(movement) < self.direction_threshold:
            return None
        return "up" if movement < 0 else "down"

    def should_trigger(self):
        if len(self.positions) < 3:
            return False
        current_time = time.time()
        if current_time - self.last_trigger_time < self.min_trigger_interval:
            return False
        current_direction = self.get_direction()
        if current_direction is None:
            return False
        if self.last_direction is not None and current_direction != self.last_direction:
            self.last_direction = current_direction
            self.last_trigger_time = current_time
            return True
        self.last_direction = current_direction
        return False

    def get_velocity(self):
        speed = self.get_smoothed_speed()
        velocity = int(min(max(speed * 3000, 40), 127))
        return velocity


# Instantiate trackers:
left_foot_tracker = MovementTracker(label="Left Foot", buffer_size=15)
right_foot_tracker = MovementTracker(label="Right Foot", buffer_size=15)
left_wrist_tracker = MovementTracker(label="Left Wrist")
right_wrist_tracker = MovementTracker(label="Right Wrist")


# -------------------------
# Vitruvian Visuals Drawing Function
# -------------------------
def draw_vitruvian_elements(image, pose_landmarks, face_landmarks=None):
    if not pose_landmarks:
        return None
    h, w = image.shape[:2]
    # Use face landmark if available; otherwise, fallback to nose
    if face_landmarks:
        head_point = face_landmarks.landmark[10]
    else:
        head_point = pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE]
    left_foot = pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_FOOT_INDEX]
    right_foot = pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX]
    lowest_foot_y = max(left_foot.y, right_foot.y)
    center_x = int(((left_foot.x + right_foot.x) / 2) * w)
    center_y = int(((head_point.y + lowest_foot_y) / 2) * h)
    body_height = abs(head_point.y - lowest_foot_y) * h
    square_size = int(body_height)
    circle_radius = square_size // 2  # inscribed circle (not used for zones now)
    # Dynamic color base (can be blended later)
    max_dimension = math.hypot(w, h)
    vitruvian_size = body_height / max_dimension
    base_intensity = int(min(max(vitruvian_size * 255 * 2, 50), 255))
    base_circle_color = (50, base_intensity, 255 - base_intensity)
    base_square_color = (255 - base_intensity, 50, base_intensity)
    line_thickness = int(max(1, vitruvian_size * 10))
    # Draw square (container)
    center = (center_x, center_y)
    top_left = (center_x - square_size // 2, center_y - square_size // 2)
    bottom_right = (center_x + square_size // 2, center_y + square_size // 2)
    cv2.rectangle(image, top_left, bottom_right, base_square_color, line_thickness)
    # Label the square
    cv2.putText(
        image,
        "Structure",
        (top_left[0], top_left[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        base_square_color,
        2,
    )
    # (We will draw our zone circles later in process_camera.)
    cv2.putText(
        image,
        f"Vitruvian Size: {vitruvian_size:.2f}",
        (10, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (240, 240, 240),
        2,
    )
    return {
        "vitruvian_size": vitruvian_size,
        "center": center,
        "square_top_left": top_left,
        "square_bottom_right": bottom_right,
        "square_size": square_size,
        "base_circle_color": base_circle_color,
        "base_square_color": base_square_color,
        "line_thickness": line_thickness,
    }


# -------------------------
# Utility function to compute which side of a line a point lies on
# -------------------------
def point_line_side(A, B, P):
    return (B.x - A.x) * (P.y - A.y) - (B.y - A.y) * (P.x - A.x)


# -------------------------
# Draw a color panel to indicate current note color per instrument.
# -------------------------
def draw_color_panel(image):
    h, w = image.shape[:2]
    panel_x1 = w - 110
    panel_x2 = w - 10
    # Synth (Channel 1)
    if continuous_synth_active and continuous_synth_note is not None:
        synth_color = get_color_for_note(continuous_synth_note)
        cv2.rectangle(image, (panel_x1, 10), (panel_x2, 60), synth_color, -1)
        cv2.putText(
            image,
            "Synth",
            (panel_x1 + 5, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    # Piano Strum (Channel 2)
    if last_piano_strum_note is not None:
        piano_color = get_color_for_note(last_piano_strum_note)
        cv2.rectangle(image, (panel_x1, 70), (panel_x2, 120), piano_color, -1)
        cv2.putText(
            image,
            "P.Strum",
            (panel_x1 + 5, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )
    # Piano Arpeggio (Channel 3)
    if last_arpeggio_notes is not None and len(last_arpeggio_notes) > 0:
        arp_color = get_color_for_note(last_arpeggio_notes[0])
        cv2.rectangle(image, (panel_x1, 130), (panel_x2, 180), arp_color, -1)
        cv2.putText(
            image,
            "Arpeggio",
            (panel_x1 + 5, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )


# -------------------------
# Main Camera Processing Function
# -------------------------
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# For index finger zone tracking
prev_index_side = None
prev_index_point = None


def process_camera():
    global prev_index_side, prev_index_point, current_chord
    global continuous_synth_active, continuous_synth_note, last_piano_strum_note, last_arpeggio_notes

    camera_active = True
    cap = cv2.VideoCapture(0)  # Use live webcam

    with mp_holistic.Holistic(
        min_detection_confidence=0.6, min_tracking_confidence=0.6
    ) as holistic:
        while camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror view and convert color space
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image_rgb)
            image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            vitruvian_info = None
            if results.pose_landmarks:
                vitruvian_info = draw_vitruvian_elements(
                    image, results.pose_landmarks, results.face_landmarks
                )
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
                )

                # -------------------------
                # Process Feet Movements (Piano Strums on Ch2)
                # -------------------------
                left_foot = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.LEFT_FOOT_INDEX
                ]
                right_foot = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.RIGHT_FOOT_INDEX
                ]
                left_foot_tracker.update(left_foot)
                right_foot_tracker.update(right_foot)
                for tracker in [left_foot_tracker, right_foot_tracker]:
                    if tracker.should_trigger():
                        velocity = tracker.get_velocity()
                        foot_px = (
                            int(tracker.positions[-1].x * image.shape[1]),
                            int(tracker.positions[-1].y * image.shape[0]),
                        )
                        cv2.circle(image, foot_px, 15, (0, 255, 255), -1)
                        # Piano strum: Channel 2 (mido channel 1)
                        note_info = trigger_movement_note(
                            velocity,
                            vitruvian_info["vitruvian_size"],
                            channel=1,
                            octave_shift=0,
                        )
                        last_piano_strum_note = note_info[0]
                        cv2.putText(
                            image,
                            f"Note: {note_info[0]} V:{note_info[1]} (P.Strum)",
                            (foot_px[0] + 20, foot_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            get_color_for_note(note_info[0]),
                            2,
                        )

                # -------------------------
                # Process Wrist Movements (Synth on Ch1)
                # -------------------------
                left_wrist = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.LEFT_WRIST
                ]
                right_wrist = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.RIGHT_WRIST
                ]
                left_wrist_tracker.update(left_wrist)
                right_wrist_tracker.update(right_wrist)
                for tracker in [left_wrist_tracker, right_wrist_tracker]:
                    if tracker.should_trigger():
                        velocity = tracker.get_velocity()
                        wrist_px = (
                            int(tracker.positions[-1].x * image.shape[1]),
                            int(tracker.positions[-1].y * image.shape[0]),
                        )
                        cv2.circle(image, wrist_px, 10, (255, 0, 255), -1)
                        # Trigger a synth note on Channel 1 (mido channel 0)
                        note_info = trigger_movement_note(
                            velocity,
                            vitruvian_info["vitruvian_size"],
                            channel=0,
                            octave_shift=0,
                        )
                        cv2.putText(
                            image,
                            f"Note: {note_info[0]} V:{note_info[1]} (Synth)",
                            (wrist_px[0] + 20, wrist_px[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            get_color_for_note(note_info[0]),
                            2,
                        )

            # -------------------------
            # Process Right-Hand for Index Finger Gestures
            # -------------------------
            if (
                results.pose_landmarks
                and results.right_hand_landmarks
                and vitruvian_info
            ):
                # Get shoulder and hip for crossing detection
                right_shoulder = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.RIGHT_SHOULDER
                ]
                right_hip = results.pose_landmarks.landmark[
                    mp_holistic.PoseLandmark.RIGHT_HIP
                ]
                index_finger = results.right_hand_landmarks.landmark[
                    mp_holistic.HandLandmark.INDEX_FINGER_TIP
                ]
                current_index_side = point_line_side(
                    right_shoulder, right_hip, index_finger
                )
                finger_px = (
                    int(index_finger.x * image.shape[1]),
                    int(index_finger.y * image.shape[0]),
                )

                # Compute distance from the Vitruvian center:
                center = vitruvian_info["center"]
                dx = finger_px[0] - center[0]
                dy = finger_px[1] - center[1]
                dist = math.hypot(dx, dy)
                # Define two zone radii relative to the Vitruvian square size:
                inner_radius = vitruvian_info["square_size"] * 0.4
                outer_radius = vitruvian_info["square_size"] * 0.8

                # Draw the inner (green) and outer (red) circles for zone visualization:
                cv2.circle(image, center, int(inner_radius), (0, 255, 0), 2)
                cv2.circle(image, center, int(outer_radius), (0, 0, 255), 2)

                # ---------- Continuous Synth Control (Channel 1) ----------
                # If the finger is far (outside the outer circle) we want a continuous synth note.
                if dist > outer_radius:
                    if not continuous_synth_active:
                        # Choose a note from the current scale and trigger it continuously on Ch1.
                        note = random.choice(SCALES[current_chord])
                        continuous_synth_note = note
                        send_midi_note_on(note, velocity=100, channel=0)
                        continuous_synth_active = True
                    cv2.putText(
                        image,
                        f"Cont. Synth: {continuous_synth_note}",
                        (finger_px[0] + 20, finger_px[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        get_color_for_note(continuous_synth_note),
                        2,
                    )
                else:
                    # If the finger has returned inside the outer circle, stop the continuous synth note.
                    if continuous_synth_active:
                        send_midi_note_off(continuous_synth_note, channel=0)
                        continuous_synth_active = False
                        continuous_synth_note = None
                    # ---------- Discrete Trigger on Crossing ----------
                    # Only trigger a discrete note when the finger crosses the body line.
                    if prev_index_side is not None:
                        if current_index_side * prev_index_side < 0:
                            # A crossing occurred; decide between Piano Strum (Ch2) or Arpeggio (Ch3)
                            # based on distance from center.
                            # (Also use the finger’s movement to compute velocity.)
                            if prev_index_point is not None:
                                dx_f = index_finger.x - prev_index_point.x
                                dy_f = index_finger.y - prev_index_point.y
                                distance_moved = math.hypot(dx_f, dy_f)
                                velocity = int(min(max(distance_moved * 2000, 40), 127))
                            else:
                                velocity = 80
                            if dist <= inner_radius:
                                # Inside inner circle: Piano Strum on Channel 2 (mido channel 1)
                                note_info = trigger_movement_note(
                                    velocity,
                                    vitruvian_info["vitruvian_size"],
                                    channel=1,
                                    octave_shift=0,
                                )
                                last_piano_strum_note = note_info[0]
                                cv2.putText(
                                    image,
                                    f"Note: {note_info[0]} V:{note_info[1]} (P.Strum)",
                                    (finger_px[0] + 20, finger_px[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    get_color_for_note(note_info[0]),
                                    2,
                                )
                            else:
                                # Between inner and outer: Piano Arpeggio on Channel 3 (mido channel 2)
                                note_info = play_arpeggio(
                                    current_chord,
                                    velocity,
                                    channel=2,
                                    vitruvian_size=vitruvian_info["vitruvian_size"],
                                )
                                last_arpeggio_notes = note_info[0]
                                cv2.putText(
                                    image,
                                    f"Arpeggio: {note_info[0]}",
                                    (finger_px[0] + 20, finger_px[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    get_color_for_note(note_info[0][0]),
                                    2,
                                )
                    prev_index_side = current_index_side
                    prev_index_point = index_finger

                # Draw a small circle on the index finger:
                cv2.circle(image, finger_px, 8, (0, 0, 255), -1)

            # -------------------------
            # Overlay Performance Instructions & Color Panel
            # -------------------------
            cv2.putText(
                image,
                "Press 'q' to quit | '1'-'5' to change chord",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (240, 240, 240),
                2,
            )
            cv2.putText(
                image,
                f"Current Chord: {current_chord}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                image,
                "Synth (Ch1): Arms & Index (far) | Piano (Ch2): Feet & Index (close)",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
            )
            cv2.putText(
                image,
                "Piano Arpeggio (Ch3): Index (mid-range)",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (240, 240, 240),
                2,
            )
            if results.pose_landmarks:
                l_vel = left_foot_tracker.get_velocity()
                r_vel = right_foot_tracker.get_velocity()
                cv2.putText(
                    image,
                    f"Feet Velocities: L {l_vel} | R {r_vel}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (180, 240, 180),
                    2,
                )

            # Draw the color panel for note feedback
            draw_color_panel(image)

            cv2.imshow("Vitruvian Performer", image)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("q"):
                camera_active = False
            elif key == ord("1"):
                current_chord = "Am"
            elif key == ord("2"):
                current_chord = "C"
            elif key == ord("3"):
                current_chord = "G"
            elif key == ord("4"):
                current_chord = "F"
            elif key == ord("5"):
                current_chord = "Em"

    cap.release()
    cv2.destroyAllWindows()
    stop_all_midi_notes()


# -------------------------
# Run the Performer!
# -------------------------
if __name__ == "__main__":
    process_camera()
    midi_out.close()
