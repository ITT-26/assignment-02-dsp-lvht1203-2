from __future__ import annotations

import argparse
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mido
import numpy as np
import pyglet
from pyglet import shapes
import sounddevice as sd

A4_MIDI = 69
A4_FREQ = 440.0

# MIDI note to frequency conversion
def midi_note_to_freq(note: int) -> float:
    return A4_FREQ * (2 ** ((note - A4_MIDI) / 12))

# Normalize frequency to be within 1.5x of target for visual comparison. 
# This avoids yellow bars appearing one octave away.
# Octave normalization added after debugging octave mismatches during testing.
def normalize_freq_to_target(freq: float, target_freq: float) -> float:
    if freq <= 0 or target_freq <= 0:
        return freq
    corrected = freq
    while corrected > target_freq * 1.5:
        corrected /= 2.0
    while corrected < target_freq / 1.5:
        corrected *= 2.0
    return corrected

# Cents error between detected frequency and target frequency. 100 cents = 1 semitone. 
# Returns a large number if either frequency is non-positive.
def cents_error(freq: float, target_freq: float) -> float:
    if freq <= 0 or target_freq <= 0:
        return 9999.0
    corrected = normalize_freq_to_target(freq, target_freq)
    return 1200.0 * math.log2(corrected / target_freq)

# Note name for MIDI note number, e.g. 60 -> C4, 61 -> C#4, etc.
def note_name(note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[note % 12]}{note // 12 - 1}"

# Represents a MIDI note event with start time, end time, MIDI note number, and frequency.
@dataclass
class NoteEvent:
    start: float
    end: float
    note: int
    freq: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

# Load MIDI file and extract note events. If a note is held, extend its end time to the next note or by a default length.
def load_midi_notes(filename: str, default_note_length: float = 0.35) -> list[NoteEvent]:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {filename}")

    midi = mido.MidiFile(str(path))
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000  # default tempo (microseconds per beat)
    current_seconds = 0.0  
    active: dict[int, list[float]] = {}
    notes: list[NoteEvent] = []

    # Process MIDI messages in order, keeping track of active notes and their start times. 
    for msg in mido.merge_tracks(midi.tracks):
        current_seconds += mido.tick2second(msg.time, ticks_per_beat, tempo)
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, []).append(current_seconds)
        elif msg.type in ("note_off", "note_on"):
            starts = active.get(msg.note)
            if starts:
                start_time = starts.pop(0)
                if current_seconds > start_time:
                    notes.append(NoteEvent(start_time, current_seconds, msg.note, midi_note_to_freq(msg.note)))

    last_time = max([n.end for n in notes], default=0.0)
    for note, starts in active.items():
        for start_time in starts:
            notes.append(NoteEvent(start_time, max(start_time + default_note_length, last_time), note, midi_note_to_freq(note)))

    notes = sorted([n for n in notes if n.duration >= 0.03], key=lambda x: x.start)
    if not notes:
        raise ValueError("No MIDI notes found in this file.")

    # Hold notes until the next note
    extended: list[NoteEvent] = []
    for i, n in enumerate(notes):
        if i < len(notes) - 1:
            new_end = max(n.end, notes[i + 1].start)
        else:
            new_end = max(n.end, n.start + default_note_length)
        extended.append(NoteEvent(n.start, new_end, n.note, n.freq))

    return extended

# The main game window. Displays target notes and microphone pitch history, and calculates score based on how well the user matches the target.
@dataclass
class PitchState:
    freq: Optional[float] = None
    rms: float = 0.0
    confidence: float = 0.0

# Pitch detection implementation was developed with help from AI-assisted
# explanations and DSP references, then tuned experimentally during testing.
class PitchDetector:
    def __init__(
        self,
        samplerate: int = 44100,
        blocksize: int = 2048,
        min_freq: float = 100.0,
        max_freq: float = 900.0,
        rms_threshold: float = 0.002,
        device: Optional[int] = None,
    ) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.rms_threshold = rms_threshold
        self.device = device
        self._latest = PitchState()
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None
        self._smooth_freq: Optional[float] = None

    def start(self) -> None:
        self._stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            latency="low",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def latest(self) -> PitchState:
        with self._lock:
            return PitchState(self._latest.freq, self._latest.rms, self._latest.confidence)

    def _callback(self, indata, frames, callback_time, status) -> None:
        data = indata[:, 0].astype(np.float64)
        freq, rms, conf = self._estimate_pitch(data)

        if freq is not None:
            if self._smooth_freq is None:
                self._smooth_freq = freq
            else:
                self._smooth_freq = 0.70 * self._smooth_freq + 0.30 * freq # Simple smoothing filter added experimentally to reduce pitch jitter.
            freq = self._smooth_freq
        else:
            self._smooth_freq = None

        with self._lock:
            self._latest = PitchState(freq=freq, rms=rms, confidence=conf)

    # Autocorrelation-based pitch estimation adapted from DSP references
    # and refined experimentally for microphone robustness.
    def _estimate_pitch(self, data: np.ndarray) -> tuple[Optional[float], float, float]:
        data = data - np.mean(data)
        rms = float(np.sqrt(np.mean(data * data)))
        if rms < self.rms_threshold:
            return None, rms, 0.0

        data = data * np.hanning(len(data))
        corr = np.correlate(data, data, mode="full")
        corr = corr[len(corr) // 2:]

        min_lag = int(self.samplerate / self.max_freq)
        max_lag = int(self.samplerate / self.min_freq)
        max_lag = min(max_lag, len(corr) - 1)
        if min_lag >= max_lag:
            return None, rms, 0.0

        search = corr[min_lag:max_lag]
        lag = int(np.argmax(search)) + min_lag
        if lag <= 0 or corr[0] <= 1e-12:
            return None, rms, 0.0

        confidence = float(corr[lag] / (corr[0] + 1e-9))
        if confidence < 0.22: # Threshold tuned experimentally to reject unstable detections and background noise.
            return None, rms, confidence

        freq = self.samplerate / lag
        if not (self.min_freq <= freq <= self.max_freq):
            return None, rms, confidence
        return float(freq), rms, confidence

# The main game window. Displays target notes and microphone pitch history, and calculates score based on how well the user matches the target.
class KaraokeGame(pyglet.window.Window):
    def __init__(self, midi_file: str, detector: PitchDetector) -> None:
        super().__init__(920, 620, caption="ITT Karaoke Game", resizable=False)
        pyglet.gl.glClearColor(0.08, 0.09, 0.12, 1.0)
        self.notes = load_midi_notes(midi_file)
        self.detector = detector
        self.midi_file = midi_file
        # Game state
        self.started = False
        self.finished = False
        self.start_wall_time = 0.0
        self.pause_offset = 0.0
        self.score = 0.0
        self.combo = 0
        # Pre-calculate total note time for final score calculation. This is the total time the player could have been singing correct notes.
        self.total_song_note_time = sum(n.duration for n in self.notes)
        self.pitch_history: deque[tuple[float, float]] = deque(maxlen=160)
        self._last_history_add = 0.0
        # Determine the MIDI note range for visual display, with some padding. This is used to map MIDI notes to vertical positions in the lanes.
        all_midi_notes = [n.note for n in self.notes]
        self.low_note = min(all_midi_notes) - 4
        self.high_note = max(all_midi_notes) + 4
        # Set up labels for title, target note, score, and instructions. Use a monospace font for better alignment of score and combo text.
        self.font = "Arial"
        self.title_label = pyglet.text.Label("", font_name=self.font, font_size=22, x=20, y=self.height - 34, color=(255,255,255,255))
        self.target_label = pyglet.text.Label("", font_name=self.font, font_size=13, x=20, y=self.height - 62, color=(230,230,230,255))
        self.score_label = pyglet.text.Label("", font_name=self.font, font_size=15, x=self.width - 280, y=self.height - 38, color=(255,255,255,255))
        self.info_label = pyglet.text.Label("", font_name=self.font, font_size=12, x=20, y=16, color=(210,210,210,255))
        self.target_lane_label = pyglet.text.Label("TARGET", font_name=self.font, font_size=12, x=18, y=360, color=(120,180,255,255))
        self.mic_lane_label = pyglet.text.Label("YOUR VOICE", font_name=self.font, font_size=12, x=18, y=150, color=(255,210,80,255))

        pyglet.clock.schedule_interval(self.update, 1 / 30)
    # Calculate the current song time, accounting for pauses. This is used to determine which target note is active and to map microphone pitch history to the correct position in the lanes.
    def song_time(self) -> float:
        if not self.started:
            return self.pause_offset
        return time.time() - self.start_wall_time + self.pause_offset
    # Find the current target note based on the song time. This is used to determine which note the player should be singing at any given moment, and to calculate score based on how well the player's pitch matches this target.
    def current_note(self, t: float) -> Optional[NoteEvent]:
        for n in self.notes:
            if n.start <= t <= n.end:
                return n
        return None
    # Update game state based on the latest detected pitch and the current target note. This is where scoring happens: if the player's pitch is close enough to the target, they earn points and build combo; if they are off, they lose combo. Also updates the pitch history for visual display in the microphone lane.
    def update(self, dt: float) -> None:
        if not self.started or self.finished:
            return

        t = self.song_time()
        pitch = self.detector.latest()
        target = self.current_note(t)

        # Store pitch for the lower lane. When a target exists, fold octave to target
        # for fair visual comparison. This avoids yellow bars appearing one octave away.
        if pitch.freq is not None and (t - self._last_history_add) >= 0.08:
            display_freq = pitch.freq
            if target is not None:
                display_freq = normalize_freq_to_target(pitch.freq, target.freq)
            midi_float = A4_MIDI + 12 * math.log2(display_freq / A4_FREQ)
            while midi_float > self.high_note:
                midi_float -= 12
            while midi_float < self.low_note:
                midi_float += 12
            self.pitch_history.append((t, midi_float))
            self._last_history_add = t
        # Scoring: If the player is singing the correct note (within 150 cents), they earn score and build combo. If they are close (within 300 cents), they earn some score but lose combo. If they are far off, they lose all combo. This encourages players to try to match the target pitch as closely as possible.
        if target is not None and pitch.freq is not None:
            err = abs(cents_error(pitch.freq, target.freq))
            # Scoring thresholds were adjusted experimentally during gameplay testing.
            if err < 150: 
                self.score += dt
                self.combo += 1
            elif err < 300:
                self.score += 0.5 * dt
                self.combo = max(0, self.combo - 1)
            else:
                self.combo = 0
        elif target is not None:
            self.combo = 0
        # Check if we've passed the end of the last note. If so, mark the game as finished. 
        # This allows the player to see their final score and gives them a chance to restart or quit.
        if t > self.notes[-1].end + 2.0:
            self.finished = True
            self.started = False
            self.pause_offset = self.notes[-1].end

    # Handle key presses for starting/pausing the game, restarting, and quitting. SPACE toggles between start and pause, R restarts the game, and ESC quits. 
    # This allows the player to control the flow of the game and gives them options to retry or exit.    
    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            if self.finished:
                self.restart()
            elif self.started:
                self.pause_offset = self.song_time()
                self.started = False
            else:
                self.start_wall_time = time.time()
                self.started = True
        elif symbol == pyglet.window.key.R:
            self.restart()
        elif symbol == pyglet.window.key.ESCAPE:
            self.close()

    # Reset game state to start a new game. This is called when the player presses R or SPACE after finishing. 
    # It clears the score, combo, pitch history, and resets timing variables so that the player can start fresh.
    def restart(self) -> None:
        self.started = False
        self.finished = False
        self.start_wall_time = 0.0
        self.pause_offset = 0.0
        self.score = 0.0
        self.combo = 0
        self.pitch_history.clear()
        self._last_history_add = 0.0

    # Convert MIDI note number to a vertical position in the lane. 
    # This maps the range of MIDI notes in the song to the vertical space of the lane, 
    # allowing us to visually represent both target notes and microphone pitch history 
    # in a way that shows how close the player's pitch is to the target. 
    def note_to_y_in_lane(self, midi_note: float, lane_bottom: float, lane_top: float) -> float:
        span = max(1, self.high_note - self.low_note)
        return lane_bottom + (midi_note - self.low_note) / span * (lane_top - lane_bottom)

    # Convert a note time to an x position on the screen, based on the current song time.
    # This allows us to scroll the target notes and microphone history from right to left as time
    # progresses, creating a "lane" effect where notes approach a hit line and the player tries 
    # to match them at the right moment.
    def time_to_x(self, note_time: float, current_time: float) -> float:
        hit_x = 260
        pixels_per_second = 150
        return hit_x + (note_time - current_time) * pixels_per_second

    # Draw the background of a lane, including horizontal lines for each MIDI note. 
    # This is called for both the target lane and the microphone lane, with different vertical positions. 
    # The horizontal lines help the player visually gauge how close their pitch is to the target notes.
    def draw_lane_background(self, bottom: int, top: int) -> None:
        lane_left = 80
        lane_right = self.width - 30
        bg = shapes.Rectangle(lane_left, bottom, lane_right - lane_left, top - bottom, color=(20, 22, 30))
        bg.opacity = 190
        bg.draw()
        for note in range(self.low_note, self.high_note + 1, 2):
            y = self.note_to_y_in_lane(note, bottom + 10, top - 10)
            color = (52, 54, 66) if note % 12 not in (0, 5) else (75, 78, 95)
            shapes.Line(lane_left, y, lane_right, y, width=1, color=color).draw()

    # Draw the target notes as blue rectangles in the upper lane. 
    # The width of the rectangle represents the duration of the note, and the vertical position 
    # represents the MIDI note number. If the current time is within the note's duration, 
    # it is drawn in a brighter color to indicate that it is the active target.
    def draw_target_notes(self, t: float) -> None:
        bottom, top = 330, 540
        lane_left = 80
        lane_right = self.width - 30
        for n in self.notes:
            if n.end < t - 1.0 or n.start > t + 4.2:
                continue
            x1 = self.time_to_x(n.start, t)
            x2 = self.time_to_x(n.end, t)
            if x2 < lane_left or x1 > lane_right:
                continue
            x1 = max(x1, lane_left)
            x2 = min(x2, lane_right)
            y = self.note_to_y_in_lane(n.note, bottom + 10, top - 10)
            width = max(7, x2 - x1)
            color = (70, 145, 255) if not (n.start <= t <= n.end) else (110, 190, 255)
            rect = shapes.Rectangle(x1, y - 7, width, 14, color=color)
            rect.opacity = 220
            rect.draw()

    # Yellow microphone lane was added iteratively during debugging
    # to visually compare detected pitch against MIDI targets.
    def draw_mic_history(self, t: float) -> None:
        bottom, top = 100, 300
        lane_left = 80
        lane_right = self.width - 30
        recent = [(pt_t, midi) for pt_t, midi in self.pitch_history if t - 4.2 <= pt_t <= t + 0.2]
        if not recent:
            return

        bar_height = 14
        min_bar_width = 8
        max_gap_seconds = 0.16

        for i, (pt_t, midi_float) in enumerate(recent):
            next_t = pt_t + 0.08
            if i + 1 < len(recent):
                candidate_next_t = recent[i + 1][0]
                if 0.0 < candidate_next_t - pt_t <= max_gap_seconds:
                    next_t = candidate_next_t

            x1 = self.time_to_x(pt_t, t)
            x2 = self.time_to_x(next_t, t)
            if x2 < lane_left or x1 > lane_right:
                continue
            x1 = max(x1, lane_left)
            x2 = min(x2, lane_right)

            y = self.note_to_y_in_lane(midi_float, bottom + 10, top - 10)
            width = max(min_bar_width, x2 - x1)
            rect = shapes.Rectangle(x1, y - bar_height / 2, width, bar_height, color=(255, 210, 65))
            rect.opacity = 215
            rect.draw()

    # Draw the hit line where the player is supposed to match the target notes. 
    # This is a vertical line that serves as a visual reference for timing, 
    # showing players where they should aim to have their pitch when singing the target notes.
    def draw_hit_line(self) -> None:
        hit_x = 260
        shapes.Line(hit_x, 92, hit_x, 548, width=2, color=(255, 255, 255)).draw()

    # Draw the entire game screen, including lane backgrounds, target notes, microphone history, hit line, and labels. 
    # This is called every frame to update the visuals based on the current game state and player input. 
    # It also updates the text of the labels to show the current target note, score, combo
    def on_draw(self):
        self.clear()
        t = self.song_time()
        target = self.current_note(t)

        self.draw_lane_background(330, 540)
        self.draw_lane_background(100, 300)
        self.draw_target_notes(t)
        self.draw_mic_history(t)
        self.draw_hit_line()

        self.title_label.text = f"Karaoke Game - {Path(self.midi_file).name}"
        self.title_label.draw()
        self.target_lane_label.draw()
        self.mic_lane_label.draw()

        if target is None:
            self.target_label.text = "Target: --"
        else:
            self.target_label.text = f"Target: {note_name(target.note)}"
        self.target_label.draw()

        accuracy = 0.0 if self.total_song_note_time <= 0 else 100.0 * self.score / self.total_song_note_time
        accuracy = max(0.0, min(100.0, accuracy))
        self.score_label.text = f"Score: {accuracy:5.1f}%   Combo: {self.combo}"
        self.score_label.draw()

        if self.finished:
            self.info_label.text = "Finished. Press R or SPACE to restart. ESC quits."
        elif self.started:
            self.info_label.text = "Sing the blue target. Yellow shows your voice. SPACE pauses. R restarts. ESC quits."
        else:
            self.info_label.text = "Press SPACE to start. Top blue = target; bottom yellow = mic bars. R restarts. ESC quits."
        self.info_label.draw()

# Handle key presses for starting/pausing the game, restarting, and quitting. SPACE toggles between start and pause, R restarts the game, and ESC quits.
def list_input_devices() -> None:
    print("Available input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"{i}: {dev['name']}")

# The main entry point of the program. Parses command line arguments for MIDI file, audio device, and other settings. 
# Initializes the pitch detector and game window, and starts the Pyglet application loop. 
# Also handles listing audio devices if requested.
def main() -> None:
    parser = argparse.ArgumentParser(description="Karaoke game with target and microphone lanes.")
    parser.add_argument("midi", help="Path to MIDI file")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--samplerate", type=int, default=44100)
    parser.add_argument("--blocksize", type=int, default=1024)
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    detector = PitchDetector(samplerate=args.samplerate, blocksize=args.blocksize, device=args.device)
    detector.start()
    try:
        KaraokeGame(args.midi, detector)
        pyglet.app.run()
    finally:
        detector.stop()

# AI tools were used during development for debugging support,
# structure suggestions, and understanding DSP-related concepts.
# The implementation and parameter tuning were iteratively modified during testing.
if __name__ == "__main__":
    main()
