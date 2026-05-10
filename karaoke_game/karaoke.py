"I sincerely apologize for not providing sufficiently detailed comments in my initial submission, "
"and I truly appreciate the opportunity to revise my work and for your time spent reviewing it again. "
"I will be more careful in the future regarding code documentation, proper attribution of sources, "
"and the responsible use of AI assistance."

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

# Standard MIDI to frequency conversion.
# I referred to the well-known formula (A4 = MIDI 69 = 440 Hz) from online sources:
# https://en.wikipedia.org/wiki/MIDI_tuning_standard
def midi_note_to_freq(note: int) -> float:
    return A4_FREQ * (2 ** ((note - A4_MIDI) / 12))

# Normalize detected microphone frequency to match the octave of the target note.
# The idea of octave normalization was suggested during development (AI-assisted discussion),
# but I implemented and adjusted the logic myself.
#
# Reason:
# During testing, the autocorrelation pitch detector sometimes returned frequencies
# that were correct musically but shifted by one octave (×2 or ÷2), which caused
# incorrect visual comparison with MIDI notes.
#
# Implementation choice:
# I used a simple iterative scaling (divide/multiply by 2) instead of log-based mapping,
# because it is easier to control and sufficient for realtime use.
#
# The threshold (1.5× range) was chosen experimentally to decide when a value is
# considered "too far" from the target octave.
def normalize_freq_to_target(freq: float, target_freq: float) -> float:
    if freq <= 0 or target_freq <= 0:
        return freq
    corrected = freq
    while corrected > target_freq * 1.5:
        corrected /= 2.0
    while corrected < target_freq / 1.5:
        corrected *= 2.0
    return corrected

# Pitch error in cents (100 cents = 1 semitone).
# I referred to the standard definition of cents from music theory:
# https://en.wikipedia.org/wiki/Cent_(music)
#
# The formula uses log2 to measure pitch difference between two frequencies.
# I implemented this directly and combined it with my octave normalization step
# to avoid large errors caused by octave mismatches during detection.
#
# Returning a large fallback value (9999.0) is my choice to indicate invalid input
# and simplify later scoring logic (treated as completely off-pitch).
def cents_error(freq: float, target_freq: float) -> float:
    if freq <= 0 or target_freq <= 0:
        return 9999.0
    corrected = normalize_freq_to_target(freq, target_freq)
    return 1200.0 * math.log2(corrected / target_freq)

# Convert MIDI note number to readable note name (e.g. 60 -> C4).
# The mapping structure (12-note chromatic scale) follows standard MIDI convention:
# https://en.wikipedia.org/wiki/MIDI
# I implemented this using modulo (for pitch class) and integer division (for octave),
# which is a common approach and sufficient for visualization purposes.
def note_name(note: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[note % 12]}{note // 12 - 1}"

# Represents a MIDI note event with timing and pitch information.
# This is my own data structure to simplify handling note events after parsing MIDI.
# I store both MIDI note number and precomputed frequency to avoid repeated conversion later.
@dataclass
class NoteEvent:
    start: float
    end: float
    note: int
    freq: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start) # Ensure duration is non-negative (can happen due to edge cases in MIDI parsing)

# Load MIDI file and extract note events.
# I used the mido library for MIDI parsing:
# https://mido.readthedocs.io/
# The overall structure (iterating through merged tracks and converting ticks to seconds)
# follows typical usage from documentation, but the note handling logic (tracking active notes,
# pairing note_on/note_off, filtering, and extending notes) was implemented and adjusted by me.
def load_midi_notes(filename: str, default_note_length: float = 0.35) -> list[NoteEvent]:
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {filename}")

    midi = mido.MidiFile(str(path))
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000  # Default tempo (used until a set_tempo message appears)
    current_seconds = 0.0  
    # Track currently active notes (note -> list of start times)
    # I used a list because the same pitch can overlap in MIDI.
    active: dict[int, list[float]] = {}
    notes: list[NoteEvent] = []

     # Convert MIDI messages into note events
    for msg in mido.merge_tracks(midi.tracks):
        
        # Convert delta ticks to seconds
        current_seconds += mido.tick2second(msg.time, ticks_per_beat, tempo)
        
        # Update tempo dynamically if present in the MIDI
        if msg.type == "set_tempo":
            tempo = msg.tempo
            continue
        
        # Note start
        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, []).append(current_seconds)
        
        # Note end (note_off OR note_on with velocity 0)
        elif msg.type in ("note_off", "note_on"):
            starts = active.get(msg.note)
            if starts:
                start_time = starts.pop(0)

                # Only keep valid notes (avoid zero-length artifacts)
                if current_seconds > start_time:
                    notes.append(NoteEvent(start_time, current_seconds, msg.note, midi_note_to_freq(msg.note)))

    # Handle notes that were never closed (edge case in some MIDI files)
    last_time = max([n.end for n in notes], default=0.0)
    for note, starts in active.items():
        for start_time in starts:
            # I assign a fallback duration using either default length or last known time
            notes.append(NoteEvent(start_time, max(start_time + default_note_length, last_time), note, midi_note_to_freq(note)))

    # Filter out extremely short notes (noise / artifacts from MIDI)
    # Threshold (0.03s) chosen experimentally to remove very short spikes
    notes = sorted([n for n in notes if n.duration >= 0.03], key=lambda x: x.start)
    if not notes:
        raise ValueError("No MIDI notes found in this file.")

    # Extend notes to fill gaps until the next note
    # This was added after testing because very short gaps made the visualization hard to follow.
    extended: list[NoteEvent] = []
    for i, n in enumerate(notes):
        if i < len(notes) - 1:
            new_end = max(n.end, notes[i + 1].start)
        else:
            new_end = max(n.end, n.start + default_note_length)
        extended.append(NoteEvent(n.start, new_end, n.note, n.freq))

    return extended

# Stores the latest detected pitch state from the microphone.
# I created this structure to group frequency, RMS (signal energy),
# and confidence for easier handling across the program.
@dataclass
class PitchState:
    freq: Optional[float] = None
    rms: float = 0.0
    confidence: float = 0.0

# Realtime microphone pitch detection setup.
# I referred to general DSP concepts (autocorrelation) and used AI-assisted explanations
# to understand the approach, but the parameter choices and integration into the realtime
# audio pipeline were implemented and tuned by me.
    def __init__(
        self,
        samplerate: int = 44100,
        blocksize: int = 2048,
        min_freq: float = 100.0,
        max_freq: float = 900.0,
        rms_threshold: float = 0.002,
        device: Optional[int] = None,
    ) -> None:
        self.samplerate = samplerate # Sampling rate for audio input
        
        # Block size affects latency vs stability
        # Chosen experimentally as a balance for realtime singing input
        self.blocksize = blocksize

        # Frequency range limited to typical human voice range
        # Adjusted during testing to avoid invalid detections
        self.min_freq = min_freq
        self.max_freq = max_freq

        # Minimum RMS threshold to ignore silence/background noise
        # Tuned experimentally
        self.rms_threshold = rms_threshold
        
        self.device = device
        
        self._latest = PitchState()
        
        # Lock for thread-safe communication between audio callback and main thread
        self._lock = threading.Lock()

        self._stream: Optional[sd.InputStream] = None
        
        # Used for smoothing pitch values over time
        self._smooth_freq: Optional[float] = None

    # Start microphone stream using sounddevice library:
    # https://python-sounddevice.readthedocs.io/
    # The callback structure follows the library API, but the pitch processing
    # logic inside the callback is implemented by me.
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

    # Return latest detected pitch (thread-safe)
    def latest(self) -> PitchState:
        with self._lock:
            return PitchState(self._latest.freq, self._latest.rms, self._latest.confidence)

    # Audio callback (called continuously)
    # I process each audio block here and estimate pitch in realtime.
    def _callback(self, indata, frames, callback_time, status) -> None:
        
        # Convert audio input to 1D array
        data = indata[:, 0].astype(np.float64)
        
        # Core pitch estimation
        freq, rms, conf = self._estimate_pitch(data)

        # Apply smoothing to reduce pitch jitter
        # Added after testing because raw output was unstable
        if freq is not None:
            if self._smooth_freq is None:
                self._smooth_freq = freq
            else:
                # Weighted smoothing (previous + current)
                # Coefficients tuned experimentally
                self._smooth_freq = 0.70 * self._smooth_freq + 0.30 * freq # Simple smoothing filter added experimentally to reduce pitch jitter.
            freq = self._smooth_freq
        else:
            # Reset when no valid pitch
            self._smooth_freq = None

        with self._lock:
            self._latest = PitchState(freq=freq, rms=rms, confidence=conf)

    # Autocorrelation-based pitch estimation.
    # I referred to DSP concepts and online explanations of autocorrelation/YIN:
    # https://en.wikipedia.org/wiki/Autocorrelation
    # https://hyuncat.github.io/blog/yin
    # I also used AI-assisted explanations to better understand how autocorrelation
    # can be applied to pitch detection, but the implementation here was written
    # and simplified by me for realtime microphone input.
    # In particular, I removed more complex steps from YIN and kept a simpler version
    # that is fast enough for realtime use, then tuned parameters based on testing.
    def _estimate_pitch(self, data: np.ndarray) -> tuple[Optional[float], float, float]:

        # Remove DC offset (center signal around 0)
        # This improves stability of autocorrelation
        data = data - np.mean(data)
        rms = float(np.sqrt(np.mean(data * data)))
        
        # Compute RMS (signal energy)
        # I use this to filter out silence / background noise
        if rms < self.rms_threshold:
            return None, rms, 0.0

        # Ignore very low-energy frames (threshold tuned experimentally)
        data = data * np.hanning(len(data))

        # Compute autocorrelation
        corr = np.correlate(data, data, mode="full")
        corr = corr[len(corr) // 2:]

        # Convert frequency range into lag range
        # (lag = samplerate / frequency)
        min_lag = int(self.samplerate / self.max_freq)
        max_lag = int(self.samplerate / self.min_freq)

        # Ensure valid search range
        max_lag = min(max_lag, len(corr) - 1)
        if min_lag >= max_lag:
            return None, rms, 0.0

        # Search for peak in autocorrelation within valid lag range
        search = corr[min_lag:max_lag]
        lag = int(np.argmax(search)) + min_lag

        # Reject invalid peaks
        if lag <= 0 or corr[0] <= 1e-12:
            return None, rms, 0.0

        # Compute confidence as normalized autocorrelation peak
        # This helps reject unstable detections
        confidence = float(corr[lag] / (corr[0] + 1e-9))

        # Threshold tuned experimentally:
        # lower values → more noise
        # higher values → miss valid notes
        if confidence < 0.22:
            return None, rms, confidence

        # Convert lag back to frequency
        freq = self.samplerate / lag

        # Reject values outside expected vocal range
        # Range chosen and adjusted during testing
        if not (self.min_freq <= freq <= self.max_freq):
            return None, rms, confidence
        return float(freq), rms, confidence

# Main game window: displays target notes, microphone pitch history, and computes score in real time.
# The overall architecture of this class (game loop structure, lane-based visualization, update/draw separation)
# is based on an AI-suggested design pattern for realtime audio visualization.
# I then adapted and extended it to fit karaoke-specific pitch matching and scoring requirements.
class KaraokeGame(pyglet.window.Window):
    def __init__(self, midi_file: str, detector: PitchDetector) -> None:
        super().__init__(920, 620, caption="ITT Karaoke Game", resizable=False)
        pyglet.gl.glClearColor(0.08, 0.09, 0.12, 1.0)
        
        # MIDI parsing logic: I followed an AI-proposed pipeline structure for loading note events,
        # but implemented and adjusted the actual parsing function (load_midi_notes) myself.
        self.notes = load_midi_notes(midi_file)
        self.detector = detector
        self.midi_file = midi_file
        # Game state
        self.started = False
        self.finished = False
        self.start_wall_time = 0.0
        self.pause_offset = 0.0

        # Scoring system:
        # The idea of combining score + combo was suggested by AI design.
        # I simplified the scoring model to use time-weighted accumulation (dt-based),
        # and manually tuned it during testing for more stable gameplay behavior.
        self.score = 0.0
        self.combo = 0

        # Total playable note duration used for final normalization of score.
        self.total_song_note_time = sum(n.duration for n in self.notes)

        # Pitch history buffer:
        # AI suggested using a deque for real-time streaming visualization.
        # I kept this structure because it is efficient for fixed-size rolling history.
        self.pitch_history: deque[tuple[float, float]] = deque(maxlen=160)
        self._last_history_add = 0.0
        
        # MIDI range mapping:
        # I compute note range dynamically from the song and add padding manually
        # to improve visual spacing in the UI lanes.
        all_midi_notes = [n.note for n in self.notes]
        self.low_note = min(all_midi_notes) - 4
        self.high_note = max(all_midi_notes) + 4

        # UI components generated by AI
        self.font = "Arial"
        self.title_label = pyglet.text.Label("", font_name=self.font, font_size=22, x=20, y=self.height - 34, color=(255,255,255,255))
        self.target_label = pyglet.text.Label("", font_name=self.font, font_size=13, x=20, y=self.height - 62, color=(230,230,230,255))
        self.score_label = pyglet.text.Label("", font_name=self.font, font_size=15, x=self.width - 280, y=self.height - 38, color=(255,255,255,255))
        self.info_label = pyglet.text.Label("", font_name=self.font, font_size=12, x=20, y=16, color=(210,210,210,255))
        self.target_lane_label = pyglet.text.Label("TARGET", font_name=self.font, font_size=12, x=18, y=360, color=(120,180,255,255))
        self.mic_lane_label = pyglet.text.Label("YOUR VOICE", font_name=self.font, font_size=12, x=18, y=150, color=(255,210,80,255))

        # Game loop scheduling (AI standard pattern for pyglet applications)
        pyglet.clock.schedule_interval(self.update, 1 / 30)
    
    # Core game logic
    # Compute current song time including pause handling.
    # This function is simple but critical; I implemented it myself based on standard game loop timing logic.
    def song_time(self) -> float:
        if not self.started:
            return self.pause_offset
        return time.time() - self.start_wall_time + self.pause_offset
    
    # Find the active note at time t.
    # AI suggested a linear scan approach for simplicity.
    # I kept it because dataset size (MIDI notes) is small, so O(n) is acceptable.
    def current_note(self, t: float) -> Optional[NoteEvent]:
        for n in self.notes:
            if n.start <= t <= n.end:
                return n
        return None
    
    # Main update loop (called 30 FPS).
    # This is the core logic of the application.
    # AI provided the general structure (update loop with pitch detection + scoring + visualization sync),
    # but I heavily modified the scoring rules and pitch handling behavior.
    def update(self, dt: float) -> None:
        if not self.started or self.finished:
            return

        t = self.song_time()
        pitch = self.detector.latest()
        target = self.current_note(t)

        # Pitch history visualization
        # AI suggested using a rolling buffer for real-time plotting (deque).
        # I implemented additional smoothing logic and timing control to avoid visual clutter.
        if pitch.freq is not None and (t - self._last_history_add) >= 0.08:
            display_freq = pitch.freq

            # Optional octave normalization to improve visual comparison between voice and target.
            # This part was added by me after observing mismatch issues during testing.
            if target is not None:
                display_freq = normalize_freq_to_target(pitch.freq, target.freq)
            midi_float = A4_MIDI + 12 * math.log2(display_freq / A4_FREQ)
            
            # Clamp MIDI range to prevent rendering overflow.
            # Implemented manually for UI stability.
            while midi_float > self.high_note:
                midi_float -= 12
            while midi_float < self.low_note:
                midi_float += 12
            self.pitch_history.append((t, midi_float))
            self._last_history_add = t

        # Scoring system 
        # Core idea (AI-suggested): compare detected pitch vs target pitch and assign score + combo.
        # However, I fully re-tuned thresholds and scoring weights based on experimentation.
            err = abs(cents_error(pitch.freq, target.freq))
            # Thresholds (150 / 300 cents) were manually tuned by me
            # to balance difficulty and avoid overly strict scoring.
            if err < 150: 
                self.score += dt
                self.combo += 1
            elif err < 300:
                self.score += 0.5 * dt
                self.combo = max(0, self.combo - 1)
            else:
                self.combo = 0
        elif target is not None:
            # No pitch detected while note is active → reset combo
            self.combo = 0
        
        # End condition
        # Game ends slightly after last note to ensure smooth UI transition.
        # This timing buffer (2 seconds) was added by me for better UX.
        if t > self.notes[-1].end + 2.0:
            self.finished = True
            self.started = False
            self.pause_offset = self.notes[-1].end


    # Handle keyboard input for game control.
    # AI suggested a simple key-event system (SPACE / R / ESC).
    # I implemented it and slightly extended pause logic to preserve timing consistency.    
    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.SPACE:
            if self.finished:
                self.restart()
            elif self.started:
                # Pause: store elapsed song time instead of wall clock
                self.pause_offset = self.song_time()
                self.started = False
            else:
                # Start/resume game
                self.start_wall_time = time.time()
                self.started = True
        elif symbol == pyglet.window.key.R:
            self.restart()
        elif symbol == pyglet.window.key.ESCAPE:
            self.close()

    # Reset game state.
    # Fully implemented by me, following standard reset-state pattern in game design.
    def restart(self) -> None:
        self.started = False
        self.finished = False
        self.start_wall_time = 0.0
        self.pause_offset = 0.0
        self.score = 0.0
        self.combo = 0
        self.pitch_history.clear()
        self._last_history_add = 0.0

    # Map MIDI note to vertical UI position.
    # AI suggested linear mapping; I kept it but added padding and clamping logic for better readability.
    def note_to_y_in_lane(self, midi_note: float, lane_bottom: float, lane_top: float) -> float:
        span = max(1, self.high_note - self.low_note)
        return lane_bottom + (midi_note - self.low_note) / span * (lane_top - lane_bottom)

    # Convert time → x coordinate (horizontal scrolling effect).
    # This scrolling lane concept was AI-suggested; I only tuned speed parameter experimentally.
    def time_to_x(self, note_time: float, current_time: float) -> float:
        hit_x = 260
        pixels_per_second = 150
        return hit_x + (note_time - current_time) * pixels_per_second

    # Draw lane background grid.
    # AI provided the general idea of using horizontal pitch guides.
    # I adjusted colors and spacing to improve visual clarity for pitch recognition.
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

    # Draw target MIDI notes (upper lane).
    # AI suggested rectangle-based note visualization (like piano-roll UI).
    # I added brightness switching for active note highlighting.
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

    # Draw microphone pitch history (lower lane).
    # AI suggested using streaming visualization; I refined it with smoothing + merging logic.
    def draw_mic_history(self, t: float) -> None:
        bottom, top = 100, 300
        lane_left = 80
        lane_right = self.width - 30
        recent = [(pt_t, midi) for pt_t, midi in self.pitch_history if t - 4.2 <= pt_t <= t + 0.2]
        if not recent:
            return

        bar_height = 14
        min_bar_width = 8
        max_gap_seconds = 0.16 # Merge nearby samples for smoother visualization (my improvement)

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

    # Draw vertical hit reference line.
    # Simple UI element fully implemented by me.
    def draw_hit_line(self) -> None:
        hit_x = 260
        shapes.Line(hit_x, 92, hit_x, 548, width=2, color=(255, 255, 255)).draw()

    # Main draw function.
    # AI provided typical game rendering structure (clear → draw layers → UI text),
    # but I fully organized layout and scoring display myself.
    def on_draw(self):
        self.clear()
        t = self.song_time()
        target = self.current_note(t)

        self.draw_lane_background(330, 540)
        self.draw_lane_background(100, 300)
        self.draw_target_notes(t)
        self.draw_mic_history(t)
        self.draw_hit_line()

        # UI labels (mostly AI structure, but manually positioned and styled)
        self.title_label.text = f"Karaoke Game - {Path(self.midi_file).name}"
        self.title_label.draw()
        self.target_lane_label.draw()
        self.mic_lane_label.draw()

        if target is None:
            self.target_label.text = "Target: --"
        else:
            self.target_label.text = f"Target: {note_name(target.note)}"
        self.target_label.draw()

        # Score normalization logic:
        # I designed this formula myself to map raw score into percentage for readability.
        accuracy = 0.0 if self.total_song_note_time <= 0 else 100.0 * self.score / self.total_song_note_time
        accuracy = max(0.0, min(100.0, accuracy))
        self.score_label.text = f"Score: {accuracy:5.1f}%   Combo: {self.combo}"
        self.score_label.draw()

        # Game state messages (fully written by me for UX clarity)
        if self.finished:
            self.info_label.text = "Finished. Press R or SPACE to restart. ESC quits."
        elif self.started:
            self.info_label.text = "Sing the blue target. Yellow shows your voice. SPACE pauses. R restarts. ESC quits."
        else:
            self.info_label.text = "Press SPACE to start. Top blue = target; bottom yellow = mic bars. R restarts. ESC quits."
        self.info_label.draw()

# List available microphone input devices.
# Direct use of sounddevice API; no AI logic involved here.
def list_input_devices() -> None:
    print("Available input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"{i}: {dev['name']}")

# Main function controlling CLI arguments and application startup.
# AI suggested argparse-based structure; I followed it and adapted parameters for audio configuration.
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

if __name__ == "__main__":
    main()
