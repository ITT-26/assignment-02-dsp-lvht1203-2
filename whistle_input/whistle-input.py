"I sincerely apologize for not providing sufficiently detailed comments in my initial submission, "
"and I truly appreciate the opportunity to revise my work and for your time spent reviewing it again. "
"I will be more careful in the future regarding code documentation, proper attribution of sources, "
"and the responsible use of AI assistance."

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyglet
from pyglet import shapes
import sounddevice as sd


@dataclass
class Pitch:
    freq: Optional[float] = None


class PitchDetector:
    # Pitch detection module using FFT-based signal processing.
    # The overall design (audio stream callback + real-time pitch estimation pipeline)
    # is based on an AI-suggested architecture for low-latency audio processing.
    # I then implemented and tuned filtering thresholds, smoothing, and noise rejection manually.
    def __init__(
        self,
        samplerate: int = 44100,
        blocksize: int = 1024,
        min_freq: float = 300.0,
        max_freq: float = 3000.0,
        rms_threshold: float = 0.002,
        device: Optional[int] = None,
    ) -> None:
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.rms_threshold = rms_threshold
        self.device = device

        # AI suggested thread-safe storage pattern for real-time audio pipeline.
        self._latest = Pitch()
        self._smooth_freq: Optional[float] = None
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None

    # Start audio stream (standard sounddevice callback pattern).
    # This is largely based on library documentation + AI structure suggestion.
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
        # Safe resource cleanup (standard audio streaming practice)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # Thread-safe retrieval of last detected pitch
    def latest(self) -> Pitch:
        with self._lock:
            return Pitch(self._latest.freq)

    # Audio callback processing pipeline.
    # AI suggested FFT + smoothing approach.
    # I tuned smoothing factor (0.65 / 0.35) and noise thresholds experimentally.
    def _callback(self, indata, frames, callback_time, status) -> None:
        freq = self._estimate_pitch(indata[:, 0].astype(np.float64))

        if freq is not None:
            if self._smooth_freq is None:
                self._smooth_freq = freq
            else:
                self._smooth_freq = 0.65 * self._smooth_freq + 0.35 * freq # simple low-pass filter for smoothing
            freq = self._smooth_freq
        else:
            self._smooth_freq = None

        with self._lock:
            self._latest = Pitch(freq)

    # FFT-based pitch estimation.
    # AI helped with the overall DSP pipeline idea (FFT + peak detection),
    # but all thresholds (noise gate, ratio checks, frequency bounds) were tuned by me.
    def _estimate_pitch(self, data: np.ndarray) -> Optional[float]:
        data = data - np.mean(data)
        rms = float(np.sqrt(np.mean(data * data)))

        # noise gate (manually tuned)
        if rms < self.rms_threshold:
            return None

        windowed = data * np.hanning(len(data))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(data), 1.0 / self.samplerate)

        mask = (freqs >= self.min_freq) & (freqs <= self.max_freq)
        if not np.any(mask):
            return None

        sub = spectrum[mask]
        peak_index = int(np.argmax(sub))
        peak_value = float(sub[peak_index])
        noise_floor = float(np.mean(sub) + 1e-9)

        # heuristic rejection of noisy peaks (experimentally tuned)
        if peak_value / noise_floor < 4.0:
            return None

        global_index = np.where(mask)[0][0] + peak_index
        freq = global_index * self.samplerate / len(data)

        if self.min_freq <= freq <= self.max_freq:
            return float(freq)

        return None



class ChirpDetector:
    # Pattern recognition over pitch history to detect "chirps" (up/down gestures).
    # AI suggested using slope-based regression approach.
    # I kept the idea but tuned thresholds + cooldown logic manually for stability.
    def __init__(
        self,
        window_seconds: float = 0.75, 
        min_points: int = 5, 
        min_freq_change: float = 120.0, 
        min_slope: float = 120.0, 
        cooldown_seconds: float = 0.55, 
    ) -> None:
        self.window_seconds = window_seconds
        self.min_points = min_points
        self.min_freq_change = min_freq_change
        self.min_slope = min_slope
        self.cooldown_seconds = cooldown_seconds

        self.history = deque()
        self.last_trigger = 0.0

    # Update the chirp detector with the latest detected pitch frequency. 
    # It maintains a history of recent pitch detections and checks for patterns that indicate an upward or downward chirp.
    # If a valid chirp is detected, it returns the direction ("up" or "down").
    def update(self, freq: Optional[float]) -> Optional[str]:
        now = time.time()

        if freq is not None:
            self.history.append((now, freq))

        # sliding window maintenance (standard streaming logic)
        while self.history and now - self.history[0][0] > self.window_seconds:
            self.history.popleft()

        if now - self.last_trigger < self.cooldown_seconds:
            return None

        direction = self.detect()
        if direction is not None:
            self.last_trigger = now
            self.history.clear()

        return direction

    # Regression-based slope detection.
    # AI suggested linear trend fitting; I directly implemented numpy polyfit version.
    def detect(self) -> Optional[str]:
        if len(self.history) < self.min_points:
            return None

        times = np.array([x[0] for x in self.history])
        freqs = np.array([x[1] for x in self.history])
        times = times - times[0]

        if times[-1] <= 0:
            return None

        freq_change = float(freqs[-1] - freqs[0])
        slope = float(np.polyfit(times, freqs, 1)[0])

        # thresholds tuned experimentally
        if abs(freq_change) < self.min_freq_change:
            return None
        if abs(slope) < self.min_slope:
            return None

        return "up" if slope > 0 else "down"


class KeyboardOutput:
    # Simple system automation layer using pynput.
    # Fully based on library usage (not AI-heavy logic).
    
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.keyboard = None

        if enabled:
            from pynput.keyboard import Controller
            self.keyboard = Controller()
    
    def press_arrow(self, direction: str) -> None:
        if not self.enabled or self.keyboard is None:
            return

        from pynput.keyboard import Key

        key = Key.up if direction == "up" else Key.down
        self.keyboard.press(key)
        self.keyboard.release(key)


class WhistleWindow(pyglet.window.Window):
    # UI layer for whistle-controlled interaction system.
    # AI provided general GUI structure (item list + selection model),
    # I adapted it into a gesture-controlled navigation system.

    def __init__(
        self,
        pitch_detector: PitchDetector,
        chirp_detector: ChirpDetector,
        keyboard_output: KeyboardOutput,
    ) -> None:
        super().__init__(620, 560, caption="Whistle Input", resizable=False) 
        pyglet.gl.glClearColor(0.08, 0.09, 0.12, 1.0) 

        self.pitch_detector = pitch_detector
        self.chirp_detector = chirp_detector
        self.keyboard_output = keyboard_output

        self.item_count = 8
        self.selected = 3

        # UI layout manually designed (AI-assisted structure, human-tuned layout)
        self.title = pyglet.text.Label(
            "Whistle Input",
            font_size=24,
            x=24,
            y=515,
            color=(255, 255, 255, 255),
        )
        self.help = pyglet.text.Label(
            "Upward chirp moves up. Downward chirp moves down. ESC quits.",
            font_size=12,
            x=24,
            y=20,
            color=(210, 210, 210, 255),
        )

        pyglet.clock.schedule_interval(self.update, 1 / 30)

    def update(self, dt: float) -> None:
        pitch = self.pitch_detector.latest()
        direction = self.chirp_detector.update(pitch.freq)

        if direction is None:
            return
        
        # selection logic fully implemented by me
        if direction == "up":
            self.selected = max(0, self.selected - 1)
        else:
            self.selected = min(self.item_count - 1, self.selected + 1)

        self.keyboard_output.press_arrow(direction)

    def on_key_press(self, symbol, modifiers) -> None:
        if symbol == pyglet.window.key.ESCAPE:
            self.close()

    def on_draw(self) -> None:
        self.clear()
        self.title.draw()

        start_y = 450
        height = 40
        gap = 10

        for i in range(self.item_count):
            y = start_y - i * (height + gap)
            selected = i == self.selected
            color = (255, 210, 80) if selected else (65, 85, 120)

            rect = shapes.Rectangle(150, y, 320, height, color=color)
            rect.draw()

            label = pyglet.text.Label(
                f"Menu item {i + 1}",
                font_size=14,
                x=310,
                y=y + 12,
                anchor_x="center",
                color=(20, 20, 25, 255) if selected else (235, 235, 240, 255),
            )
            label.draw()

        self.help.draw()


def list_input_devices() -> None:
    print("Available input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"{i}: {dev['name']}")


def main() -> None:
    # CLI structure suggested by AI (argparse-based design).
    # I adapted parameters for low-latency audio processing and usability.

    parser = argparse.ArgumentParser(description="Whistle chirp input.")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--samplerate", type=int, default=44100) # reduce sampling rate for lower latency
    parser.add_argument("--blocksize", type=int, default=1024) # reduce blocksize for lower latency
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    pitch_detector = PitchDetector(
        samplerate=args.samplerate,
        blocksize=args.blocksize,
        device=args.device,
    )
    chirp_detector = ChirpDetector()
    keyboard_output = KeyboardOutput(args.keyboard)

    pitch_detector.start()
    try:
        WhistleWindow(pitch_detector, chirp_detector, keyboard_output)
        pyglet.app.run()
    finally:
        pitch_detector.stop()


if __name__ == "__main__":
    main()
