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
    # Detect the pitch of incoming audio using a simple FFT-based method. 
    # It applies a Hanning window to the audio data, computes the FFT, and looks
    # for the peak in the spectrum within a specified frequency range. 
    # It also applies a smoothing filter to the detected frequency to reduce jitter,
    # and uses an RMS threshold to ignore low-volume noise. The latest detected pitch is stored in a thread-safe manner for retrieval by the main game loop.
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

        self._latest = Pitch()
        self._smooth_freq: Optional[float] = None
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None

    # Start the audio input stream and begin processing audio data. 
    # The callback function will be called for each block of audio data, where it will estimate the pitch 
    # and update the latest detected pitch in a thread-safe manner. 
    # The stream is configured for low latency to ensure responsive pitch detection for the game.
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

    # Stop the audio input stream and clean up resources. 
    # This should be called when the program is exiting to ensure that the audio stream is properly closed.
    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    # Retrieve the latest detected pitch in a thread-safe manner. 
    # This method can be called by the main game loop to get the current pitch frequency, 
    # which will be used to update the microphone lane and detect chirps. 
    # It returns a Pitch object containing the frequency, or None if no valid pitch is currently detected.
    def latest(self) -> Pitch:
        with self._lock:
            return Pitch(self._latest.freq)

    # The audio callback function that processes incoming audio data. 
    # It estimates the pitch of the audio block and updates the latest detected pitch. 
    # The pitch estimation is done by applying a Hanning window to the audio data, computing the FFT, 
    # and finding the peak in the spectrum within the specified frequency range. 
    # The detected frequency is smoothed over time to reduce jitter, and low-volume noise is ignored 
    # based on the RMS threshold. The latest pitch is stored in a thread-safe manner for retrieval by the main game loop
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

    # Estimate the pitch of a block of audio data using an FFT-based method. 
    # It applies a Hanning window to the audio data, computes the FFT, and looks
    # for the peak in the spectrum within the specified frequency range. 
    # It also applies an RMS threshold to ignore low-volume noise, and checks the strength of the detected peak against the noise floor to avoid false detections. 
    # If a valid pitch is detected, it returns the frequency in Hz; otherwise, it returns None.
    def _estimate_pitch(self, data: np.ndarray) -> Optional[float]:
        data = data - np.mean(data)
        rms = float(np.sqrt(np.mean(data * data)))
        if rms < self.rms_threshold:
            return None

        # FFT-based pitch estimation adapted with help from AI explanations and online DSP references.
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

        # Ignore weak peaks caused by background noise.
        if peak_value / noise_floor < 4.0: # Noise thresholds were adjusted experimentally during testing.
            return None

        global_index = np.where(mask)[0][0] + peak_index
        freq = global_index * self.samplerate / len(data)

        if self.min_freq <= freq <= self.max_freq:
            return float(freq)

        return None


# Detect chirps based on the history of detected pitches. 
# It looks for a consistent increase or decrease in pitch over a short time window,
# and applies thresholds on the minimum frequency change and slope to determine if a valid chirp has occurred. 
# It also implements a cooldown period after detecting a chirp to prevent multiple triggers from the same gesture.
class ChirpDetector:
    def __init__(
        self,
        window_seconds: float = 0.75, # time window to analyze for chirp patterns
        min_points: int = 5, # minimum number of pitch detections required in the window to consider it a valid chirp
        min_freq_change: float = 120.0, # minimum frequency change in Hz to consider it a valid chirp
        min_slope: float = 120.0, # minimum slope of the frequency change in Hz/second to consider it a valid chirp
        cooldown_seconds: float = 0.55, # minimum time between consecutive chirp detections to prevent multiple triggers from the same gesture
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

        while self.history and now - self.history[0][0] > self.window_seconds:
            self.history.popleft()

        if now - self.last_trigger < self.cooldown_seconds:
            return None

        direction = self.detect()
        if direction is not None:
            self.last_trigger = now
            self.history.clear()

        return direction

    # Analyze the history of detected pitches to determine if a valid chirp has occurred. 
    # It checks if there are enough data points, if the frequency change exceeds the minimum threshold, 
    # and if the slope of the frequency change is consistent with an upward or downward chirp.
    def detect(self) -> Optional[str]:
        if len(self.history) < self.min_points:
            return None

        times = np.array([x[0] for x in self.history])
        freqs = np.array([x[1] for x in self.history])
        times = times - times[0]

        if times[-1] <= 0:
            return None

        # Chirp direction is estimated from pitch slope over time.
        # The implementation was refined experimentally during testing.
        freq_change = float(freqs[-1] - freqs[0])
        slope = float(np.polyfit(times, freqs, 1)[0])

        if abs(freq_change) < self.min_freq_change:
            return None
        if abs(slope) < self.min_slope:
            return None

        return "up" if slope > 0 else "down"

# Handle keyboard output for simulating arrow key presses based on detected chirps. 
# If enabled, it uses the pynput library to send keyboard events to the system. 
# The press_arrow method can be called with the direction of the detected chirp to simulate 
# an arrow key press, which can be used to control the game based on the player's whistle gestures.
class KeyboardOutput:
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


# The main window of the whistle input application. 
# It displays a simple interface with instructions and visual feedback for the detected pitch and chirps.
class WhistleWindow(pyglet.window.Window):
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

# List available audio input devices to the console. 
# This can be used by the user to identify which device index corresponds to their microphone for use with the pitch detector.
def list_input_devices() -> None:
    print("Available input devices:\n")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"{i}: {dev['name']}")

# The main entry point of the program. Parses command line arguments for audio device, 
# sampling rate, block size, and other settings.
# Initializes the pitch detector and chirp detector, and starts the Pyglet application loop with the WhistleWindow. 
# It also handles listing audio devices if requested, 
# and ensures that the audio stream is properly stopped when the program exits.   
def main() -> None:
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
