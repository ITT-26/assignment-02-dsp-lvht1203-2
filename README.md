[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/B3oR_XLF)
# ITT Karaoke Game

## Installation

```bash
pip install numpy mido pyglet sounddevice
```

## Usage
```bash
python karaoke.py read_midi/berge.mid
```

## Controls
- SPACE: Start/Pause
- R: Restart
- ESC: Quit

## Gameplay
- Blue lane: target notes from MIDI
- Yellow lane: your detected pitch
- Try to align both to get a higher score

## How it works
- MIDI file → converted into note events (time + frequency)
- Microphone → pitch detected using autocorrelation
- Pitch is compared to the target using cents (log scale)
- Score increases when the pitch is close


# Whistle Input

## Installation
```bash
pip install numpy pyglet sounddevice pynput
```

## Usage

### Run the GUI:
```bash
python whistle-input.py
```

### Enable keyboard events:
```bash
python whistle-input.py --keyboard
```

## Controls
- Upward chirp → move up
- Downward chirp → move down
- ESC → quit

## How it works
- The microphone signal is analyzed in real time
- The dominant frequency is detected from the audio signal
- Increasing frequency = upward chirp
- Decreasing frequency = downward chirp

Noise is reduced using:
- RMS threshold
- peak-to-noise filtering
- cooldown between detections
