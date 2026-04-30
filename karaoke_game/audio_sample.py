import sounddevice as sd
import numpy as np
import pyqtgraph as pg


# Set up audio stream
# reduce chunk size and sampling rate for lower latency
CHUNK_SIZE = 1024 # Number of audio frames per buffer
RATE = 44100 # Audio sampling rate (HZ)
CHANNELS = 1 # Mono audio

# print info about audio devices
print("Available input devices:\n")
devices = sd.query_devices()

input_devices = []
for i, dev in enumerate(devices):
    if dev['max_input_channels'] > 0:
        print(f"{i}: {dev['name']}")
        input_devices.append(i)

# let user select audio device
input_device = int(input("\nSelect input device: "))


# set up interactive plot
app = pg.mkQApp("Audio Visualizer")

win = pg.GraphicsLayoutWidget(title="Live Audio")
plot = win.addPlot()
plot.setYRange(-1, 1)

curve = plot.plot(pen='w')

win.show()


# audio callback to safe data
def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    data = indata[:, 0]  # mono
    curve.setData(data)


# open audio input stream
stream = sd.InputStream(
    device=input_device,
    channels=CHANNELS,
    samplerate=RATE,
    blocksize=CHUNK_SIZE,
    callback=audio_callback,
    latency='low'
)


# continously capture and plot audio signal
with stream:
    print("\nStreaming... (Ctrl+C to stop)")
    pg.exec()