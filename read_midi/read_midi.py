import mido
from mido import MidiFile

for msg in MidiFile('berge.mid').play():
    print(msg)
