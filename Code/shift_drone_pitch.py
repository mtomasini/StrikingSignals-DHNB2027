"""
In this file I wrote a function to simply synthetize a sound / drone for each of the 12 notes. Each sound bite is saved in
the folder `./drones_samples/`.

We use the original drone from Zeitgeist at time 39:04 as base for our library, and shift the pitch.
"""

import librosa
import soundfile

notes = librosa.key_to_notes("C:maj")
original_drone, sr_original_drone = librosa.load('../Data/drone_zeitgeist_003904.wav')

notes = ['G', 'Gs', 'A', 'As', 'B', 'C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs']
shift = range(-7, 5)
notes_dictionary = dict(zip(notes, shift))

for note in notes:
    pitched_drone = librosa.effects.pitch_shift(original_drone, sr=sr_original_drone, n_steps = notes_dictionary[note])
    soundfile.write(f'../Data/drones_samples/{note}_drone.wav', pitched_drone, format = 'wav', samplerate=sr_original_drone)