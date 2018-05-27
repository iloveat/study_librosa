# Beat tracking example
from __future__ import print_function
import librosa


# 1. Load the audio as a waveform `y`
#    Store the sampling rate as `sr`
y, sr = librosa.load('test.wav')

# 2. Run the default beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print('Estimated tempo: {:.2f} beats per minute'.format(tempo))

# 3. Convert the frame indices of beat events into timestamps
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

print('Saving output to beat_times.csv')
librosa.output.times_csv('beat_times.csv', beat_times)



