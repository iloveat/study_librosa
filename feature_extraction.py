# Feature extraction example
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


# Load the audio
y, sr = librosa.load('test.wav')

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
hop_length = 512

# Separate harmonics and percussives into two waveforms
y_harmonic, y_percussive = librosa.effects.hpss(y)
librosa.output.write_wav('y_harmonic.wav', y_harmonic, sr)
librosa.output.write_wav('y_percussive.wav', y_percussive, sr)

# Beat track on the percussive signal
tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
librosa.output.times_csv('beat_times.csv', beat_times)

# Compute MFCC features from the raw signal
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
plt.figure()
librosa.display.specshow(mfcc, x_axis='time')
plt.title('MFCC')
plt.colorbar()
plt.tight_layout()
plt.show()

# And the first-order differences (delta features)
mfcc_delta = librosa.feature.delta(mfcc)

# Stack and synchronize between beat events
# This time, we'll use the mean value (default) instead of median
beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

# Compute chroma features from the harmonic signal
chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

# Aggregate chroma features between beat events
# We'll use the median value of each feature between beat frames
beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)

# Finally, stack all beat-synchronous features together
beat_features = np.vstack([beat_chroma, beat_mfcc_delta])





