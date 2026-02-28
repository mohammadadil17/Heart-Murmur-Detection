import librosa
import numpy as np

MAX_TIME = 256

def create_mel_spectrogram(file_path):

    signal, fs = librosa.load(file_path, sr=2000)

    mel_spec = librosa.feature.melspectrogram(
        y=signal,
        sr=fs,
        n_mels=128
    )

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize (same as notebook)
    mel_spec_db = (mel_spec_db - np.mean(mel_spec_db)) / np.std(mel_spec_db)

    # (Mel, Time) → (Time, Mel)
    mel_spec_db = mel_spec_db.T

    # Pad / Truncate to 256 time steps
    if mel_spec_db.shape[0] < MAX_TIME:
        pad = MAX_TIME - mel_spec_db.shape[0]
        mel_spec_db = np.pad(
            mel_spec_db,
            ((0, pad), (0, 0)),
            mode='constant'
        )
    else:
        mel_spec_db = mel_spec_db[:MAX_TIME, :]

    return mel_spec_db