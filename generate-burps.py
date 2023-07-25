import os
import glob
import numpy as np
import random
import librosa
import soundfile as sf

directory_path = "burps-audio/"
wav_files = glob.glob(os.path.join(directory_path, '*.wav'))

random.seed(42)
for file_path in wav_files:
    filename = os.path.basename(file_path)
    part = filename.split('.')[0]
    for i in range(2, 5):
        new_file_path = f"{directory_path}/{part}_{str(i)}.wav"
        audio_data, sr = librosa.load(file_path, sr=None, mono=False)

        # Convert the stereo audio data into two separate channels
        channel_1 = audio_data[0]
        channel_2 = audio_data[1]

        # Calculate the sample index corresponding to the burp time
        time = np.linspace(0, 4, len(channel_1))
        burp_sample = np.argmax(np.abs(channel_1))
        burp_time = time[burp_sample]
        burp_sample = int(burp_time * sr)

        # Crop a one-second window around the burp time
        window_size = sr  # One second window size
        window_start = burp_sample - int(window_size / 2)
        window_end = window_start + window_size

        # Make sure the window does not go out of bounds
        window_start = max(window_start, 0)
        window_end = min(window_end, len(channel_1))

        # Crop the audio data for both channels
        cropped_channel_1 = channel_1[window_start:window_end]
        cropped_channel_2 = channel_2[window_start:window_end]

        # Generate random periods of silence
        silence_before = np.random.uniform(0.5, 2.5)  # seconds
        silence_after = 3.0 - silence_before
        random_gain = np.random.uniform(0.5, 2.0)

        # Calculate the number of samples for the silent periods
        silence_samples_before = int(silence_before * sr)
        silence_samples_after = int(silence_after * sr)
        padding_samples = len(channel_1) - silence_samples_before - silence_samples_after - sr
        silence_samples_after += padding_samples

        # Generate the silent periods as empty arrays
        silence_before_array = np.zeros(silence_samples_before)
        silence_after_array = np.zeros(silence_samples_after)

        cropped_channel_1 = cropped_channel_1 * random_gain
        cropped_channel_2 = cropped_channel_2 * random_gain

        # Concatenate the audio arrays: silence_before + cropped_audio + silence_after
        final_stereo_audio_1 = np.concatenate((silence_before_array, cropped_channel_1, silence_after_array))
        final_stereo_audio_2 = np.concatenate((silence_before_array, cropped_channel_2, silence_after_array))

        # Create a new stereo audio array with the cropped channels
        cropped_stereo_audio = np.array([final_stereo_audio_1, final_stereo_audio_2])

        # If needed, you can save the cropped stereo audio back to a WAV file
        sf.write(new_file_path, cropped_stereo_audio.T, sr)