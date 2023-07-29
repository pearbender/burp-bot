import numpy as np
import os
import shutil
import librosa
import soundfile as sf
from datetime import datetime


TEMP_SLICE_DIR = './temp-rt-clips'


class AudioBuffer:
    def __init__(self, size: int) -> None:
        self.max_size = size
        self.buffer = None
        self.current_clip_number = 0
        
        if os.path.exists(TEMP_SLICE_DIR):
            shutil.rmtree(TEMP_SLICE_DIR)

        os.makedirs(TEMP_SLICE_DIR)

    
    def add_clip(self, clip: np.ndarray) -> None:
        if self.buffer is None:
            self.buffer = clip
            return
        
        self.buffer = np.concatenate([self.buffer, clip])

        if self.buffer.size > self.max_size:
            self.buffer = self.buffer[-self.max_size:]


    def save_latest(self, sr: int, samples: int) -> None:
        if self.buffer is None or self.buffer.size < samples:
            print("Not enough audio in buffer yet")
            return None 

        date = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S")
        filename = os.path.join(TEMP_SLICE_DIR, f'{date}_slice_{self.current_clip_number}.wav')

        sf.write(filename, self.buffer[-samples:], sr)

        self.current_clip_number += 1

        return filename
