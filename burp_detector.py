from collections.abc import Sequence
import librosa
import numpy as np
import typing
import math
from model_loader import BurpEvaluator
import soundfile as sf
from datetime import datetime
import os
import shutil
from tqdm import tqdm

RawAudio = np.ndarray


class AudioBuffer:
    buffer: RawAudio = np.ndarray((0, 2), np.float32)


class BurpDetector:
    sr: int = 44100
    buffer: RawAudio = np.ndarray((0, 2), dtype=np.float32)
    interest_size: int = 100000
    buffer_size: int = 10000
    slice_size: int = 1000
    slice_stride: int = 100
    last: int = 0
    buffer_start_sample: int = 0
    model: BurpEvaluator = None
    is_interesting: bool = False
    interest_started_at: int = 0
    interest_saved_parts: int = 0
    interest_burps_detected: int = 0
    interest_dir: str = 'interesting'
    slices_dir: str = 'detected_slices'
    window_size: int = 0
    required_votes: int = 0
    floating_window: "Sequence[int]" = []
    window_index: int = 0
    current_clip_number: int = 0
    max_total_votes: int = 0
    output: bool = False
    verbose: bool = False
    just_burp: bool = False


    def __init__(self, models: "Sequence[str]", slice_size: int, 
                 slice_stride: int, sample_rate: int, buffer_size: int = None, 
                 required_votes: int = None, window_size: int = None) -> None:
        
        self.sr = sample_rate
        self.slice_size = slice_size
        self.slice_stride = slice_stride

        if buffer_size is None:
            self.buffer_size = max(self.slice_size, self.slice_stride)
        else:
            self.buffer_size = buffer_size

        self.interest_size = 10 * 60 * self.sr

        if window_size is None:
            self.window_size = math.ceil(self.slice_size / self.slice_stride)
        else:
            self.window_size = window_size

        self.max_total_votes = self.window_size * len(models)

        if required_votes is None:
            self.required_votes = math.ceil(self.max_total_votes / 3)
        else:
            self.required_votes = required_votes
        
        self.floating_window = [0 for _ in range(self.window_size)]

        self.model = BurpEvaluator(models, sample_rate)

    
    def prepare_dirs(self):
        if os.path.exists(self.interest_dir):
            shutil.rmtree(self.interest_dir)

        os.makedirs(self.interest_dir)

        if os.path.exists(self.slices_dir):
            shutil.rmtree(self.slices_dir)

        os.makedirs(self.slices_dir)


    def process_audio(self, sig: np.ndarray, sr: int) -> RawAudio:
        # Convert to stereo
        if sig.ndim == 1:
            sig = np.repeat(np.expand_dims(sig, axis=0), 2, axis=0)

        if sig.ndim != 2 or 2 not in sig.shape:
            tqdm.write(F'Audio chunk array has wrong dimensions: {sig.shape}')
            return None

        if sig.shape[1] != 2:
            sig = sig.T

        if sr != self.sr:
            sig = librosa.resample(sig, orig_sr=sr, target_sr=self.sr, axis=0)

        return sig


    def append_data(self, sig: RawAudio) -> None:
        self.buffer = np.concatenate([self.buffer, sig])


    def get_slices(self) -> "Sequence[RawAudio]":
        slices = []
        # last = 3
        # stride = 2
        # slice_size = 6
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
        #       ^
        #      [3 4 5 6 7 8]
        #          [5 6 7 8 9 10]
        #              [7 8 9 10 11 12]
        #                  [9 10 11 12 13 14]
        #                        ^
        # last = 3
        # stride = 3
        # slice_size = 6
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17
        #       ^
        #      [3 4 5 6 7 8]
        #            [6 7 8 9 10 11]
        #                  [9 10 11 12 13 14]
        #                           ^
        for i in range(self.last, 
                       self.buffer.shape[0] - (self.slice_size - 1), 
                       self.slice_stride):
            
            slices += [self.buffer[i: i + self.slice_size, :]]
            self.last += self.slice_stride

        return slices


    def trim_buffer(self) -> None:
        if self.is_interesting:
            if self.buffer.shape[0] >= self.interest_size:
                self.save_interest()
            else:
                return
        
        if self.buffer.shape[0] <= self.buffer_size:
            return

        trim_amount = self.buffer.shape[0] - self.buffer_size
        self.last -= trim_amount
        self.buffer_start_sample += trim_amount
        self.buffer = self.buffer[trim_amount :, :]

    
    def evaluate_slice(self, slice: RawAudio) -> typing.Tuple[int, int]:
        detections = self.model.evaluate_array(slice.T)
        
        if detections > 0:
            self.save_slice(slice)
        
        self.floating_window[self.window_index] = detections

        if sum(self.floating_window) > 0:
            self.start_interest()
        else:
            self.stop_interest()

        text = ""
        for i in range(self.window_size):
            value = self.floating_window[(i + self.window_index + 1) % self.window_size]
            if len(self.model.models) <= 9:
                text += ' ' if value == 0 else str(value)
            else:
                text += '  |' if value == 0 else f'{value: >2}|'

        if self.max_total_votes > 9:
            text = text[:-1]

        chart = ''
        amount = sum(self.floating_window)
        for i in range(self.max_total_votes):
            if i < amount:
                chart += '='
            elif i == self.required_votes - 1:
                chart += ':'
            else:
                chart += ' '

        date = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S")

        if self.output and (self.verbose or sum(self.floating_window) != 0):
            tqdm.write(f"[{text}]<[{detections: ^3}] |{chart}| {date}")
        
        self.window_index = (self.window_index + 1) % self.window_size

        return sum([min(1, x) for x in self.floating_window]), sum(self.floating_window)


    def save_slice(self, slice: RawAudio) -> None:
        date = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S")
        filename = os.path.join(self.slices_dir, f'{date}_slice_{self.current_clip_number}.wav')

        sf.write(filename, slice, self.sr)

        self.current_clip_number += 1


    def start_interest(self) -> None:
        if self.is_interesting:
            return
        
        self.is_interesting = True
        self.interest_saved_parts = 0
        self.interest_burps_detected = 0
        self.interest_started_at = self.buffer_start_sample

    
    def to_ms(self, samples: int) -> int:
        return samples * 1000 // self.sr


    def save_interest(self) -> None:
        date = datetime.utcnow().strftime("%Y-%m-%d-%H:%M:%S")
        start_ms = self.to_ms(self.interest_started_at)
        end_ms = self.to_ms(self.interest_started_at + self.buffer.shape[0])
        filename = f'{date}_interest_{start_ms}ms_to_{end_ms}ms_{self.interest_saved_parts}pt_{self.interest_burps_detected}burps.wav'
        filename = os.path.join(self.interest_dir, filename)
        
        if self.interest_started_at == self.buffer_start_sample + self.buffer.shape[0]:
            return

        if self.interest_started_at <= self.buffer_start_sample: 
            sf.write(filename, self.buffer, self.sr)
        else:
            sf.write(filename, 
                     self.buffer[self.interest_started_at - self.buffer_start_sample :, :], 
                     self.sr)
        
        if self.output:
            tqdm.write(f'Saved section of interest to {filename}')

        self.interest_saved_parts += 1
        self.interest_burps_detected = 0
        self.interest_started_at = self.interest_started_at + self.buffer.shape[0]


    def stop_interest(self) -> None:
        if not self.is_interesting:
            return

        self.is_interesting = False
        self.save_interest()


    def add_audio_from_array(self, sig: np.ndarray, sr: int):
        # Resample and transpose if necessary
        audio = self.process_audio(sig, sr)

        if audio is None:
            return None

        # Append to buffer
        self.append_data(audio)

        # Generate slices if enough data
        slices = self.get_slices()

        detected = 0

        # Run through model
        for slice in slices:
            window_detections, window_total = self.evaluate_slice(slice)
            if window_total >= self.required_votes:
                if not self.just_burp:
                    detected += 1
                    self.interest_burps_detected += 1
                    self.just_burp = True
            else:
                self.just_burp = False

        self.trim_buffer()

        return None if len(slices) == 0 else detected
