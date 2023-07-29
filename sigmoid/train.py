from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchaudio
from torchaudio import transforms
import os
import torch.nn.functional as F
from torch.nn import init
from torch import nn
import random

from tqdm import trange, tqdm

from model import *


BATCH_SIZE = 64 # Tweak appropriately to your VRAM
EPOCHS = 500
TRAINING_VALIDATION_SPLIT = 0.8


def time_shift(aud, shift_limit):
    sig, sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)


def speed_transform(aud):
    sig, sr = aud
    
    speed_factor = random.choice([0.9, 1.0, 1.1])
    if speed_factor == 1.0: # no change
        return sig, sr
    
    sox_effects = [
        ["speed", str(speed_factor)],
        ["rate", str(sr)],
    ]
    
    return torchaudio.sox_effects.apply_effects_tensor(sig, sr, sox_effects)


def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)


def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    return aug_spec


class SoundDS(Dataset):
    def __init__(self, burps_files, not_burps_files):
        self.burps_files = burps_files
        self.not_burps_files = not_burps_files
        self.duration = 4000
        self.sr = 44100
        self.channel = 2

    def __len__(self):
        return len(self.burps_files) + len(self.not_burps_files)

    def __getitem__(self, idx):
        if idx < len(self.burps_files):
            audio_file = self.burps_files[idx]
            class_id = torch.tensor([1.0])
        else:
            audio_file = self.not_burps_files[idx - len(self.burps_files)]
            class_id = torch.tensor([0.0])

        aud = torchaudio.load(audio_file)
        shift_aud = time_shift(aud, 0.4)
        spec = spectro_gram(shift_aud)
        aug_spec = spectro_augment(spec, 0.1, 2)

        return aug_spec, class_id


class SoundDSOptimized(Dataset):
    def __init__(self, burps_tensors, not_burps_tensors):
        self.burps_tensors = burps_tensors
        self.not_burps_tensors = not_burps_tensors
        self.sr = 44100
        self.channel = 2

    def __len__(self):
        return len(self.burps_tensors) + len(self.not_burps_tensors)

    def __getitem__(self, idx):
        if idx < len(self.burps_tensors):
            audio_file = self.burps_tensors[idx]
            class_id = torch.tensor([1.0])
        else:
            audio_file = self.not_burps_tensors[idx - len(self.burps_tensors)]
            class_id = torch.tensor([0.0])

        shift_aud = time_shift(audio_file, 0.4)
        spec = spectro_gram(shift_aud)
        aug_spec = spectro_augment(spec, 0.1, 2)

        return aug_spec, class_id


def training(model, train_dl, num_epochs, val_dl):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01,
                                                    steps_per_epoch=int(
                                                        len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold=1e-3)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
    #                                                 steps_per_epoch=int(
    #                                                     len(train_dl)),
    #                                                 epochs=num_epochs,
    #                                                 anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in trange(num_epochs, unit='epoch', dynamic_ncols=True, leave=False):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        correct_true_prediction = 0
        correct_false_prediction = 0
        total_true_prediction = 0
        total_false_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            #inputs_m, inputs_s = inputs.mean(), inputs.std()
            #inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            prediction = torch.greater(outputs, 0.5)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            correct_true_prediction += ((prediction == labels) & (prediction == 1)).sum().item()
            correct_false_prediction += ((prediction == labels) & (prediction == 0)).sum().item()
            total_true_prediction += (labels == 1).sum().item()
            total_false_prediction += (labels == 0).sum().item()

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        acc_true = correct_true_prediction/total_true_prediction
        acc_false = correct_false_prediction/total_false_prediction
        tqdm.write(f'Epoch: {epoch}, '
              f'Loss: {avg_loss:.4f}, '
              f'Accuracy: {acc:.4f} '
              f'[True: {correct_true_prediction}/{total_true_prediction} {acc_true:.4f}, '
              f'False: {correct_false_prediction}/{total_false_prediction} {acc_false:.4f}] '
              f'LR: {scheduler.get_last_lr()}')

        if epoch % 10 == 0:
            tqdm.write('Validation')
            model.eval()
            inference(model, val_dl)
            model.train()


    print('Finished Training')


def inference(model, val_dl):
    correct_true_prediction = 0
    correct_false_prediction = 0
    total_true_prediction = 0
    total_false_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            #inputs_m, inputs_s = inputs.mean(), inputs.std()
            #inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            prediction = torch.greater(outputs, 0.5)
            # Count of predictions that matched the target label
            correct_true_prediction += ((prediction == labels) & (prediction == 1)).sum().item()
            correct_false_prediction += ((prediction == labels) & (prediction == 0)).sum().item()
            total_true_prediction += (labels == 1).sum().item()
            total_false_prediction += (labels == 0).sum().item()

    acc = (correct_true_prediction + correct_false_prediction) / (total_true_prediction + total_false_prediction)
    acc_true = correct_true_prediction/total_true_prediction
    acc_false = correct_false_prediction/total_false_prediction
    tqdm.write(f'Accuracy: {acc:.4f} '
          f'[True: {correct_true_prediction}/{total_true_prediction} {acc_true:.4f}, '
          f'False: {correct_false_prediction}/{total_false_prediction} {acc_false:.4f}] '
          f'Total items: {total_true_prediction + total_false_prediction}')


burps_folder_path = "../burps-audio"
burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]

burps_folder_path = "../not-burps-audio"
not_burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]

#Store all of the data in memory, if not enough ram available, comment out
print('Loading burp tensors')
burp_tensors = []
for file in tqdm(burps_files, unit='file', dynamic_ncols=True, leave=False):
    burp_tensors += [torchaudio.load(file)]

print('Loading non-burp tensors')
non_burp_tensors = []
for file in tqdm(not_burps_files, unit='file', dynamic_ncols=True, leave=False):
    non_burp_tensors += [torchaudio.load(file)]

print('Starting')
myds = SoundDSOptimized(burp_tensors, non_burp_tensors) # If not enough ram available, use the below line instead
# myds = SoundDS(burps_files, not_burps_files)

num_items = len(myds)
num_train = round(num_items * TRAINING_VALIDATION_SPLIT)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

num_epochs = EPOCHS  # Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs, val_dl)
print('Done')

myModel.eval()
# Run inference on trained model with the validation set
inference(myModel, val_dl)

torch.save(myModel.state_dict(), "model.pt")
