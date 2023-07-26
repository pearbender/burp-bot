from torch.utils.data import DataLoader, Dataset, random_split
import torch
import torchaudio
from torchaudio import transforms
import os
import torch.nn.functional as F
from torch.nn import init
from torch import nn

from model import *


BATCH_SIZE = 64
EPOCHS = 70
TRAINING_VALIDATION_SPLIT = 0.9


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
            class_id = 0
        else:
            audio_file = self.not_burps_files[idx - len(self.burps_files)]
            class_id = 1

        sig, sr = torchaudio.load(audio_file)
        spec = transforms.MelSpectrogram(
            sr, n_fft=1024, hop_length=None, n_mels=64)(sig)
        spec = transforms.AmplitudeToDB(top_db=80)(spec)

        return spec, class_id


def training(model, train_dl, num_epochs, val_dl):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                    steps_per_epoch=int(
                                                        len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
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
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            correct_true_prediction += ((prediction == labels) & (prediction == 0)).sum().item()
            correct_false_prediction += ((prediction == labels) & (prediction == 1)).sum().item()
            total_true_prediction += (labels == 0).sum().item()
            total_false_prediction += (labels == 1).sum().item()

            # if i % 10 == 0:    # print every 10 mini-batches
            #    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        acc_true = correct_true_prediction/total_true_prediction
        acc_false = correct_false_prediction/total_false_prediction
        print(f'Epoch: {epoch}, '
              f'Loss: {avg_loss:.4f}, '
              f'Accuracy: {acc:.4f} '
              f'[True: {correct_true_prediction}/{total_true_prediction} {acc_true:.4f}, '
              f'False: {correct_false_prediction}/{total_false_prediction} {acc_false:.4f}]')

        if epoch % 5 == 0:
            print('Validation')
            inference(model, val_dl)


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
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_true_prediction += ((prediction == labels) & (prediction == 0)).sum().item()
            correct_false_prediction += ((prediction == labels) & (prediction == 1)).sum().item()
            total_true_prediction += (labels == 0).sum().item()
            total_false_prediction += (labels == 1).sum().item()

    acc = (correct_true_prediction + correct_false_prediction) / (total_true_prediction + total_false_prediction)
    acc_true = correct_true_prediction/total_true_prediction
    acc_false = correct_false_prediction/total_false_prediction
    print(f'Accuracy: {acc:.4f} '
          f'[True: {correct_true_prediction}/{total_true_prediction} {acc_true:.4f}, '
          f'False: {correct_false_prediction}/{total_false_prediction} {acc_false:.4f}] '
          f'Total items: {total_true_prediction + total_false_prediction}')


burps_folder_path = "./burps-audio"
burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]

burps_folder_path = "./not-burps-audio"
not_burps_files = [os.path.join(burps_folder_path, file) for file in os.listdir(
    burps_folder_path) if file.lower().endswith(".wav")]

myds = SoundDS(burps_files, not_burps_files)
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

# Run inference on trained model with the validation set
inference(myModel, val_dl)

torch.save(myModel.state_dict(), "model.pt")
