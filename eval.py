from model import *
import torch

model = AudioClassifier()
model.load_state_dict(torch.load("./model.pt"))
model.eval()
