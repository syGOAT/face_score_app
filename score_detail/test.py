import torch
from torch import nn
from models.pfld import PFLDEncoder, DecoderFc, DecoderScore
from dataset import load_data_FBP
import math


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = PFLDEncoder().to(device)
encoder.load_state_dict(torch.load('checkpoint/encoder.pth', map_location=device))
decoder_fc = DecoderFc(176).to(device)
decoder_fc.load_state_dict(torch.load('checkpoint/fc.pth', map_location=device))
decoder_score = DecoderScore(176).to(device)
decoder_score.load_state_dict(torch.load('checkpoint1.pth'))
encoder.eval()
decoder_fc.eval()
#decoder_score.eval()
train_loader, val_loader, _, _ = load_data_FBP(batch_size=64)


for X, y in val_loader:
    decoder_score.eval()
    X, y = X.to(device), y.to(device)
    encoder_output = encoder(X)
    y_hat = decoder_score(encoder_output)
    s = y_hat.reshape(-1)
    s = torch.pow(3.9, s) + 25
    s[s > 99.8] = 99.8
    print(s)
    break