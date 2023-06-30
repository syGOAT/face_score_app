import numpy as np
import io
import torch
from torchvision import transforms
from PIL import Image

from models.pfld import PFLDEncoder, DecoderFc, DecoderScore


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

def compute_nme(preds, target):

    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)

    for i in range(N):
        pts_pred, pts_gt = preds[i, ], target[i, ]
        if L == 19:  # aflw
            interocular = 34  # meta['box_size'][i]
        elif L == 29:  # cofw
            interocular = np.linalg.norm(pts_gt[8, ] - pts_gt[9, ])
        elif L == 68:  # 300w
            # interocular
            interocular = np.linalg.norm(pts_gt[36, ] - pts_gt[45, ])
        elif L == 98:
            interocular = np.linalg.norm(pts_gt[60, ] - pts_gt[72, ])
        else:
            raise ValueError('Number of landmarks is wrong')
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt,
                                        axis=1)) / (interocular * L)

    return rmse        


def main(face_b):
    encoder = PFLDEncoder().to(device)
    encoder.load_state_dict(torch.load('checkpoint/encoder.pth', map_location=device))
    decoder_fc = DecoderFc(176)
    decoder_fc.load_state_dict(torch.load('checkpoint/fc.pth', map_location=device))
    decoder_score = DecoderScore(176)
    decoder_score.load_state_dict(torch.load('checkpoint1.pth'))

    transform = transforms.Compose([
        transforms.Resize([112, 112]), 
        transforms.ToTensor(), 
    ])

    face = Image.open(io.BytesIO(face_b))
    face = transform(face)
    face = torch.unsqueeze(face, dim=0)

    encoder.eval()
    decoder_fc.eval()
    decoder_score.eval()

    with torch.no_grad():
        face = face.to(device)
        encoder = encoder.to(device)
        multi_scale = encoder(face)
        score = decoder_score(multi_scale).reshape(-1)
        score = torch.pow(3.9, score) + 25
        score = score if score < 99.8 else 99.8
        
    landmarks = decoder_fc(multi_scale).detach()
    landmarks = landmarks.reshape(landmarks.shape[0], -1, 2) # 1,98,2
    idols = idol_pipei(landmarks)

    return {
        'score': float(score), 
        'idols': idols
    }



idol_dict = {
    'zsw': 0, 
    'zjy': 0, 
    'zrf': 0, 
    'xlz': 0, 
    'lhq': 0, 
    'mm': 0

}

def idol_pipei(landmark):
    for idol in idol_dict:
        idol_landmark = torch.load('idol/{}.pt'.format(idol))
        near = torch.abs(landmark - idol_landmark)
        idol_dict[idol] = float(torch.sum(near))
    return idol_dict


if __name__ == "__main__":
    import requests
    url = 'https://focnal.xyz/static/out/f5af86fd491d4bcf961556bdf02c5236.jpg'
    face_b = requests.get(url).content
    res = main(face_b)
    print(res)
