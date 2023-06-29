import numpy as np

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


def main(mypic):
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

    mypic = transform(mypic)
    mypic = torch.unsqueeze(mypic, dim=0)

    encoder.eval()
    decoder_fc.eval()
    decoder_score.eval()

    with torch.no_grad():
        mypic = mypic.to(device)
        encoder = encoder.to(device)
        multi_scale = encoder(mypic)
        score = decoder_score(multi_scale).reshape(-1)
        score = torch.pow(3.9, score) + 25
        score = score if score < 99.8 else 99.8
        return float(score)



if __name__ == "__main__":
    mypic = 'data/SCUT-FBP5500_v2/Images/CM423.jpg'
    mypic = Image.open(mypic)
    res = main(mypic)
    print(res)
