import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import simps

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from PIL import Image

from models.pfld import PFLDEncoder, DecoderFc, DecoderScore

cudnn.benchmark = True
cudnn.determinstic = True
cudnn.enabled = True
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')

def compute_nme(preds, target):
    """ preds/target:: numpy array, shape is (N, L, 2)
        N: batchsize L: num of landmark 
    """
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


def validate(mypic, encoder, decoder_fc, decoder_score):
    encoder.eval()
    decoder_fc.eval()
    decoder_score.eval()

    cost_time = []
    with torch.no_grad():
        mypic = mypic.to(device)
        encoder = encoder.to(device)

        start_time = time.time()
        multi_scale = encoder(mypic)
        score = decoder_score(multi_scale)
        print(score)
        landmarks = decoder_fc(multi_scale)
        cost_time.append(time.time() - start_time)

        landmarks = landmarks.cpu().numpy()
        landmarks = landmarks.reshape(landmarks.shape[0], -1,
                                        2)  # landmark
     
        if show_image:
            show_mypic = np.array(
                np.transpose(mypic[0].cpu().numpy(), (1, 2, 0)))
            show_mypic = (show_mypic * 255).astype(np.uint8)
            np.clip(show_mypic, 0, 255)

            pre_landmark = landmarks[0] * [112, 112]

            show_mypic = cv2.cvtColor(show_mypic, cv2.COLOR_RGB2BGR)
            #cv2.imwrite("show_mypic.jpg", show_mypic)
            #mypic_clone = cv2.imread("show_mypic.jpg")

            for (x, y) in pre_landmark.astype(np.int32):
                cv2.circle(show_mypic, (x, y), 1, (255, 0, 0), -1)
            cv2.imshow("show_mypic.jpg", show_mypic)
            cv2.waitKey(0)
  


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
    mypic = Image.open(mypic)
    mypic = transform(mypic)

    mypic = torch.unsqueeze(mypic, dim=0)

    validate(mypic, encoder, decoder_fc, decoder_score)


show_image = True


if __name__ == "__main__":
    mypic = 'data/SCUT-FBP5500_v2/Images/CM423.jpg'
    main(mypic)
