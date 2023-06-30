from typing import Any
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys


img_path = 'data/SCUT-FBP5500_v2/Images'
train_txt = 'data/SCUT-FBP5500_v2/train.txt'
val_txt = 'data/SCUT-FBP5500_v2/test.txt'


class MyDataset(Dataset):
    def __init__(self, split, transform=None) :
        imgs = []
        if split == 'train':
            des_txt = train_txt
        elif split == 'valid':
            des_txt = val_txt
        else:
            print('Split must be train or balid!')
            sys.exit(1)
        
        with open(des_txt, 'r')as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                imgs.append((words[0], torch.tensor([float(words[1])], dtype=torch.float32)))

        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)
        
    def __getitem__(self, index):
        pic, score = self.imgs[index]
        pic = Image.open(img_path + '/' + pic)
        pic = self.transform(pic)
        return pic, score
    

def load_data_FBP(batch_size, ):
    trans = [
        transforms.RandomPosterize(bits=2, p=0.2), 
        transforms.Resize([112, 112]),
        transforms.ToTensor(), 
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
    ]
    train_trans = transforms.Compose(trans + 
        [transforms.RandomHorizontalFlip(), transforms.RandomGrayscale()])
    val_trans = transforms.Compose(trans)
    train_data = MyDataset(split='train', transform=train_trans)
    val_data = MyDataset(split='valid', transform=val_trans)
    return DataLoader(train_data, batch_size=batch_size, shuffle=True), \
           DataLoader(val_data, batch_size=batch_size), \
           train_data.__len__(), val_data.__len__()
        

if __name__ == '__main__':
    _, val_data, train_len, val_len = load_data_FBP(64)
    print('train_len = {}, val_len = {}'.format(train_len, val_len))
    for X, y in val_data:
        print(y)
        import torchvision.transforms.functional as F
        for i in range(10):
            demo_img = F.to_pil_image(X[i])
            demo_img.show()
        break
