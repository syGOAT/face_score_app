import torch
from torch import nn
from models.pfld import PFLDEncoder, DecoderFc, DecoderScore
from dataset import load_data_FBP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = PFLDEncoder().to(device)
encoder.load_state_dict(torch.load('checkpoint/encoder.pth', map_location=device))
decoder_fc = DecoderFc(176).to(device)
decoder_fc.load_state_dict(torch.load('checkpoint/fc.pth', map_location=device))
decoder_score = DecoderScore(176).to(device)
    
    

def train(train_loader, val_loader, num_epochs, device, loss, optimizer):
    for epoch in range(num_epochs):
        for X, y in train_loader:
            encoder.eval()
            decoder_score.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            encoder_output = encoder(X)
            y_hat = decoder_score(encoder_output)
            l = loss(y_hat, y).mean()
            l.backward()
            optimizer.step()
        print('epoch', epoch, 'train loss =', float(l.mean()))
        for X, y in val_loader:
            encoder.eval()
            decoder_score.eval()
            X, y = X.to(device), y.to(device)
            encoder_output = encoder(X)
            y_hat = decoder_score(encoder_output)
            vl = loss(y_hat, y).mean()
        print('val loss =', float(vl))

'''
def predict(loader, device):
    net.to(device)
    net.eval()
    num_attr_right, num_attr_s = 0, 0
    num_all_right, num_all = 0, 0
    num_likely = 0
    if not device:
        device = next(iter(net.parameters()) ).device
    with torch.no_grad():
        for X, y in loader:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            num_attr_s += y.numel()
            num_all += y.shape[0]
            _, _, y_hat = net(X)
            y_hat = torch.where(y_hat>0.5,torch.ones_like(y_hat),torch.zeros_like(y_hat))
            cmp =  y_hat.type(y.dtype) == y
            #print(cmp)
            num_attr_right += cmp.type(y.dtype).sum().item()
            num_all_right += cmp.all(dim=1).sum().item()
            for i in range(cmp.shape[0]):
                if cmp[i].sum() >= 8:
                    num_likely += 1
    return num_attr_right / num_attr_s, num_likely / num_all
                        #num_all_right / num_all, 完全匹配准确率
'''

if __name__ == '__main__':

    train_loader, val_loader, _, _ = load_data_FBP(batch_size=64)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(params=decoder_score.parameters(), lr=5e-4)

    train(train_loader, val_loader, 8, device, loss, optimizer)
    torch.save(decoder_score.state_dict(), 'checkpoint1.pth')