import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from data import get_loader
from utils import clip_gradient, adjust_lr
import logging
import torch.backends.cudnn as cudnn
from options import opt
from utils import print_network
from models.SANet import SANet
from utils import hybrid_e_loss
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_ssim
import pytorch_iou
from data import Test_dataset
import torch.utils.data as data



def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

cudnn.benchmark = True

#build the model

model = SANet()
model.load_pre('./swin_base_patch4_window12_384_22k.pth')

print_network(model, 'SANet')
# if(opt.load is not None):
#     model.load_state_dict(torch.load(opt.load))
#     print('load model from ', opt.load)
model= nn.DataParallel(model).cuda()

params1, params2=[],[]
for name, param in model.named_parameters():
    if check_keywords_in_name(name, ('rgb_swin')):
        params1.append(param)
    else:
        params2.append(param)


optimizer = torch.optim.Adam([{'params':params1, 'lr': opt.lr*0.1},{'params':params2}], opt.lr)



milestones=[60,60+12,60+12*2,60+12*3,60+12*4]
scheduler_focal = MultiStepLR(optimizer, milestones, gamma=0.7, last_epoch=-1)

#set the path
image_root = './VT5000/VT5000_clear/Train/RGB/'
gt_root = './VT5000/VT5000_clear/Train/GT/'
depth_root= './VT5000/VT5000_clear/Train/T/'
save_path='./BBSNet_cpts/'

if not os.path.exists(save_path):
    os.makedirs(save_path)

#load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, batchsize= 8, trainsize= 384)

total_step = len(train_loader)

#set loss function
CE= torch.nn.BCEWithLogitsLoss()
ssim_loss = pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = pytorch_iou.IOU(size_average=True)

def only_iou_loss(pred,target):

    pred= torch.sigmoid(pred)
    # ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)

    loss = iou_out

    return loss


step = 0

best_mae = 1
best_epoch = 1


#train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, depths) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

            pred1,pred2,pred3,pred4,pred5,pred6,pred7= model(images,depths)

            loss2= CE(pred2, gts)+ only_iou_loss(pred2, gts)
            loss3= CE(pred3, gts)+ only_iou_loss(pred3, gts)
            loss4= CE(pred4, gts)+ only_iou_loss(pred4, gts)
            loss5= CE(pred5, gts)+ only_iou_loss(pred5, gts)
            loss6= CE(pred6, gts)+ only_iou_loss(pred6, gts)
            loss7= CE(pred7, gts)+ only_iou_loss(pred7, gts)

           

            loss = loss2+loss3+loss4+loss5*(1./4.)+loss6*(1./16.)+loss7*(1./64.)
            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            if i % 10 == 0 or i == total_step:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, Loss1: {:.4f}, loss_edge: {:.4f},'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss2.data, 0))

        if (epoch) % 1 == 0:
            torch.save(model.state_dict(), save_path+'BBSNet_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt: 
        print('Keyboard Interrupt: save model and exit.')
        print('save checkpoints successfully!')
        raise


 
if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch+1):
        
        print('learning_rate', optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'])
        train(train_loader, model, optimizer, epoch, save_path)
        scheduler_focal.step()

