import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
import os, argparse
import cv2
from models.SANet import SANet
from data import Test_dataset, Test_dataset2
import torch.utils.data as data


parser = argparse.ArgumentParser()

parser.add_argument('--test_path',type=str,default='./VT5000/VT5000_clear/Test/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path


# load the model
model = nn.DataParallel(SANet())
image_root = dataset_path + '/RGB/'
gt_root = dataset_path +'/GT/'
depth_root=dataset_path +'/T/'
test_dataset=Test_dataset(image_root, gt_root,depth_root, 384)
batch= 1
data_loader= data.DataLoader(dataset=test_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
for checkp in range(1, 120+1):
    if checkp % 1 ==0:
    
        with torch.no_grad():
            modelpath= './BBSNet_cpts/BBSNet_epoch_' + str(checkp)+ '.pth'
            model.load_state_dict(torch.load(modelpath))
        
            model.cuda()
            model.eval()
            save_path = './test_maps/VT5000/'+str(checkp)+ '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            loss_e= 0

            for i, (image, gt_shape, depth, name) in enumerate(data_loader):
                print('batch {:4d}'.format(i))
                image = image.cuda()
                depth = depth.cuda()
                
                bz= image.shape[0]
                pred,_,_,_,_,_,_ = model(image, depth)

                for j in range(bz):
                    res2= pred[j]
                    res2 = torch.sigmoid(res2).data.cpu().numpy().squeeze()
                    name2= name[j]
                    res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
                    res2= res2*255
                    cv2.imwrite(save_path+ name2, res2)
            print('Test Done!', checkp)


#####VT1000
dataset_path= './VT1000/'
image_root = dataset_path + '/RGB/'
gt_root = dataset_path +  '/GT/'
depth_root=dataset_path +'/T/'
test_dataset=Test_dataset2(image_root, gt_root,depth_root, 384)
batch= 1
data_loader= data.DataLoader(dataset=test_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
for checkp in range(1, 120+1):
    if checkp % 1 ==0:
        with torch.no_grad():
            modelpath= './BBSNet_cpts/BBSNet_epoch_' + str(checkp)+ '.pth'
            model.load_state_dict(torch.load(modelpath))

            model.cuda()
            model.eval()
            save_path = './test_maps/VT1000/'+str(checkp)+ '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            loss_e= 0

            for i, (image, gt_shape, depth, name) in enumerate(data_loader):
    
                print('batch {:4d}'.format(i))
                image = image.cuda()
                depth = depth.cuda()
                bz= image.shape[0]
                pred,_,_,_,_,_,_ = model(image, depth)

                for j in range(bz):
                    res2= pred[j]
                    res2 = torch.sigmoid(res2).data.cpu().numpy().squeeze()
                    name2= name[j]
                    res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
                    res2= res2*255
                    cv2.imwrite(save_path+ name2, res2)

            print('Test Done!', checkp)

# # #####VT821
dataset_path= './VT821/'
image_root = dataset_path + '/RGB/'
gt_root = dataset_path +  '/GT/'
depth_root=dataset_path +'/T/'
test_dataset=Test_dataset2(image_root, gt_root,depth_root, 384)
batch= 1
data_loader= data.DataLoader(dataset=test_dataset,
    batch_size=batch,
    shuffle=False,
    num_workers=4,
    pin_memory=True)
for checkp in range(1, 120+1):
    if checkp % 1 ==0:
        with torch.no_grad():
            modelpath= './BBSNet_cpts/BBSNet_epoch_' + str(checkp)+ '.pth'
            model.load_state_dict(torch.load(modelpath))
            model.cuda()
            model.eval()
            save_path = './test_maps/VT821/'+str(checkp)+ '/'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            loss_e= 0
            for i, (image, gt_shape, depth, name) in enumerate(data_loader):
                print('batch {:4d}'.format(i))
                image = image.cuda()
                depth = depth.cuda()
                bz= image.shape[0]
                pred,_,_,_,_,_,_ = model(image, depth)
                for j in range(bz):
                    res2= pred[j]
                    res2 = torch.sigmoid(res2).data.cpu().numpy().squeeze()
                    name2= name[j]
                    res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
                    res2= res2*255
                    cv2.imwrite(save_path+ name2, res2)

            print('Test Done!', checkp)
