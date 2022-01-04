import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms#, utils

import glob
from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import BuildingDataset

import time
from tqdm import tqdm
import acc

from model.CGSANet import CGSANet
# --------- 1. get image path and name ---------
image_dir = r"E:\Data\Dataset\INRIA_512\val\images"
label_dir = r"E:\Data\Dataset\INRIA_512\val\gt"
output_height = 512
output_width = output_height
model_dir = './Best_weights/INRIA/epoch_ 90_best_iou_loss_0.437474_iou_80.900604.pth'
in_channel = 3
threshold = 0.5

img_name_list = glob.glob(image_dir +"\\" +"*.tif")
lbl_name_list = []
for img_path in img_name_list:
    img_name = img_path.split("\\")[-1]

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    lbl_name_list.append(label_dir + "\\"+imidx + '.tif')    

# --------- 2. dataloader ---------
#1. dataload
test_dataset = BuildingDataset(img_name_list = img_name_list, lbl_name_list = lbl_name_list,transform=transforms.Compose([RescaleT(512),ToTensor()]))
test_dataloader = DataLoader(test_dataset, batch_size=1,shuffle=False,num_workers=0)

# --------- 3. model define ---------
print("...load Model...")
net = CGSANet(in_channel)

pre_model = torch.load(model_dir)
model2_dict = net.state_dict()
state_dict = {k:v for k,v in pre_model.items() if k in model2_dict.keys()}
model2_dict.update(state_dict)
net.load_state_dict(model2_dict)
# net.load_state_dict(torch.load(model_dir))
if torch.cuda.is_available():
    net.cuda()
net.eval()

hists = [[0, 0], [0, 0]]
time_start_all = time.time() 
# --------- 4. inference for each image ---------
for data_test in tqdm(test_dataloader):  
    inputs_test = data_test['image']
    inputs_test = inputs_test.type(torch.FloatTensor)

    labels_test = data_test['label']

    if torch.cuda.is_available():
        inputs_test = Variable(inputs_test.cuda())
    else:
        inputs_test = Variable(inputs_test)

    d1, d2, d3, d4, d6 = net(inputs_test)
    y_pb = d1[:,0,:,:]
    y_pb = torch.ge(torch.sigmoid(y_pb), threshold).float()
    pred = y_pb.cpu().detach().numpy()
    hist_t = acc.hist(labels_test.numpy(), pred)
    hists = hists + hist_t
    del d1, d2, d3, d4, d6, y_pb, pred,hist_t

recall, precision, iou, f1measure, accuray= acc.show_cd_rpqf1_pixel(hists)
print("finished predicting ", image_dir)