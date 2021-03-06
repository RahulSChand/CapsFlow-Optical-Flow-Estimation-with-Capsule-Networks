import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from data_shape_double import ShapeData
import torch.optim as optim
from torch import autograd
from shape_model_double import ShapeNet
import shutil
from tensorboardX import SummaryWriter
import math
import time
from torch.autograd import Variable
from utils import show_flow
#
def save_checkpoint(state, is_best, filename='shape.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'workshop_exp_double_final3.pth.tar')

def load_my_state_dict(model,state_dict):
    own_state = model.state_dict()
    for name,param in state_dict.items():
        if name not in own_state:
            continue
        else:
            try:
                #print(name)
                param=param.data
                own_state[name].copy_(param)
            except:
                continue

BATCH_SIZE=64

device = torch.device("cuda:1")
model = ShapeNet(BATCH_SIZE,0.1,1,math.pi,"train").to(device)

####
PATH = "workshop_exp_double_final.pth.tar"
best_model = torch.load(PATH)
#load_my_state_dict(model,best_model['state_dict'])
model.load_state_dict(best_model['state_dict'])

train_set = ShapeData(None)
training_data_loader = DataLoader(dataset=train_set, num_workers=8, batch_size=BATCH_SIZE, shuffle=True,drop_last=True)

writer = SummaryWriter("workshop_double_flow_final3")

count=0

def epe_loss(pred,truth):
    diff = pred-truth
    return torch.mean(torch.norm(diff,p=2,dim=1))

def get_grad_norm(model):
    total_norm=0
    for name,p in model.named_parameters():

            ##print("")
            param_norm = p.grad.data.norm(2)


            ##print(name,param_norm)
            total_norm += param_norm.item() ** 2
    ##print("------")
    total_norm = total_norm ** (1. / 2)
    return total_norm

def margin_loss(activation,ground1,ground2,m):

    #ground --> (B,5)
    #activation --> (B,5)

    #activation --> (B,4)
    #ground --> (B,4)
    correct = ground1*activation
    correct = torch.sum(correct,dim=1,keepdim=True)
    #(B,1)

    activation = (1-ground2)*activation

    zero = torch.tensor(0,dtype=torch.float32).cuda("cuda:1")
    l = torch.mean(torch.max(zero,m-(correct-activation))**2)
    return l


optimizer = optim.Adam(model.parameters(), lr=0.0005)
batch_size=BATCH_SIZE

loss_flow_val_1 =0
loss_act_val_1 =0
loss_flow_val_2 =0
loss_act_val_2 =0
loss_flow_av =0
loss_act_av =0

def get_image_for_flow(input):

    out = torch.Tensor.cpu(input).detach().numpy()
    out = show_flow(out)
    out = np.transpose(out,[2,0,1])
    out = torch.tensor(out)

    return out

name_list = ['hex','square','triangle','circle','star']
loss_mse = nn.MSELoss()
for epoch in range(500):
    epoch_loss=0
    is_best=False
    for iteration, batch in enumerate(training_data_loader, 0):
            with autograd.detect_anomaly():

                start =time.time()
                img1,img2,flow1,flow2,class_tensor1,class_tensor2,img_recon_1,img_recon_2 = batch['img1'].cuda("cuda:1"),batch['img2'].cuda("cuda:1"),batch['flow1'].cuda("cuda:1"),batch['flow2'].cuda("cuda:1"),batch['class_tensor1'].cuda("cuda:1"),batch['class_tensor2'].cuda("cuda:1"),batch['comp1'].cuda("cuda:1"),batch['comp2'].cuda("cuda:1")
                
                #img1,img2,flow1,class_tensor1 = batch['img1'].cuda("cuda:1"),batch['img2'].cuda("cuda:1"),batch['flow'].cuda("cuda:1"),batch['class_tensor'].cuda("cuda:1")

                flow1 = flow1.type(torch.cuda.FloatTensor)
                #flow2 = flow2.type(torch.cuda.FloatTensor)
                
                optimizer.zero_grad()

                flow_pred1,flow_pred2,recon1,recon2,activation = model(img1,img2,class_tensor1,class_tensor2)

                #
                #print(flow_pred1)
                #flow_pred1_boolean = flow_pred1 and flow_pred1
                #print(flow_pred1_boolean)
                
                flow_pred1_copy = flow_pred1.clone()
                flow_pred2_copy = flow_pred2.clone()
                
                flow_pred2_copy[torch.abs(flow_pred2)<0.1] = 0.0
                
                #flow_pred2_copy[flow_pred2>-0.001] = 0.0
                flow_pred1_copy[torch.abs(flow_pred1)<0.1] = 0.0
                #flow_pred1_copy[flow_pred1>-0.001] = 0.0

                flow_pred1[torch.abs(flow_pred1)<0.1]  = 0.0
                flow_pred2[torch.abs(flow_pred2)<0.1] = 0.0

                #flow_pred1_copy --> (B,2,128,128) --> 1's where there is non-zero flow

                flow_pred1_boolean = (flow_pred1_copy[:,0,:,:]!=0) | (flow_pred1_copy[:,1,:,:]!=0) 
                flow_pred1_boolean = torch.unsqueeze(flow_pred1_boolean,1)



                flow_pred2_boolean = (flow_pred2_copy[:,0,:,:]!=0) | (flow_pred2_copy[:,1,:,:]!=0) 
                flow_pred2_boolean = torch.unsqueeze(flow_pred2_boolean,1)

                #flow_pred1_boolean = flow_pred1_copy!=0.0
                #flow_pred2_boolean = flow_pred2_copy!=0.0
                
                flow_boolean = flow_pred1_boolean*flow_pred2_boolean
                flow_boolean = 1 - flow_boolean
                flow_boolean = flow_boolean.type(torch.cuda.FloatTensor)
                #print(torch.sum(flow_boolean))
                flow_pred_combined = flow_pred1 + flow_pred2*flow_boolean
                #THis is correct now.
                ##What should be the precedence??????????

                loss_flow1 = epe_loss(flow_pred_combined,flow1)
                #loss_flow2 = epe_loss(flow_pred2,flow2)

                loss_act1 = margin_loss(activation,class_tensor1,class_tensor2,0.95)
                loss_act2 = margin_loss(activation,class_tensor2,class_tensor1,0.95)

                loss_recon1 = loss_mse(recon1,img_recon_1)
                loss_recon2 = loss_mse(recon2,img_recon_2)

                #loss = loss_flow1 + loss_flow2 + 10*loss_act1 + 10*loss_act2 + 200*loss_recon1 + 200*loss_recon2
                loss = 10*loss_act1 + 50000*loss_recon1 + loss_flow1*100
                
                loss_flow_val_1 += loss_flow1.item()
                loss_act_val_1 += loss_act1.item()

                #loss_flow_val_2 += loss_flow2.item()
                loss_act_val_2 += loss_act2.item()

                loss_recon_val_1 = loss_recon1.item()
                loss_recon_val_2 = loss_recon2.item()
                
                

                loss_flow_av = loss_flow_val_1
                loss_act_av = (loss_act_val_1 + loss_act_val_2)/2
                loss_recon_av = (loss_recon_val_1+loss_recon_val_2)/2

                loss.backward()



                if count%20==0:

                    print(count)
                    writer.add_scalar('workshop_double_flow_final3/loss_flow_1', float(loss_flow_val_1)/20, count)
                    loss_flow_val_1 =0

                    writer.add_scalar('workshop_double_flow_final3/loss_act_1', float(loss_act_val_1)/20, count)
                    loss_act_val_1 =0

                    
                    #writer.add_scalar('workshop_double_flow_final/loss_flow_2', float(loss_flow_val_2)/20, count)
                    #loss_flow_val_2 =0

                    writer.add_scalar('workshop_double_flow_final3/loss_act_2', float(loss_act_val_2)/20, count)
                    loss_act_val_2 =0

                    writer.add_scalar('workshop_double_flow_final3/loss_flow_av', float(loss_flow_av)/20, count)
                    loss_flow_av =0

                    writer.add_scalar('workshop_double_flow_final3/loss_act_av', float(loss_act_av)/20, count)
                    loss_act_av =0
                    
                    writer.add_scalar('workshop_double_flow_final3/loss_recon_val_1', float(loss_recon_val_1)/20, count)
                    loss_recon_val_1 =0

                    writer.add_scalar('workshop_double_flow_final3/loss_recon_val_2', float(loss_recon_val_2)/20, count)
                    loss_recon_val_2 =0
                    '''
                    writer.add_scalar('workshop_double_flow_final3/loss_flow_av', float(loss_recon_val_1)/20, count)
                    loss_recon_val_1 =0

                    writer.add_scalar('workshop_double_flow_final3/loss_flow_av', float(loss_recon_val_2)/20, count)
                    loss_recon_val_2 =0
                    '''


                    save_checkpoint({'epoch': epoch + 1,'state_dict': model.state_dict(),'optimizer' : optimizer.state_dict()},True)

                if count%100==0:
                    x = torch.Tensor.cpu(recon1[0]).detach().numpy()
                    y = torch.Tensor.cpu(recon2[0]).detach().numpy()

                    #print(np.sum((x==y).astype(np.float32)))

                    writer.add_image('img1',img1[0],count)
                    writer.add_image('img2',img2[0],count)
                    writer.add_image('recon_1',recon1[0],count)
                    writer.add_image('img_recon_1',img_recon_1[0],count)
                    writer.add_image('recon_2',recon2[0],count)
                    writer.add_image('img_recon_2',img_recon_2[0],count)
                    writer.add_image('flow_boolean',flow_boolean[0],count)
                    writer.add_image('flow_boolean1',flow_pred1_boolean[0],count)
                    writer.add_image('flow_boolean2',flow_pred2_boolean[0],count)

                    out1 = get_image_for_flow(flow_pred1[0])
                    out2 = get_image_for_flow(flow_pred2[0])
                    out3 = get_image_for_flow(flow_pred_combined[0])
                    gr1 = get_image_for_flow(flow1[0])
                    #gr2 = get_image_for_flow(flow2[0])

                    writer.add_image('flow1',out1,count)
                    writer.add_image('flow2',out2,count)
                    writer.add_image('flow_combined',out3,count)

                    writer.add_image('gr1',gr1,count)
                    #writer.add_image('gr2',gr2,count)


                optimizer.step()

                count=count+1

writer.close()
