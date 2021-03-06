import torch
#import torch.utils.data.Dataset
from PIL import Image
import numpy as np
import cv2
import time
import os
from torch.utils.data import Dataset
import torchvision
#from transform import RandomHorizontalFlip,RandomColorWarp,RandomRotate,RandomVerticalFlip
#
import skimage
import random
#from utils import show_flow


def get_split():

    img_list = [str(i).zfill(7) for i in range(22232)]
    return img_list

def get_split2(fileName= "/sdb/flying_chairs.txt"):
    fileName = "/sdb/flying_chairs/FlyingChairs_train_val.txt"

    lines = [line.rstrip('\n') for line in open(fileName)]
    train_data=[]
    validation_data=[]
    test_data=[]
    count=1
    data={}

    #print(lines)
    count=2
    for i in lines:
        if count<1024:
            train_data.append(str(count).zfill(5))
        else:
            if count<1240:
                validation_data.append(str(count).zfill(5))
            else:
                break
        count=count+1
        
    data['train']=train_data
    data['validation']=validation_data
    data['test']=test_data

    print(len(train_data))
    print(len(validation_data))
    print(len(test_data))

    return data


class FlyingChairs(Dataset):

        def __init__(self,root_dir='data',list_IDs=[]):
       
            self.root_dir = root_dir
            self.list_IDs=list_IDs
            #self.softmax_list = np.array([-16+4*i for i in range(0,10)])
            #self.softmax_list = np.reshape(self.softmax_list,[1,1,10])
            
        def __len__(self):
            return len(self.list_IDs)
        
        def __getitem__(self, idx):

            folder = '/sdb/flying_chairs/train/'
            folder_seg = '/home/rahulc/videoCapsule/flying2_seg_new/'

            img1 = np.array(Image.open(folder+self.list_IDs[idx]+'-img_0.png').resize((256,192),Image.ANTIALIAS),dtype=np.float32)/255
            img2 = np.array(Image.open(folder+self.list_IDs[idx]+'-img_1.png').resize((256,192),Image.ANTIALIAS),dtype=np.float32)/255
            seg = np.array(Image.open(folder_seg+'seg_'+self.list_IDs[idx]+'.png').resize((256,192),Image.ANTIALIAS),dtype=np.float32)/255
            
            #(A,B,1)
            #(A,B)
            img1 = np.transpose(img1,[2,0,1])
            img2 = np.transpose(img2,[2,0,1])

            fileread = open(folder+self.list_IDs[idx]+'-flow_01.flo','rb')
            flownumber = np.fromfile(fileread,np.float32,count=1)[0]
            w=int(np.fromfile(fileread,np.int32,count=1))
            h=int(np.fromfile(fileread,np.int32,count=1))
            flow_target = np.resize(np.fromfile(fileread,np.float32,2*w*h),(h,w,2))
            #print(flow_target.shape,"before")
            #(384,512,2)
            
            
            
            #transforms = [RandomHorizontalFlip(),RandomVerticalFlip(),RandomRotate(angle=17),RandomColorWarp(0.5,0.5)]
            
            '''
            r = np.random.random()
            #print(r)
            if r<0.5:
                #print("some transform")
                if r<0.125:
                    imgConcat,flow_target = transforms[0](imgConcat,flow_target)
                if r>0.125 and r<0.25:
                    imgConcat,flow_target = transforms[1](imgConcat,flow_target)
                if r>0.25 and r<0.375:
                    imgConcat,flow_target = transforms[2](imgConcat,flow_target)
                else:
                    imgConcat,flow_target = transforms[3](imgConcat,flow_target)
            
            '''
            #flow_target --> (1,384,512)
            #multiply it with image1 --> (3,384,512)
            #(224,224,2)

            #print(flow_target.shape,"before I reshape")
            flow_target = cv2.resize(flow_target,(256,192))
            #flow_target_mask = cv2.resize(flow_target_mask,(256,192))
            #flow_target_mask[flow_target_mask>0] = 1.0
            '''
            flow_target_mask = np.sum(np.abs(flow_target),axis=2,keepdims=True)
            flow_target_mask[flow_target_mask<5]  = 0.0
            flow_target_mask[flow_target_mask>5] = 1.0
            '''
            
            
            

            #flow_target_mask = flow_target_mask*flow_target
            
            #print(flow_target_mask)

            #print(flow_target.shape,"before transpose")

            flow_target = np.transpose(flow_target,[2,0,1])
            #flow_target_mask = np.transpose(flow_target_mask,[2,0,1])
            

            name = np.array([int(self.list_IDs[idx])])
            
            '''
            print(img1.shape,"img1")
            print(flow_target.shape,"flow")
            print(flow_target_mask.shape,"mask")
            '''
            seg = np.sum(seg,axis=2)/3.0
            flow_target = flow_target*seg
            seg = np.expand_dims(seg,axis=0)

            #print(img1.shape,"img1")
            #print(img2.shape,"img2")
            #print(flow_target.shape,"flow_target")
            #print(seg.shape,"seg_target")
            

            sample = {'img1':img1,'img2':img2,'flow':flow_target,'mask':seg,'name':name}
            
            return sample     



'''
class FlyingChairs(Dataset):

        def __init__(self,root_dir='data',list_IDs=[]):
       
            self.root_dir = root_dir
            self.list_IDs=list_IDs

        def __len__(self):
            return len(self.list_IDs)
        
        def __getitem__(self, idx):

            folder = '/sdb/rahulc/data/FlyingChairs_release/'
            img1 = np.array(Image.open(folder+'data/' +'/'+self.list_IDs[idx]+'_img1.ppm'),dtype=np.float32)/255
            img2 = np.array(Image.open(folder+'data/' +'/'+self.list_IDs[idx]+'_img2.ppm'),dtype=np.float32)/255
            
            #img1 = np.transpose(img1,[1,2,0])
            #img2 = np.transpose(img2,[1,2,0])




            img1= np.expand_dims(img1,0)
            img2= np.expand_dims(img2,0)

            imgConcat = np.concatenate([img1,img2],0)

            fileread = open(folder+'data/' +'/'+self.list_IDs[idx]+'_flow.flo','rb')
            flownumber = np.fromfile(fileread,np.float32,count=1)[0]
            w=int(np.fromfile(fileread,np.int32,count=1))
            h=int(np.fromfile(fileread,np.int32,count=1))
            flow_target = np.resize(np.fromfile(fileread,np.float32,2*w*h),(h,w,2)) 
            #(384,512,2)
            
            
            
            transforms = [RandomHorizontalFlip(),RandomVerticalFlip(),RandomRotate(angle=17),RandomColorWarp(0.5,0.5)]
            

            r = np.random.random()
            #print(r)
            if r<0.5:
                #print("some transform")
                if r<0.125:
                    imgConcat,flow_target = transforms[0](imgConcat,flow_target)
                if r>0.125 and r<25:
                    imgConcat,flow_target = transforms[1](imgConcat,flow_target)
                if r>0.25 and r<0.375:
                    imgConcat,flow_target = transforms[2](imgConcat,flow_target)
                else:
                    imgConcat,flow_target = transforms[3](imgConcat,flow_target)
            
            
            
            imgConcat = np.transpose(imgConcat,[3,0,1,2])
            flow_target = np.transpose(flow_target,[2,0,1])
            #(2,384,512)

            #flow_target_mask = np.pad(flow_target,((0,0),(1,2),(2,3)),'edge')
            #print(np.amax(flow_target))
            #print(np.amin(flow_target))
            
            flow_target_mask = np.copy(flow_target)
            flow_target_mask = flow_target_mask[:,2:382,:]
            
            flow_target_mask = np.pad(flow_target_mask,((0,0),(0,0),(3,3)),'edge')

            flow_target_mask = skimage.measure.block_reduce(flow_target_mask,(1,38,37),np.mean)
            
            flow_target_mask = np.resize(flow_target_mask,(2,10*14))
            classes = np.arange(-5,5)
            flow_target_mask = flow_target_mask/8
            flow_target_mask = np.digitize(flow_target_mask,classes,right=True)
            flow_target_mask[flow_target_mask>=10]=9
            flow_target_mask[flow_target_mask<=0]=0
            
            #flow_target_mask = flow_target_mask.astype(np.uint8)


            #low_target_ground = np.zeros()
            #(0,9) values
            #flow_target_back = -1*flow_target
            #flow_target = flow_target.astype(int)
            #np.savetxt("flow_target",flow_target[0],fmt='%4.1f')
            #np.savetxt("flow_target_mask",flow_target_mask[0],fmt='%4.1f')
            
            #np.savetxt("flow_target_2",flow_target[1],fmt='%4.1f')
            #np.savetxt("flow_target_mask_2",flow_target_mask[1],fmt='%4.1f')
            #flow_target_concat = np.concatenate([flow_target,flow_target],0)
            #print(flow_target.shape)
            #img1 = show_flow(flow_target)
            #cv2.imwrite("flow_image.png",img1)
            #name = np.array([int(self.list_IDs[idx])])
            sample = {'image':imgConcat,'target':flow_target,'mask':flow_target_mask}
            #print(sample['image'].shape)
            #print(flow_target.shape)
            #time.sleep(100)

            return sample     


'''
