import torch
torch.cuda.set_device(1)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from VideoLayerClassDGX import C3D,PrimaryCaps,ConvCaps,transposeConv,MaskCapsule
from model_shape_double import ConvCaps,ConvLayer,MaskCapsule,PrimaryCaps,transposeConvRecon,transposeConvFlow

class ShapeNet(nn.Module):

    def __init__(self,batch_size,epsilon,inverse_temperature,pi,config):

        super(ShapeNet, self).__init__()

        #self.convulation_layer = C3D("3D-color")

        self.convulation_layer = ConvLayer()
        #16
        self.primary_caps = PrimaryCaps(16,(3,3),(2,2),(1,1)) #(2,24,32)
        #9
        self.conv_caps = ConvCaps(batch_size,(7,7),(2,2),(0,0),16,16,config) #Stride inherently (1,2,2) and padding (0,0,0) #Output --> (10,14)
        #2
        self.conv_caps_class_1 = MaskCapsule(batch_size,16,5,config)
        #(2,2)


        self.weight_1 = nn.Conv2d(16,16,kernel_size=(1,1),stride=(1,1),padding=(0,0))

        self.transpose_conv_flow = transposeConvFlow()
        self.transpose_conv_recon = transposeConvRecon()


        self.epsilon = epsilon
        self.inverse_temperature = inverse_temperature
        self.pi = pi
        self.batch_size=batch_size

        self.prelu1 = torch.nn.PReLU()
        #self.prelu2 = torch.nn.PReLU()
        
        self.fc1 = nn.Linear(6*6*16,256)
        self.fc2 = nn.Linear(256,8*8*16)


    def forward(self,input1,input2,ground,ground2):

        ##print(self.batch_size)

        ####print(input.shape)

        #Ground ---> (2,9*11)

        self.conv_output_image_1,self.conv_output_image_2 = self.convulation_layer(input1,input2)

        conv_output_shape = self.conv_output_image_1.shape
        conv_output_shape = (conv_output_shape[2],conv_output_shape[3])

        #----------------PrimaryCaps------------------

        self.primary_caps_pose_input_1,self.primary_caps_activation_input_1 = self.primary_caps(self.conv_output_image_1,conv_output_shape)
        self.primary_caps_pose_input_2,self.primary_caps_activation_input_2 = self.primary_caps(self.conv_output_image_2,conv_output_shape)

        primary_caps_output_shape = self.primary_caps_pose_input_1.shape

        height_primary_caps,width_primary_caps = primary_caps_output_shape[1],primary_caps_output_shape[2]

        #----------------ConvCaps------------------


        self.inverse_temperature=1
        self.conv_capsule_pose_image_1,self.conv_capsule_activation_image_1 = self.conv_caps(self.primary_caps_pose_input_1,self.primary_caps_activation_input_1,height_primary_caps,width_primary_caps,self.epsilon,self.inverse_temperature,self.pi)
        self.inverse_temperature=1
        self.conv_capsule_pose_image_2,self.conv_capsule_activation_image_2 = self.conv_caps(self.primary_caps_pose_input_2,self.primary_caps_activation_input_2,height_primary_caps,width_primary_caps,self.epsilon,self.inverse_temperature,self.pi)
        #(B*198,32,16)

        self.conv_capsule_pose_image_2 = self.conv_capsule_pose_image_2.view(-1,16,4,4)
        self.conv_capsule_pose_image_1 = self.conv_capsule_pose_image_1.view(-1,16,4,4)


        #self.padding = torch.nn.ReplicationPad2d((1,0,1,0))
        #print(self.conv_capsule_pose_image_1.shape,"THIS IS SHAPE BEFORE MASK CAPSULE")
        #print(ground==ground2)
        self.style1,self.velocity_activation1,self.style1_2,_ = self.conv_caps_class_1(self.conv_capsule_pose_image_1,self.conv_capsule_activation_image_1,6,6,ground,ground2,self.epsilon,self.inverse_temperature,self.pi)
        self.style2,self.velocity_activation2,self.style2_2,_  = self.conv_caps_class_1(self.conv_capsule_pose_image_2,self.conv_capsule_activation_image_2,6,6,ground,ground2,self.epsilon,self.inverse_temperature,self.pi)

        #print(self.style1==self.style1_2)

        image_1_recon_1,image_1_flow_1 = self.relu_function(self.style1)
        image_1_recon_2,image_1_flow_2 = self.relu_function(self.style1_2)

        _,image_2_flow_1 = self.relu_function(self.style2)
        _,image_2_flow_2 = self.relu_function(self.style2_2)
        
        transform_2 = torch.sub(image_2_flow_2,image_1_flow_2)
        transform_1 = torch.sub(image_2_flow_1,image_1_flow_1)
        
        #print(image_1_recon_1==image_1_recon_2)

        recon_shape_1, recon_shape_1_flow = self.transpose_conv_recon(image_1_recon_1)
        recon_shape_2, recon_shape_2_flow = self.transpose_conv_recon(image_1_recon_2)

        flow_1 = self.transpose_conv_flow(transform_1,recon_shape_1_flow)
        flow_2 = self.transpose_conv_flow(transform_2,recon_shape_2_flow)

        self.velocity_activation1 = self.velocity_activation1.view(self.batch_size,6,6,5)
        self.velocity_activation1 = torch.mean(self.velocity_activation1 ,dim=(1,2))

        flow_1 = 2*F.interpolate(flow_1,scale_factor=2)
        flow_2 = 2*F.interpolate(flow_2,scale_factor=2)
        
        return flow_1,flow_2,recon_shape_1,recon_shape_2,self.velocity_activation1

    def relu_function(self,input_tensor):

        input_tensor = input_tensor.view(self.batch_size,6,6,16)
        input_tensor = input_tensor.view(self.batch_size,-1).contiguous()
        input_tensor = F.relu(self.fc1(input_tensor))
        input_tensor = F.relu(self.fc2(input_tensor))
        #(8,8)
        input_tensor = input_tensor.view(self.batch_size,16,8,8)
        #This will go to reconstruction decoder
        input_transform = self.prelu1(self.weight_1(input_tensor))

        return input_tensor,input_transform
        

    def get_input_trans(self,style1,style2,batch_size):

        style1 = style1.view(batch_size,6,6,16)
        style1 = style1.permute(0,3,1,2)

        style1_transform = self.prelu1(self.weight_1(style1))
        style1_transform = self.prelu2(self.weight_2(style1_transform))

        style2 = style2.view(batch_size,6,6,16)
        style2 = style2.permute(0,3,1,2)

        style2_transform = self.prelu1(self.weight_1(style2))
        style2_transform = self.prelu2(self.weight_2(style2_transform))


        transform = torch.sub(style1_transform,style2_transform)

        return transform
