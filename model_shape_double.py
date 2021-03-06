import torch
import torch.nn as nn
import torch.nn.functional as F
#import correlation
#from correlation_new import correlation_custom
#from spatial_correlation_sampler import spatial_correlation_sample

class ConvLayer(nn.Module):
    """
    The C3D network as described in [1].
    """
    def __init__(self):

        super(ConvLayer, self).__init__()

        self.conv1_input_1 = nn.Conv2d(1,32,kernel_size=(3,3),padding=(1,1),stride=(2,2))
        self.batch_norm_1 = nn.BatchNorm2d(32)
        #self.batch_norm_1_ = nn.BatchNorm2d(32)
        #(256)
        #-----------------------------------------------------------

        #self.conv2_input_1 = nn.Conv2d(32,64,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        #self.batch_norm2_input_1 = nn.BatchNorm2d(64)

        self.conv2_input_2 = nn.Conv2d(32,64,kernel_size=(3,3),padding=(1,1),stride=(2,2))
        self.batch_norm2_input_2 = nn.BatchNorm2d(64)

        self.conv3_input_3 = nn.Conv2d(64,128,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.batch_norm3_input_3 = nn.BatchNorm2d(128)

        #self.conv3_input_3 = nn.Conv2d(64,128,kernel_size=(3,3),padding=(1,1),stride=(2,2))
        #self.batch_norm3_input_3 = nn.BatchNorm2d(128)

        self.conv3_input_2 = nn.Conv2d(128,256,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        self.batch_norm3_input_2 = nn.BatchNorm2d(256)

        #self.conv3_input_4 = nn.Conv2d(256,256,kernel_size=(3,3),padding=(1,1),stride=(1,1))
        #self.batch_norm3_input_4 = nn.BatchNorm2d(256)

        self.padding = torch.nn.ReplicationPad2d((1,2,1,2))
        #(48,64)
        #Convert this to 24,32 using primary capsule layer
        #


    def forward(self,input1,input2):

        input1 = self.padding(input1)
        input2 = self.padding(input2)



        self.out1 = F.leaky_relu(self.batch_norm_1(self.conv1_input_1(input1)),negative_slope=0.1,inplace=True)
        #self.out1 = F.leaky_relu(self.batch_norm2_input_1(self.conv2_input_1(self.out1)),negative_slope=0.1,inplace=True)
        self.out1 = F.leaky_relu(self.batch_norm2_input_2(self.conv2_input_2(self.out1)),negative_slope=0.1,inplace=True)
        #self.out1 = F.leaky_relu(self.batch_norm3_input_1(self.conv3_input_1(self.out1)),negative_slope=0.1,inplace=True)
        self.out1 = F.leaky_relu(self.batch_norm3_input_3(self.conv3_input_3(self.out1)),negative_slope=0.1,inplace=True)
        self.out1 = F.leaky_relu(self.batch_norm3_input_2(self.conv3_input_2(self.out1)),negative_slope=0.1,inplace=True)
        #self.out1 = F.leaky_relu(self.batch_norm3_input_4(self.conv3_input_4(self.out1)),negative_slope=0.1,inplace=True)


        self.out2 = F.leaky_relu(self.batch_norm_1(self.conv1_input_1(input2)),negative_slope=0.1,inplace=True)
        #self.out2 = F.leaky_relu(self.batch_norm2_input_1(self.conv2_input_1(self.out2)),negative_slope=0.1,inplace=True)
        self.out2 = F.leaky_relu(self.batch_norm2_input_2(self.conv2_input_2(self.out2)),negative_slope=0.1,inplace=True)
        #self.out2 = F.leaky_relu(self.batch_norm3_input_1(self.conv3_input_1(self.out2)),negative_slope=0.1,inplace=True)
        self.out2 = F.leaky_relu(self.batch_norm3_input_3(self.conv3_input_3(self.out2)),negative_slope=0.1,inplace=True)
        self.out2 = F.leaky_relu(self.batch_norm3_input_2(self.conv3_input_2(self.out2)),negative_slope=0.1,inplace=True)
        #self.out2 = F.leaky_relu(self.batch_norm3_input_4(self.conv3_input_4(self.out2)),negative_slope=0.1,inplace=True)

        return self.out1,self.out2
        #(B,CH,W,H)


class PrimaryCaps(nn.Module):

    #Convert #(B,CH,W,H) ---> (B,W,H,Capsule,4,4) ----> via ---> (B,W,H,Capsule*4*4)
    #Input is 48 --> Take down to 24 or take input as 24?
    #Start with 48 then go to 24

    def __init__(self,capsule_next,kernel_size,stride,padding):
        super(PrimaryCaps, self).__init__()

        self.capsule_next=capsule_next

        self.kernel_size = kernel_size
        self.stride=stride
        self.padding = padding
        self.conv_capsule = nn.Conv2d(256,16*self.capsule_next,kernel_size=kernel_size,stride=stride,padding=padding)
        #(b,16,2,21,25)

        self.conv_activation = nn.Conv2d(256,1*self.capsule_next,kernel_size=kernel_size,stride=stride,padding=padding)
        #(N,C,D,H,W)

    def forward(self,x,input_shape):



        self.output_shape = (int((input_shape[0]-self.kernel_size[0]+2*self.padding[0])/self.stride[0])+1,int((input_shape[1]-self.kernel_size[1]+2*self.padding[1])/self.stride[1])+1)

        self.capsule = F.relu(self.conv_capsule(x))
        self.capsule = torch.reshape(self.capsule,[-1,4,4,self.capsule_next,self.output_shape[0],self.output_shape[1]])
        #self.capsule = self.capsule.permute(0,4,5,6,3,1,2)
        self.capsule = self.capsule.permute(0,4,5,3,1,2).contiguous()


        self.activation = torch.sigmoid(self.conv_activation(x))
        #(b,32,2,24,32)
        self.activation = self.activation.permute(0,2,3,1).contiguous()


        ######print(self.capsule.shape)
        ######print(self.activation.shape)

        return (self.capsule,self.activation)
        #(B,24,32,32,4,4),  #(b,24,32,32)


class ConvCaps(nn.Module):

    def __init__(self,batch_size,kernel_size,stride,padding,capsule_previous,capsule_next,config):

        super(ConvCaps, self).__init__()

        self.capsule_previous=capsule_previous
        self.capsule_next=capsule_next
        self.batch_size  = batch_size


        self.weightMatrix = nn.Parameter(torch.randn((1,capsule_next,capsule_previous,4,4),device="cuda:1"))
        torch.nn.init.xavier_uniform(self.weightMatrix)

        self.Bv = nn.Parameter(torch.cuda.FloatTensor(1,capsule_next,1,16).fill_(0.1).cuda("cuda:1"))

        self.Ba = nn.Parameter(torch.cuda.FloatTensor(1,capsule_next,1).fill_(0.1).cuda("cuda:1"))

        self.count_number = 0
        self.config=config
        self.boost_update = 0.10
        self.boost_weights = torch.cuda.FloatTensor(self.capsule_next).fill_(1).cuda("cuda:1")

        self.average_rank = torch.cuda.FloatTensor(self.capsule_next).fill_(0).cuda("cuda:1")

        self.target_freq_min = 0.03
        self.target_freq_max = 0.12
        self.batch_count = 0
        self.inverse = 12
        self.unsupervised_masking=False
        self.kernel_size = kernel_size
        self.stride=stride
        self.kernel_size=kernel_size
        self.padding = padding
        self.alpha = 0.995

    def forward(self,inputPose,inputActivation,height,width,epsilon,inverse_temperature,pi):

        #InputPoseShape --> (b*w*h,32,4,4)
        self.batch_count+=1
        self.height = height
        self.width = width


        self.stack_number = (int((self.height-self.kernel_size[0]+2*self.padding[0])/self.stride[0])+1)*(int((self.width-self.kernel_size[1]+2*self.padding[1])/self.stride[1])+1)




        self.epsilon = epsilon
        self.inverse_temperature = inverse_temperature
        self.pi = pi
        self.calculate_vote(inputPose,inputActivation)
        self.EM_routing(epsilon,inverse_temperature,pi)

        self.count_number +=1
        if (self.unsupervised_masking):
            self.unsupervised_routing()

        self.mean = self.mean.view(-1,self.capsule_next,16).contiguous()
        self.output_activation = self.output_activation.view(-1,self.capsule_next).contiguous()

        #Mean currently is --->
        #Will activation not change? ---> Depends you can do it---> okay
        #(b*Total,32,16)
        #(B*Total,32)
        return  self.mean,self.output_activation


    #No need for stack --> But may be slower than stack
    def calculate_vote(self,inputPose,inputActivation):
        #Input Pose: (B,W,H,Caps,4,4)


        self.poseStack = torch.stack([inputPose[:,i*self.stride[0]:i*self.stride[0]+self.kernel_size[0],j*self.stride[1]:j*self.stride[1]+self.kernel_size[1],:,:,:] for i in range(0,int((self.height+2*self.padding[0]-self.kernel_size[0])/self.stride[0])+1) for j in range(0,int((self.width+2*self.padding[1]-self.kernel_size[1])/self.stride[1])+1)],dim=1)

        #(B,W,H,CH,4,4)
        #(B,Total,Kernel[0],Kernel[1],32,4,4)
        self.poseStack = self.poseStack.view(self.batch_size*self.stack_number,-1,self.capsule_previous,4,4)

        self.poseStack = torch.mean(self.poseStack,dim=1,keepdim=True)
        #(B*Total,1,32,4,4)

        #---------------------For activation
        self.activationStack = torch.stack([inputActivation[:,i:i+self.kernel_size[0],j:j+self.kernel_size[1],:] for i in range(0,self.height-self.kernel_size[0]+1,self.stride[0])   for j in range(0,self.width-self.kernel_size[1]+1,self.stride[1])],1)
        #(b,198,1,5,5,32)

        self.activationStack = self.activationStack.view(self.batch_size*self.stack_number,-1,self.capsule_previous)
        #(b*198,1*5*5,32)

        #Average it
        self.activationStack = torch.mean(self.activationStack,dim=1,keepdim=True)
        #(b,198,1,32)

        self.vote = torch.matmul(self.weightMatrix,self.poseStack)

        self.vote = self.vote.view(-1,self.capsule_next,self.capsule_previous,16)
        #(Batch*Stack,32,32,16)



    def EM_routing(self,epsilon,inverse_temperature,pi):

        self.routing = torch.cuda.FloatTensor(self.batch_size*self.stack_number,self.capsule_next,self.capsule_previous).fill_(float(1)/float(self.capsule_next))

        self.routing = self.routing.cuda("cuda:1")

        self.output_activation = self.activationStack.clone()

        for i in range(3):
            self.M_step(epsilon,inverse_temperature)
            self.E_step(pi)
            inverse_temperature = inverse_temperature + 1

        #(b*198,10,16)
        self.mean = torch.squeeze(self.mean)

        #(b*198,10) -- Dont squeeze Now---
        self.output_activation = torch.squeeze(self.output_activation)


    def M_step(self,epsilon,inverse_temperature):
        #Vote --> (b*9*11,10,32,16)
        #Rotuing --> (b*9*11,10,32)
        #Acitvation --> (b*9*11,1,32)

        self.routing = torch.mul(self.routing,self.activationStack)
        #(b*9*11,10,32)
        ###print(self.output_activation)
        self.routing_expand = torch.unsqueeze(self.routing,3)
        #(b*9*11,10,32,1)

        self.routing_sum = torch.sum(self.routing_expand,dim=2,keepdim=True)
        #(b*9*11,10,1,1)


        self.mean = torch.sum(torch.mul(self.routing_expand,self.vote),dim=2,keepdim=True)
        #(b*9*11,10,1,16)

        self.mean = torch.div(self.mean,self.routing_sum+0.00001)
        #(b*9*11,10,1,16)


        #-------------Update variance for each dimension (16 dimensions)

        self.vote_sub_mean = torch.sub(self.vote,self.mean)**2


        self.vote_sub_mean2 = torch.sum(torch.mul(self.vote_sub_mean,self.routing_expand),dim=2,keepdim=True)
        #(b*9*11,10,1,16)
        self.variance = torch.div(self.vote_sub_mean2,self.routing_sum+0.001)
        #(b*99,10,1,16)


        #-----------Update cost-----------------

        self.cost = torch.mul(self.Bv + torch.log(torch.sqrt(self.variance+0.000001)+0.00001),self.routing_sum)
        #(b*198,10,1,16)
        #Total 198 spatial-temporal positions and 32 capsules for each of them and (16) pose for each.

        self.sum_cost = torch.sum(self.cost,dim=3)
         #a=np.random.random()

        self.sum_cost_mean = torch.mean(self.sum_cost,dim=1,keepdim=True)
        self.sum_cost_std_dev = torch.sqrt(torch.sum((self.sum_cost-self.sum_cost_mean)**2,dim=1,keepdim=True)/self.capsule_next+ 0.00001)

        self.output_activation = torch.sigmoid(inverse_temperature*(self.Ba+(self.sum_cost_mean-self.sum_cost)/(self.sum_cost_std_dev+0.00001)))



    def E_step(self,pi):

        #------Step 1---------------
        self.probability = torch.div(self.vote_sub_mean,2*self.variance+0.00001)
        #(b*198,32,32,16)

        self.probability = torch.exp(-1*torch.sum(self.probability,dim=3))
        #(b*198,32,32)

        self.denominator = torch.sqrt(2*pi*torch.sum(self.variance,dim=3))
        #(b*198,32,1)

        self.probability = torch.div(self.probability,self.denominator+0.0001)
        #(b*198,32,32)

        #--------Step 2-------------------
        #self.routing = torch.mul(self.output_activation,self.probability)
        #(b,198,32,32)

        self.activation_probability = torch.mul(self.output_activation,self.probability)
        #(b*198,32,32)

        #---Changing this implementation to only sum across first 32)
        #self.activation_probability_sum = torch.sum(torch.sum(self.routing,dim=2,keepdim=True),dim=1,keepdim=True)
        self.activation_probability_sum = torch.sum(self.activation_probability,dim=1,keepdim=True)
        #(b*198,1,32)

        self.routing = torch.div(self.activation_probability,self.activation_probability_sum+0.00001)

        #(b*198,10,32)


    def calculate_mask(self,capsule_number):


        #Averaging with respect to D*W*H

        self.routing = self.routing.view(self.batch_size,-1,self.capsule_next,self.capsule_previous)
        #(B,Total,32,32)


        self.routing_capsule_sum = torch.sum(self.routing,dim=[1,3])
        #(b,32)

        self.routing_capsule_sum = self.routing_capsule_sum * self.boost_weights

        _,order = torch.topk(self.routing_capsule_sum,k=capsule_number,dim=1)
        _,self.rank = torch.topk(-order,k=capsule_number,dim=1)
        self.rank = self.rank.type(torch.cuda.FloatTensor)
        self.rank_normalized = torch.div(self.rank,capsule_number-1)
        self.rank_normalized = self.rank_normalized.type(torch.cuda.FloatTensor)
        self.masking = torch.exp(self.inverse*self.rank_normalized*-1)
        self.masking[self.masking<0.01]=0

        #(b,32) , mean is (b*Total,32,16) so mean = (b,total,32,16) (b,32) --> (b,1,32,1)
        #(b,32) , activation now is (b*198,32) so acitvation = (b,198,32) and (b,32) --> (b,1,32)


        #(b,1,1,1,32,1,1)

        self.mean = self.mean.view(self.batch_size,-1,self.capsule_next,16)
        self.output_activation = self.output_activation.view(self.batch_size,-1,self.capsule_next)

        self.masking_expand_dims = torch.unsqueeze(self.masking,dim=1)

        self.output_activation = torch.mul(self.masking_expand_dims,self.output_activation)

        self.masking_expand_dims = torch.unsqueeze(self.masking_expand_dims,dim=3)

        self.mean = torch.mul(self.masking_expand_dims,self.mean)


    def post_batch_update(self):

        self.rank = torch.sum(torch.eq(self.rank,0),dim=0)/(self.capsule_next*self.batch_size)

        #(32)

        self.rank = self.rank.type(torch.cuda.FloatTensor)

        ###print(self.average_rank.type)
        #rint(self.alpha.type)
        self.average_rank = self.alpha * self.average_rank + (1-self.alpha)*self.rank



        if (self.batch_count%50==0):
            for i in range(self.capsule_next):
                if self.average_rank.data[i]<self.target_freq_min:
                    self.boost_weights[i] = self.boost_weights[i]+self.boost_update
                if self.average_rank.data[i]>self.target_freq_max:
                    self.boost_weights[i] = max(1,self.boost_weights[i]-self.boost_update)

    def unsupervised_routing(self):

        self.calculate_mask(self.capsule_next)
        self.post_batch_update()



class MaskCapsule(nn.Module):

    def __init__(self,batch_size,capsule_previous,capsule_next,config):

        super(MaskCapsule, self).__init__()



        self.capsule_previous=capsule_previous
        self.capsule_next=capsule_next
        self.batch_size  = batch_size


        self.weightMatrix = nn.Parameter(torch.randn((1,capsule_next,capsule_previous,4,4),device="cuda:1"))
        torch.nn.init.xavier_uniform(self.weightMatrix)

        self.Bv = nn.Parameter(torch.cuda.FloatTensor(1,capsule_next,1,16).fill_(0.1).cuda("cuda:1"))


        self.Ba = nn.Parameter(torch.cuda.FloatTensor(1,capsule_next,1).fill_(0.1).cuda("cuda:1"))

        self.count_number = 0
        self.config=config

    def forward(self,inputPose,inputActivation,height,width,ground,ground2,epsilon,inverse_temperature,pi):

        #InputPoseShape --> (b*w*h,32,4,4)
        self.height = height
        self.width = width

        self.ground = ground
        self.ground2 = ground2


        self.stack_number = self.height*self.width
        self.epsilon = epsilon
        self.inverse_temperature = inverse_temperature
        self.pi = pi
        self.calculate_vote(inputPose,inputActivation)
        self.EM_routing(epsilon,inverse_temperature,pi)
        self.masking()
        self.count_number +=1
        #(b*9*11,10)    (b,9*11,16)
        return  self.mask_output,self.output_activation,self.mask_output2,self.mean


    #No need for stack --> But may be slower than stack
    def calculate_vote(self,inputPose,inputActivation):
        #Input Pose : (b*9*11,32,4,4)
        #Weight : (10,32,4,4)

        #(b*w*h,p,n,m) * (q,p,n,m) --> (b,w,h,q,n,m)


        #self.vote = torch.einsum('dpnm,qpnm->dqpnm',inputPose,self.weightMatrix)


        inputPose = torch.unsqueeze(inputPose,1)


        self.vote = torch.matmul(self.weightMatrix,inputPose)

        #(b,1,32,4,4)
        #(1,10,32,4,4)

        self.vote = self.vote.view(-1,self.capsule_next,self.capsule_previous,16)
        self.inputActivation = inputActivation
    def masking(self):

        #Output currently --> (Batch,9,11,10,16) --> (Batch,1,9*11,10,16)
        #Across each 9*11 dimension one capsule will win
        #Ground_Shape -> (B*10*14,10,1)

        if self.config == "train":
            #self.mean ---> (B*W*H,4,16)
            #ground ---> (B,4)

            #self.mean ---> (B,-1,4,16)
            '''
            self.output_activation = self.output_activation.view(self.batch_size,2,2,5)
            self.output_activation = torch.mean(self.output_activation,dim=(1,2))

            self.index = torch.argmax(self.output_activation,dim=1)

            self.index = self.index.contiguous()

            #print(self.index)

            self.ground = torch.zeros([self.batch_size,5]).cuda("cuda:1")

            ###print(self.ground)

            self.ground[torch.arange(self.batch_size),self.index] = 1
            '''
            #print(self.ground)
            self.mean = self.mean.view(self.batch_size,-1,5,16)
            #(B,2*2,5,16)
            self.ground = self.ground.view(self.batch_size,1,5,1)
            self.ground2 = self.ground2.view(self.batch_size,1,5,1)
            #(B,1,5,1) --> (B,5) --> one hot vector

            #print(self.mean.shape,"THIS IS SHAPE OF MEAN")

            self.mask_output = torch.mul(self.mean,self.ground)
            self.mask_output2 = torch.mul(self.mean,self.ground2)

            self.mask_output = torch.sum(self.mask_output,dim=2)
            self.mask_output2 = torch.sum(self.mask_output2,dim=2)

            self.mask_output = self.mask_output.view(-1,16)
            self.mask_output2 = self.mask_output2.view(-1,16)


        if self.config=="inference":
            self.ground = torch.unsqueeze(self.ground,2)

            self.mask_output = torch.mul(self.mean,self.ground)

            self.mask_output = torch.sum(self.mask_output,dim=1)

        if self.config=="test":
            #(B*10*14,10)
            self.index = torch.argmax(self.output_activation,dim=1)
            self.index = self.index.contiguous()

            #(B*10*14)
            self.ground = torch.zeros([self.batch_size*12*16,10]).cuda("cuda:1")
            ###print(self.index.shape)
            self.ground[torch.arange(self.batch_size*12*16),self.index] = 1
            self.ground = torch.unsqueeze(self.ground,2)

            self.mask_output = torch.mul(self.mean,self.ground)
            #(B*10*14,10,16)

            self.mask_output = torch.sum(self.mask_output,dim=1)
            #(B*10*14)


    def EM_routing(self,epsilon,inverse_temperature,pi):

        self.routing = torch.cuda.FloatTensor(self.batch_size*self.stack_number,self.capsule_next,self.capsule_previous).fill_(float(1)/float(self.capsule_next)).cuda("cuda:1")

        self.output_activation = self.inputActivation.clone()
        self.output_activation = torch.unsqueeze(self.output_activation,dim=1)

        for i in range(3):
            self.M_step(epsilon,inverse_temperature)
            self.E_step(pi)
            inverse_temperature = inverse_temperature + 1

        #(b*198,10,16)
        self.mean = torch.squeeze(self.mean)

        #(b*198,10)
        self.output_activation = torch.squeeze(self.output_activation)



    def M_step(self,epsilon,inverse_temperature):
        #Vote --> (b*9*11,10,32,16)
        #Rotuing --> (b*9*11,10,32)
        #Acitvation --> (b*9*11,1,32)

        self.routing = torch.mul(self.routing, torch.unsqueeze(self.inputActivation,dim=1))
        #(b*9*11,10,32)

        self.routing_expand = torch.unsqueeze(self.routing,3)
        #(b*9*11,10,32,1)

        self.routing_sum = torch.sum(self.routing_expand,dim=2,keepdim=True)
        #(b*9*11,10,1,1)


        ###print(self.vote.shape)
        self.mean = torch.sum(torch.mul(self.routing_expand,self.vote),dim=2,keepdim=True)
        #(b*9*11,10,1,16)

        self.mean = torch.div(self.mean,self.routing_sum+0.00001)
        #(b*9*11,10,1,16)


        #-------------Update variance for each dimension (16 dimensions)
        ###print(self.vote)
        self.vote_sub_mean = torch.sub(self.vote,self.mean)**2

        ###print("--")
        self.vote_sub_mean2 = torch.sum(torch.mul(self.vote_sub_mean,self.routing_expand),dim=2,keepdim=True)
        #(b*9*11,10,1,16)
        self.variance = torch.div(self.vote_sub_mean2,self.routing_sum+0.001)
        #(b*99,10,1,16)


        #-----------Update cost-----------------

        self.cost = torch.mul(self.Bv + torch.log(torch.sqrt(self.variance+0.000001)+0.00001),self.routing_sum)
        #(b*198,10,1,16)
        #Total 198 spatial-temporal positions and 32 capsules for each of them and (16) pose for each.

        self.sum_cost = torch.sum(self.cost,dim=3)
         #a=np.random.random()

        self.sum_cost_mean = torch.mean(self.sum_cost,dim=1,keepdim=True)
        self.sum_cost_std_dev = torch.sqrt(torch.sum((self.sum_cost-self.sum_cost_mean)**2,dim=1,keepdim=True)/self.capsule_next + 0.0001)

        self.output_activation = torch.sigmoid(inverse_temperature*(self.Ba+(self.sum_cost_mean-self.sum_cost)/(self.sum_cost_std_dev+0.00001)))





    def E_step(self,pi):

        #------Step 1---------------
        self.probability = torch.div(self.vote_sub_mean,2*self.variance+0.00001)
        #(b*198,32,32,16)

        self.probability = torch.exp(-1*torch.sum(self.probability,dim=3))
        #(b*198,32,32)

        self.denominator = torch.sqrt(2*pi*torch.sum(self.variance,dim=3)+0.00001)
        #(b*198,32,1)

        self.probability = torch.div(self.probability,self.denominator+0.0001)
        #(b*198,32,32)

        #--------Step 2-------------------
        #self.routing = torch.mul(self.output_activation,self.probability)
        #(b,198,32,32)

        self.activation_probability = torch.mul(self.output_activation,self.probability)
        #(b*198,32,32)

        #---Changing this implementation to only sum across first 32)
        #self.activation_probability_sum = torch.sum(torch.sum(self.routing,dim=2,keepdim=True),dim=1,keepdim=True)
        self.activation_probability_sum = torch.sum(self.activation_probability,dim=1,keepdim=True)
        #(b*198,1,32)

        self.routing = torch.div(self.activation_probability,self.activation_probability_sum+0.00001)

        #(b*198,10,32)



#Input to this will be (9,11,16) and (9,11,32*16) and (H,W,32*16)

#Aim to remove extra conv layers from here if possible
'''
class transposeConv(nn.Module):

    def __init__(self):
        super(transposeConv,self).__init__()

        #17 is input, pad with 0,1

        self.padding = torch.nn.ReplicationPad2d((1,0,1,0))

        self.conv1 = nn.Conv2d(16,16*4,kernel_size=(1,1),stride=(1,1),padding=(0,0))

        self.transposeConv1 = nn.ConvTranspose2d(16*4,16*4,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #4

        self.transposeConv2 = nn.ConvTranspose2d(16*4,16*2,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #8

        self.conv_primary = nn.Conv2d(256,16*2,kernel_size=(3,3),stride=(1,1),padding=(0,0))

        self.transposeConv3 = nn.ConvTranspose2d(16*2,16*2,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #16

        self.transposeConv4 = nn.ConvTranspose2d(16*4,16*2,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #32

        self.transposeConv5 = nn.ConvTranspose2d(16*2,8,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #64

        self.transposeConv6 = nn.ConvTranspose2d(8,2,kernel_size=(4,4),stride=(2,2),padding=(1,1))


    def forward(self,primaryCaps,style):

        out = F.leaky_relu(self.conv1(style))


        out = F.leaky_relu(self.transposeConv1(out))
        out = F.leaky_relu(self.transposeConv2(out))

        out = F.leaky_relu(self.transposeConv3(out))

        prim = F.leaky_relu(self.conv_primary(self.padding(primaryCaps)))

        out = torch.cat([out,prim],dim=1)

        out = F.leaky_relu(self.transposeConv4(out))

        out = F.leaky_relu(self.transposeConv5(out))

        out = self.transposeConv6(out)

        return out
'''
'''
class transposeConv(nn.Module):

    def __init__(self):
        super(transposeConv,self).__init__()

        self.conv1 = nn.Conv2d(16,16*2,kernel_size=(1,1),stride=(1,1),padding=(0,0))
        self.conv2 = nn.Conv2d(16*2,16*2,kernel_size=(1,1),stride=(1,1),padding=(0,0))



        self.transposeConv1 = nn.ConvTranspose2d(16*4,16*4,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #4

        self.transposeConv2 = nn.ConvTranspose2d(16*4,16*4,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #8

        self.transposeConv3 = nn.ConvTranspose2d(16*4,16*2,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #16

        self.transposeConv4 = nn.ConvTranspose2d(16*2,16,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #32

        self.conv_flow1 = nn.Conv2d(16,2,kernel_size=(1,1),stride=(1,1),padding=(0,0))

        self.conv_s = nn.Conv2d(256,16*2,kernel_size=(1,1),stride=(1,1),padding=(0,0))

        #self.transposeConv5 = nn.ConvTranspose2d(16*2,8,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #64

        #self.transposeConv6 = nn.ConvTranspose2d(8,2,kernel_size=(4,4),stride=(2,2),padding=(1,1))


    def forward(self,style_transform,style):

        #style -> 16
        #style-transsform -> 16/32

        out = F.leaky_relu(self.conv1(style_transform))
        out = F.leaky_relu(self.conv2(out))

        style = F.leaky_relu(self.conv_s(style))

        out = torch.cat([out,style],dim=1)

        out = F.leaky_relu(self.transposeConv1(out))
        out = F.leaky_relu(self.transposeConv2(out))
        out = F.leaky_relu(self.transposeConv3(out))
        out = F.leaky_relu(self.transposeConv4(out))
        out = self.conv_flow1(out)

        return out
'''
'''
class transposeConv(nn.Module):

    def __init__(self):
        super(transposeConv,self).__init__()

        self.conv1 = nn.Conv2d(16,16*2,kernel_size=(1,1),stride=(1,1),padding=(0,0))
        self.conv2 = nn.Conv2d(16*2,16*2,kernel_size=(1,1),stride=(1,1),padding=(0,0))

        self.padding = nn.ReflectionPad2d((1,0,1,0))

        self.transposeConv1 = nn.ConvTranspose2d(16*2,16*4,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #4

        self.transposeConv2 = nn.ConvTranspose2d(16*4,16*4,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #8

        self.transposeConv3 = nn.ConvTranspose2d(16*4,16*2,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #16

        self.transposeConv4 = nn.ConvTranspose2d(16*4,16,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #32

        self.conv_flow1 = nn.Conv2d(16,2,kernel_size=(1,1),stride=(1,1),padding=(0,0))

        self.conv_s = nn.Conv2d(256,16*2,kernel_size=(3,3),stride=(1,1),padding=(0,0))


    def forward(self,style_transform,style):

        #style -> 16
        #style-transsform -> 16/32

        out = F.leaky_relu(self.conv1(style_transform))
        out = F.leaky_relu(self.conv2(out))

        out = F.leaky_relu(self.transposeConv1(out))
        out = F.leaky_relu(self.transposeConv2(out))
        out = F.leaky_relu(self.transposeConv3(out))

        style = F.leaky_relu(self.conv_s(self.padding(style)))
        out = torch.cat([out,style],dim=1)

        out = F.leaky_relu(self.transposeConv4(out))
        out = self.conv_flow1(out)

        return out
'''

class transposeConvRecon(nn.Module):

    def __init__(self):
        super(transposeConvRecon,self).__init__()

        #self.conv1 = nn.Conv2d(16,16*4,kernel_size=(1,1),stride=(1,1),padding=(0,0))

        self.transposeConv2 = nn.ConvTranspose2d(16,16*2,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #16

        self.transposeConv3 = nn.ConvTranspose2d(16*2,16*4,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #32

        self.transposeConv4 = nn.ConvTranspose2d(16*4,16,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #64

        self.transposeConv5 = nn.ConvTranspose2d(16,8,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #128
        
        self.conv_flow1 = nn.Conv2d(8,1,kernel_size=(1,1),stride=(1,1),padding=(0,0))

    def forward(self,style):
        
        out = F.relu(self.transposeConv2(style))        
        out_recon = F.relu(self.transposeConv3(out))
        out = F.relu(self.transposeConv4(out_recon))
        out = F.relu(self.transposeConv5(out))
        out = F.sigmoid(self.conv_flow1(out))

        return out,out_recon


class transposeConvFlow(nn.Module):

    def __init__(self):
        super(transposeConvFlow,self).__init__()

        self.padding = nn.ReflectionPad2d((1,0,1,0))

        self.transposeConv1 = nn.ConvTranspose2d(16,16*4,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #16
        
        self.transposeConv2 = nn.ConvTranspose2d(16*4,16*4,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #32
        
        self.transposeConv3 = nn.ConvTranspose2d(16*8,16*2,kernel_size=(4,4),stride=(2,2),padding=(1,1))
        #64
        
        self.conv_flow1 = nn.Conv2d(16*2,16,kernel_size=(1,1),stride=(1,1),padding=(0,0))
        self.conv_flow2 = nn.Conv2d(16,2,kernel_size=(1,1),stride=(1,1),padding=(0,0))
        
    
    def forward(self,motion,structure):
        
        #structure -> 16
        #structure-transsform -> 16/32
        
        out = F.leaky_relu(self.transposeConv1(motion))
        out = F.leaky_relu(self.transposeConv2(out))
        #32
        out = torch.cat([out,structure],dim=1)
        out = F.leaky_relu(self.transposeConv3(out))
        out = F.leaky_relu(self.conv_flow1(out))
        out = self.conv_flow2(out)
        #64
        
        return out
