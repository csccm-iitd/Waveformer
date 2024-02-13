from IPython import get_ipython
get_ipython().magic('reset -sf')


# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
# from Training_wno_weights import *
from WNO_encode import *
from timeit import default_timer
from utilities3 import *
from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)
torch.manual_seed(0)
np.random.seed(0)


# device= torch.device('cpu')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:1')
# %%
""" Model configurations """

TRAIN_PATH = 'Allen_Cahn_pde_65_65_1000.mat'
ntrain = 300
ntest = 20

num_epochs = 1
num_epochs2 = 200
# batch_size = 80
learning_rate = 5e-4


step_size = 20
gamma = 0.75

level = 4# The automation of the mode size will made and shared upon acceptance and final submission
width = 40

batch_size = 5

r = 1
h = int(((129 - 1)/r) + 1)
s = h

T_in = 10
# T = 10
T_out = 35
step =1

""" Read data """
reader = MatReader(TRAIN_PATH)
x_train_d = reader.read_field('sol').permute(3,0,1,2)
x_train = x_train_d[:ntrain,::r,::r,:]

# x_train_enc0 = x_train[:,:,:,0:T_in]
# x_train_out0 = x_train[:,:,:,T_in:T_out]

x_train_enc1 = x_train[:,:,:,0:T_in]
x_train_enc2 = x_train[:,:,:,1:T_in+1]
x_train_out1 = x_train[:,:,:,T_in+1:T_out+1]

x_test = x_train_d[ntrain:ntrain+ntest,::r,::r,:]

# test_enc0 = x_test[:,:,:,0:T_in]
# test_out0  = x_test[:,:,:,T_in:T_out]

test_enc1 = x_test[:,:,:,0:T_in]
test_enc2 = x_test[:,:,:,1:T_in+1]
test_out1  = x_test[:,:,:,T_in+1:T_out+1+80]

# test_out2  = x_test[:,:,:,T_in+1:T_out+1]
# x_test_enc = time_windowing(x_test,T).permute(0,1,3,4,2).to(device)
# train_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_enc0, x_train_out0), batch_size=batch_size, shuffle=True)
train_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train_enc1, x_train_enc2, x_train_out1), batch_size=batch_size, shuffle=True)
# test_loader1 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_enc0, test_out0), batch_size=batch_size, shuffle=True)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_enc1, test_enc2, test_out1), batch_size=batch_size, shuffle=True)
test_loader_pred2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_enc1, test_enc2, test_out1), batch_size=1, shuffle=False)
#%%
# model = WNO2d(width, level, x_train_enc0[0:1,:,:,:].permute(0,3,1,2)).to(device)
# print(count_params(model))
    
# """ Training and testing """
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
# n0_ws,n0_bs,n1_ws,n1_bs,n2_ws,n2_bs = training_model(model, optimizer,scheduler,num_epochs,train_loader1,test_loader1,batch_size,ntrain,ntest,T_in,T_out,step)
#%%
""" The model definition """
model2 = WNO2dtransformer(width, level, x_train_enc1[0:1,:,:,0:T_in].permute(0,3,1,2)).to(device)
print(count_params(model2))

""" Training and testing """
optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

for ep in range(num_epochs2):
    model2.train()
    t1 = default_timer()
    train_l2 = 0
    train_dl = 0
    train_pl = 0
    for (dat1, dat2, dat_out) in train_loader2:
        data_loss = 0
        xx = dat1.to(device)
        yy = dat2.to(device) 
        zz = dat_out.to(device) 
        loss_physics = 0  
        for t in range(0, T_out-T_in, step):
            # x = xx[:,:,:,t:t+T_in]
            # y = yy[:,:,:,t:t+T_in]
            z = zz[:,:,:,t:t+step]
            im =  model2(xx,yy)  
            if t == 0:
               pred = im
            else:
               pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., 1:], yy[:,:,:,-1:]), dim=-1)
            yy = torch.cat((yy[..., 1:], im), dim=-1)
            
            data_loss += F.mse_loss(im.reshape(batch_size, -1), z.reshape(batch_size, -1)) 
            
        loss = data_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_l2 += loss.item()
        train_dl += data_loss.item()
        train_pl += 0
        
    scheduler.step()
    model2.eval()
    
    
     
    test_l2_step = 0
    test_l2_full = 0

    with torch.no_grad():
        for xx, yy, zz in test_loader2:
            loss = 0
            xx = xx.to(device)
            yy = yy.to(device)
            zz = zz.to(device)
            for t in range(0, T_out-T_in, step):
                z = zz[:,:,:,t:t+step]
                im = model2(xx,yy)
                loss += torch.norm(im-z, p=2)/torch.norm(z, p=2)
                if t == 0:
                   pred = im
                else:
                   pred = torch.cat((pred, im), -1)
                
                xx = torch.cat((xx[..., 1:], yy[:,:,:,-1:]), dim=-1)
                yy = torch.cat((yy[..., 1:], im), dim=-1)
                
            test_l2_step += loss.item()
            test_l2_full += (torch.norm(pred-zz, p=2)/torch.norm(zz, p=2)).item()
            
    train_l2 /= (ntrain*T_out)
    train_dl /= (ntrain*T_out)
    train_pl /= (ntrain*T_out)
    test_l2_step /= (ntest*T_out)
    t2 = default_timer()
    print('Epoch %d - Time %0.4f - Train %0.4f - Data %0.6f - PDE %0.6f - Test %0.6f' 
          % (ep, t2-t1, train_l2, train_dl, train_pl, test_l2_step)) 
   
# %%
""" Prediction """
pred0 = torch.zeros(test_out1.shape)
index = 0
test_e = torch.zeros(test_out1.shape)        
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=1, shuffle=False)
with torch.no_grad():
     for xx, yy, zz in test_loader_pred2:
        test_l2_step = 0
        test_l2_full = 0
        loss = 0
        xx = xx.to(device)
        yy = yy.to(device)
        zz = zz.to(device)
        for t in range(0, T_out-T_in+80, step):
            z = zz[:,:,:,t:t+step]
            im = model2(xx,yy)         
            loss += torch.norm(im-z, p=2)/torch.norm(z, p=2)
            if t == 0:
               pred = im
            else:
               pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., 1:], yy[:,:,:,-1:]), dim=-1)
            yy = torch.cat((yy[..., 1:], im), dim=-1)
            
        
        pred0[index,:,:,:] = pred
        test_l2_step += loss.item()
        test_l2_full += (torch.norm(pred-zz, p=2)/torch.norm(zz, p=2)).item()
        test_e[index] = test_l2_step
        
        print(index, test_l2_step/ ntest/ (T_out/step), test_l2_full/ ntest)
        index = index + 1
        
print('Mean Testing Error:', 100*torch.mean(test_e).numpy()/ (T_out/step), '%')
#%%
scipy.io.savemat('pred/prediction_transformerwno.mat', mdict={'pred': pred0.cpu().numpy()})
# %%

""" Plotting """ # for paper figures please see 'WNO_testing_(.).py' files
figure7 = plt.figure(figsize = (25, 5))
plt.subplots_adjust(hspace=0.3)
batch_no = 9
index = 0
for tvalue in range(test_enc1.shape[3]):
    if tvalue % 1 == 1:
        plt.subplot(1,10, index+1)
        plt.imshow(test_enc1.cpu().numpy()[batch_no,:,:,tvalue], label='True', cmap='jet')
        # plt.colorbar()
        plt.title('Actual')
        # plt.subplot(3,5, index+1+5)
        # plt.imshow(pred0.cpu().detach().numpy()[batch_no,:,:,tvalue], cmap='jet')
        # plt.colorbar()
        # plt.title('Identified')
        # plt.subplot(3,5, index+1+10)
        # plt.imshow(test_out1.cpu().numpy()[batch_no,:,:,tvalue]-pred0.cpu().detach().numpy()[batch_no,:,:,tvalue], cmap='jet')
        # plt.colorbar()
        # plt.title('Error')
        # plt.margins(0)
        index = index + 1

#%%
# torch.save(model2,'model/model_Allencahn')

model2 = torch.load('model/model_Allencahn')
#%%
figure1 = plt.figure(figsize = (15, 5))
figure1.text(0.04,0.57,'\n         Truth', rotation=90, color='red', fontsize=15)
figure1.text(0.04,0.34,'\n    Prediction', rotation=90, color='green', fontsize=15)
figure1.text(0.04,0.17,'\n Error', rotation=90, color='purple', fontsize=15)
# figure1.text(0.04,0.75,'  Initial \n Condition', rotation=90, color='b', fontsize=20)
plt.subplots_adjust(wspace=0.7)
index = 0
bt_no = 1
for value in range(test_out1.shape[-1]):
    if value % 31 == 0:
        print(value)
        # plt.subplot(4,4, index+1)
        # plt.imshow(test_out1.numpy()[15,:,:,0], cmap='jet', extent=[0,1,0,1], interpolation='Gaussian')
        # plt.title('t={}s'.format(value+10), color='b', fontsize=18, fontweight='bold')
        # plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        # plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        plt.subplot(3,5, index+1)
        plt.imshow(test_out1[bt_no,:,:,value], cmap='jet', extent=[0,1,0,1], interpolation='bicubic')
        plt.title('t={}s'.format(value+10), color='b', fontsize=20, fontweight='bold')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        plt.subplot(3,5, index+1+5)
        plt.imshow(pred0[bt_no,:,:,value], cmap='jet', extent=[0,1,0,1], interpolation='bicubic')
        plt.colorbar(fraction=0.045)
        plt.xlabel('x',fontweight='bold'); plt.ylabel('y',fontweight='bold')
        plt.xticks(fontweight='bold'); plt.yticks(fontweight='bold');
        
        plt.subplot(3,5, index+1+10)
        plt.imshow(np.abs(test_out1[bt_no,:,:,value]-pred0[bt_no,:,:,value]), cmap='jet', extent=[0,1,0,1], interpolation='Gaussian',vmin=0,vmax=0.5)
        plt.xlabel('x', fontweight='bold'); plt.ylabel('y', fontweight='bold'); 
        plt.colorbar(fraction=0.045,format='%.0e')
        
        plt.margins(0)
        index = index + 1
        
        
        
#%%




