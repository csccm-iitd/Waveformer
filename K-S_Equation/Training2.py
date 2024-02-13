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
from pytorch_wavelets import DWT1D, IDWT1D
torch.manual_seed(0)
np.random.seed(0)


# device= torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda:1')
# %%
""" Model configurations """

TRAIN_PATH = 'data/data_sol.mat'
ntrain = 180
ntest = 20

# num_epochs = 1
num_epochs2 = 200
# batch_size = 80
learning_rate = 1e-4


step_size = 20
gamma = 0.75

level = 3# The automation of the mode size will made and shared upon acceptance and final submission
width = 80

batch_size = 2

r = 1
# h = int(((241 - 1)/r) + 1)
s = r

T_in = 50
# T = 10
T_out = 110
step =1

""" Read data """
reader = MatReader(TRAIN_PATH)
x_train_d = reader.read_field('sol')

x_train = x_train_d[:ntrain,::r,::r]
x_train = x_train[:,:,700:]


# x_train_enc0 = x_train[:,:,:,0:T_in]
# x_train_out0 = x_train[:,:,:,T_in:T_out]

x_train_enc1 = x_train[:,:,0:T_in]
x_train_enc2 = x_train[:,:,1:T_in+1]
x_train_out1 = x_train[:,:,1:T_out+1]

x_test = x_train_d[ntrain:ntrain+ntest,::r,::r]
x_test = x_test[:,:,700:]

# test_enc0 = x_test[:,:,:,0:T_in]
# test_out0  = x_test[:,:,:,T_in:T_out]

test_enc1 = x_test[:,:,0:T_in]
test_enc2 = x_test[:,:,1:T_in+1]
test_out1  = x_test[:,:,T_in+1:T_out+1+200]

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
model2 = WNO1dtransformer(width, level, x_train_enc1[0:1,:,0:T_in].permute(0,2,1)).to(device)
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
        for t in range(0,T_out-T_in, step):
            # x = xx[:,:,:,t:t+T_in]
            # y = yy[:,:,:,t:t+T_in]
            z = zz[:,:,t+T_in:t+T_in+step]
            im =  model2(xx,yy)  
            if t == 0:
               pred = im
            else:
               pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., 1:], yy[:,:,-1:]), dim=-1)
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
                z = zz[:,:,t:t+step]
                im = model2(xx,yy)
                loss += torch.norm(im-z, p=2)/torch.norm(z, p=2)
                if t == 0:
                   pred = im
                else:
                   pred = torch.cat((pred, im), -1)
                
                xx = torch.cat((xx[..., 1:], yy[:,:,-1:]), dim=-1)
                yy = torch.cat((yy[..., 1:], im), dim=-1)
                
            test_l2_step += loss.item()
            test_l2_full += (torch.norm(pred-zz, p=2)/torch.norm(zz, p=2)).item()
            
    train_l2 /= (ntrain*T_out)
    train_dl /= (ntrain*T_out)
    train_pl /= (ntrain*T_out)
    test_l2_step /= (ntest*T_out)
    t2 = default_timer()
    print('Epoch %d - Time %0.6f - Train %0.6f - Data %0.6f - PDE %0.6f - Test %0.6f' 
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
        for t in range(0, T_out-T_in+200, step):
            z = zz[:,:,t:t+step]
            im = model2(xx,yy)         
            loss += torch.norm(im-z, p=2)/torch.norm(z, p=2)
            if t == 0:
               pred = im
            else:
               pred = torch.cat((pred, im), -1)
            xx = torch.cat((xx[..., 1:], yy[:,:,-1:]), dim=-1)
            yy = torch.cat((yy[..., 1:], im), dim=-1)
            
        
        pred0[index,:,:] = pred
        test_l2_step += loss.item()
        test_l2_full += (torch.norm(pred-zz, p=2)/torch.norm(zz, p=2)).item()
        test_e[index] = test_l2_step
        
        print(index, test_l2_step/ ntest/ (T_out/step), test_l2_full/ ntest)
        index = index + 1
        
print('Mean Testing Error:', 100*torch.mean(test_e).numpy()/ (T_out/step), '%')

# %%
# """ Plotting """ # for paper figures please see 'WNO_testing_(.).py' files
# figure7 = plt.figure(figsize = (10, 5))
# plt.subplots_adjust(hspace=0.2)
# batch_no = 0
# index = 0
# for tvalue in range(test_out1.shape[3]):
#     if tvalue % 4 == 1:
#         plt.subplot(3,5, index+1)
#         plt.imshow(test_out1.cpu().numpy()[batch_no,:,T_in:T_in+T_out], label='True', cmap='jet')
#         plt.colorbar()
#         plt.title('Actual')
#         plt.subplot(3,5, index+1+5)
#         plt.imshow(pred0.cpu().detach().numpy()[batch_no,:,T_in:T_in+T_out], cmap='jet')
#         plt.colorbar()
#         plt.title('Identified')
#         plt.subplot(3,5, index+1+10)
#         plt.imshow(test_out1.cpu().numpy()[batch_no,:,T_in:T_in+T_out]-pred0.cpu().detach().numpy()[batch_no,:,T_in:T_in+T_out], cmap='jet')
#         plt.colorbar()
#         plt.title('Error')
#         plt.margins(0)
#         index = index + 1



time_steps = torch.tensor(range(0,test_out1.shape[2]))
error_t = torch.mean((pred0-test_out1)**2,[0,1])/torch.mean((test_out1)**2,[0,1])
plt.plot(time_steps,error_t)
