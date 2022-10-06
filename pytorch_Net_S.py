import os
import numpy as np
from PIL import Image
from PIL import ImageFile
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import time
import scipy
from torch.autograd import Variable
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
from dataloader import get_patient_info, CTImg
from metrics import PSNR, MAE

def get_random_sample(shape, method = 'normal'):
    if method == 'uniform':
        sample_z = np.random.uniform(-1, 1, size = shape).astype(np.float32)
    elif method == 'random':
        sample_z = 2.0 * np.random.random(size = shape) - 1.0
    else:
        sample_z = np.random.normal(size = shape)
        sample_z = (sample_z - sample_z.min()) / (sample_z.max() - sample_z.min())
        sample_z = 2.0 * sample_z - 1.0
    return sample_z

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seedis_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class G_Down(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, pooling=False, dropout=0.0):
        super(G_Down, self).__init__()
        if pooling == True:
            layers = [nn.Conv2d(in_size, out_size, 4, 2, 1)]
        else:
            layers = [nn.Conv2d(in_size, out_size, 3, 1, 1)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2,inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class G_Up(nn.Module):
    def __init__(self, in_size, out_size, uppooling=False, dropout=0.0):
        super(G_Up, self).__init__()
        if uppooling:
            layers = [nn.ConvTranspose2d(in_size, out_size, 4, 2, 1)]
        else:
            layers = [nn.ConvTranspose2d(in_size, out_size, 3, 1, 1)]
        layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.ReLU(inplace=True))
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return x


class Generator(nn.Module):
    def __init__(self, input_shape, cat=True):
        super(Generator, self).__init__()
        
        channels, _, _ = input_shape
        if cat:
            channels*=2 
        self.down1 = G_Down(channels, 32, normalize=False) 
        self.down2 = G_Down(32, 32) 
        self.down3 = G_Down(32, 64, pooling=True, dropout=0.5) 
        self.down4 = G_Down(64, 64)         
        self.down5 = G_Down(64, 128, pooling=True, dropout=0.5) 
        self.down6 = G_Down(128, 128, normalize=False) 

        self.up1 = G_Up(256, 64, uppooling=True, dropout=0.5)
        self.up2 = G_Up(64, 64)
        self.up3 = G_Up(128, 32, uppooling=True, dropout=0.5)
        self.up4 = G_Up(32, 32)
        self.up5 = G_Up(32, 3)

        self.final = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size = 3,stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):               #[batchsize,   6, 64, 64]
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)              #[batchsize,  32, 64, 64]
        d2 = self.down2(d1)             #[batchsize,  32, 64, 64]
        d3 = self.down3(d2)             #[batchsize,  64, 32, 32]
        d4 = self.down4(d3)             #[batchsize,  64, 32, 32]
        d5 = self.down5(d4)             #[batchsize, 128, 16, 16]
        d6 = self.down6(d5)             #[batchsize, 128, 16, 16]
        cat1 = torch.cat((d6, d5), 1)   #[batchsize, 256, 16, 16]
        u1 = self.up1(cat1)             #[batchsize,  64, 32, 32]
        u2 = self.up2(u1)               #[batchsize,  64, 32, 32]
        cat2 = torch.cat((u2, d4), 1)   #[batchsize, 128, 32, 32]
        u3 = self.up3(cat2)             #[batchsize,  32, 64, 64]    
        u4 = self.up4(u3)               #[batchsize,  32, 64, 64]
        u5 = self.up5(u4)               #[batchsize,   3, 64, 64]
        return self.final(u5)           #[batchsize,   3, 64, 64]


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape
        self.input_shape = (channels*2, height, width)                        #[batchsize,   3, 64, 64]
        # Calculate output of image discriminator (PatchGAN)
        self.output_shape = (1, height // 2 ** 3, width // 2 ** 3)

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if normalization:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels*2, 64, normalization=False),      #[batchsize,   64, 32, 32]
            *discriminator_block(64, 128),                                  #[batchsize,  128, 16, 16]
            *discriminator_block(128, 256),                                 #[batchsize,  256,  8,  8]
            *discriminator_block(256, 512),                                 #[batchsize,  512,  4,  4]
        )
        
        self.final = nn.Sequential(
            nn.Linear(512 * 20 * 20, 1),
            nn.Sigmoid(),
        )


    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        conv = self.model(img)
        conv = conv.view(conv.shape[0], -1)
        return self.final(conv).view(-1)

class Denoiser_UNet(nn.Module):
    def __init__(self, input_shape):
        super(Denoiser_UNet, self).__init__()
        
        channels, _, _ = input_shape
        self.down1 = G_Down(channels, 32, normalize=False) 
        self.down2 = G_Down(32, 32) 
        self.down3 = G_Down(32, 64, pooling=True, dropout=0.5) 
        self.down4 = G_Down(64, 64)         
        self.down5 = G_Down(64, 128, pooling=True, dropout=0.5) 
        self.down6 = G_Down(128, 128, normalize=False) 

        self.up1 = G_Up(256, 64, uppooling=True, dropout=0.5)
        self.up2 = G_Up(64, 64)
        self.up3 = G_Up(128, 32, uppooling=True, dropout=0.5)
        self.up4 = G_Up(32, 32)
        self.up5 = G_Up(32, 3)

        self.final = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size = 3,stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, img):                 #[batchsize,   3, inputshape, inputshape]
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(img)                #[batchsize,  32, inputshape, inputshape]
        d2 = self.down2(d1)                 #[batchsize,  32, inputshape, inputshape]
        d3 = self.down3(d2)                 #[batchsize,  64, inputshape/2, inputshape/2]
        d4 = self.down4(d3)                 #[batchsize,  64, inputshape/2, inputshape/2]
        d5 = self.down5(d4)                 #[batchsize, 128, inputshape/4, inputshape/4]
        d6 = self.down6(d5)                 #[batchsize, 128, inputshape/4, inputshape/4]
        cat1 = torch.cat((d6, d5), 1)       #[batchsize, 256, inputshape/4, inputshape/4]
        u1 = self.up1(cat1)                 #[batchsize,  64, inputshape/2, inputshape/2]
        u2 = self.up2(u1)                   #[batchsize,  64, inputshape/2, inputshape/2]
        cat2 = torch.cat((u2, d4), 1)       #[batchsize, 128, inputshape/2, inputshape/2]
        u3 = self.up3(cat2)                 #[batchsize,  32, inputshape, inputshape]    
        u4 = self.up4(u3)                   #[batchsize,  32, inputshape, inputshape]
        u5 = self.up5(u4)                   #[batchsize,   3, inputshape, inputshape]
        return self.final(u5)               #[batchsize,   3, inputshape, inputshape] 

def diff(x,use_image_gradient ='G1'):
    if use_image_gradient == 'G1':
        g1 = x - nn.AvgPool2d(3, stride=1, padding=1)(x)
        g2 = x - nn.AvgPool2d(7, stride=1, padding=3)(x)
        g = torch.cat((g1, g2), 1)
    else:
        g = x
    return g

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
def merge(images, size):
    if(len(images.shape) > 3):
        h, w, c = images.shape[1], images.shape[2], images.shape[3]
        img = np.zeros((int(h * size[0]), int(w * size[1]), c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image

        if c == 1:
            img = img.reshape(img.shape[0], img.shape[1])
    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((int(h * size[0]), int(w * size[1])))

        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image
    return img
import imageio

def imsave(images, size, path):
    merge_img = 255 * merge(images, size)
    merge_img = np.clip(merge_img, 0, 255).astype(np.uint8)
    return imageio.imwrite(path, merge_img)   
def save_image(image, image_path):
    image = 255 * inverse_transform(image)
    image = np.clip(image, 0, 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[-1] == 1:
        image = np.reshape(image, (image.shape[0], image.shape[1]))
    scipy.misc.imsave(image_path, image)
def inverse_transform(images):
    return (images + 1.) / 2.
#####################################################
################ 1.  param  ############################
#####################################################
#same_seeds(33)  
batch_size = 8 
num_epoch = 35
lr = 2e-5
channels = 3
img_size = 320
input_shape = (channels, img_size, img_size)

root ='dataset'

CT_dir = os.path.join(root, "CT") # supervised + unsupervised
OMA_dir = os.path.join(root, "OMA") # supervised
Mask_dir = os.path.join(root, "Body_Mask") # supervised + unsupervised
######################################################
######################################################
patients_id_list_test = [ item for item in os.listdir(OMA_dir) if os.path.isdir(os.path.join(OMA_dir, item)) ]
patients_id_list_train = [ item for item in os.listdir(CT_dir) if (os.path.isdir(os.path.join(CT_dir, item)) and item not in patients_id_list_test)]
print("Total number of patients: ", len(patients_id_list_test)+len(patients_id_list_train))

train_patient_info_noise, train_patient_info_clear, train_noise_num, train_clear_num = get_patient_info(CT_dir, OMA_dir, patients_id_list_train, semi=True)
test_patient_info_noise, test_patient_info_clear, test_noise_num, test_clear_num = get_patient_info(CT_dir, OMA_dir, patients_id_list_test, semi=True)



############### 2. split in patients id #################
print(train_patient_info_noise.sample(n = 10))
print(train_patient_info_clear.sample(n = 10))
print("Train noise: %d, Test noise: %d, Total noise: %d"%(train_noise_num,test_noise_num,train_noise_num + test_noise_num))
print("Train clear: %d, Test clear: %d, Total clear: %d"%(train_clear_num,test_clear_num,train_clear_num + test_clear_num))

train_transform = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([  
    transforms.Resize((img_size, img_size)),                                 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

train_set_noise1 = CTImg(transform = train_transform, patient_info = train_patient_info_noise,CT_dir=CT_dir,OMA_dir=OMA_dir,Mask_dir=Mask_dir)

train_set_noise = ConcatDataset([train_set_noise1, train_set_noise1, train_set_noise1, train_set_noise1])
train_set_noise = ConcatDataset([train_set_noise,train_set_noise])

train_set_clear = CTImg(transform = train_transform, patient_info = train_patient_info_clear,CT_dir=CT_dir,OMA_dir=OMA_dir,Mask_dir=Mask_dir)
test_set_noise =  CTImg(transform = test_transform, patient_info = test_patient_info_noise,CT_dir=CT_dir,OMA_dir=OMA_dir,Mask_dir=Mask_dir)
test_set_clear =  CTImg(transform = test_transform, patient_info = test_patient_info_clear,CT_dir=CT_dir,OMA_dir=OMA_dir,Mask_dir=Mask_dir)


train_noise_loader = DataLoader(train_set_noise, batch_size = batch_size, shuffle=True)
train_clear_loader = DataLoader(train_set_clear, batch_size = batch_size, shuffle=True)
test_noise_loader = DataLoader(test_set_noise, batch_size = batch_size, shuffle=False)
test_clear_loader = DataLoader(test_set_clear, batch_size = batch_size, shuffle=False)


g_loss = torch.nn.BCEWithLogitsLoss()
d_loss = torch.nn.BCEWithLogitsLoss()
dnn_loss = torch.nn.MSELoss()

cuda = torch.cuda.is_available()

# Initialize generator and discriminator
Gen = Generator(input_shape)
Dis = Discriminator(input_shape)
Dnn = Denoiser_UNet(input_shape)

if cuda:
    Gen = Gen.cuda()
    Dis = Dis.cuda()
    Dnn = Dnn.cuda()
    g_loss.cuda()
    d_loss.cuda()
    dnn_loss.cuda()


# Initialize weights
Gen.apply(weights_init_normal)
Dis.apply(weights_init_normal)
Dnn.apply(weights_init_normal)


# Optimizers
optimizer_Gen = torch.optim.Adam(Gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_Dis = torch.optim.Adam(Dis.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_Dnn = torch.optim.Adam(Dnn.parameters(), lr=lr, betas=(0.5, 0.999))

# Input tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
fix_batch_sample_z = Tensor(get_random_sample(([batch_size] + list(input_shape)), method = 'uniform'))

# axes[0].title.set_text('origin_clear')
# axes[1].title.set_text('gen_noise')
# axes[2].title.set_text('gen_img')
# axes[3].title.set_text('origin_clear')
# axes[4].title.set_text('dnn_noise')
# axes[5].title.set_text('dnn_img')

def clp(img):
    return torch.clamp(img, -1, 1) * 0.5 + 0.5
    
metric_list = list()
metric_list = pd.DataFrame(metric_list, columns = ['N_GT_psnr', 'DN_GT_psnr', 'N_GT_mae', 'DN_GT_mae', 'N_GT_ssim', 'DN_GT_ssim'])    

start_time = time.time()
for epoch in range(num_epoch):
    print("")
    
    for i, ((noise_img, noise_cls,_,_,_), (clear_img, clear_cls,_,_,_)) in enumerate(zip(train_noise_loader, train_clear_loader)):
           
        """ Train D """
        optimizer_Dis.zero_grad()
        batch_sample_z = Tensor(get_random_sample(([len(clear_img)] + list(input_shape)), method = 'uniform'))
        g_noise = Gen(torch.cat((Variable(batch_sample_z).cuda(),Variable(clear_img).cuda()), 1))
        g_img = g_noise + Variable(clear_img).cuda()

        noisy_real = diff(Variable(noise_img).cuda())
        noisy_fake = diff(g_img)
        #if i ==0:
        #    print(f"shape of noisy_real: {noisy_real.shape}, shape of noisy_fake: {noisy_fake.shape}")
        real_logit = Dis(noisy_real.detach())
        fake_logit = Dis(noisy_fake.detach())
        
        real_label = Variable(noise_cls.float().cuda()) #1
        fake_label = Variable(clear_cls.float().cuda()) #0
        
        real_loss = d_loss(real_logit, real_label)
        fake_loss = d_loss(fake_logit, fake_label)
        loss_D = (real_loss + fake_loss) / 2
        
        loss_D.backward()
        optimizer_Dis.step()
        
        """ train G and Dnn"""
        optimizer_Gen.zero_grad()
        optimizer_Dnn.zero_grad()
        batch_sample_z = Tensor(get_random_sample(([len(clear_img)] + list(input_shape)), method = 'uniform'))
        g_noise = Gen(torch.cat((Variable(batch_sample_z).cuda(),Variable(clear_img).cuda()), 1))
        # if i ==0 :
        #     torchvision.utils.save_image(g_noise.cpu(), './samples/train_gen_noise_img_{:02d}_{:04d}.png'.format(epoch, i))
        #     plt.imshow(np.transpose(g_noise.cpu().numpy(), (1,2,0)))
        g_img = g_noise + Variable(clear_img).cuda()
        # if i ==0 :
        #     plt.imshow(np.transpose(g_img.cpu().numpy(), (1,2,0)))
        #     torchvision.utils.save_image(g_img.cpu(), './samples/train_gen_img_{:02d}_{:04d}.png'.format(epoch, i))
        noisy_fake = diff(g_img)
        fake_logit = Dis(noisy_fake)         
        loss_G = g_loss(fake_logit, torch.ones((len(clear_img))).cuda())
        loss_G.backward()
        optimizer_Gen.step()
        
        loss_Dnn = 0   
        if epoch > 5:        
            dnn_pred = Dnn(g_noise.detach())
            out = g_img.detach() - dnn_pred    
            loss_Dnn = dnn_loss(out,Variable(clear_img).cuda())  
            loss_Dnn.backward()
            optimizer_Dnn.step()      
        print("Epoch: [{:2d}] [{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}, dnn_loss: {:.8f}".format(
                        epoch, i, time.time() - start_time, loss_D, loss_G, loss_Dnn),end='\r')
    
    with torch.no_grad():
        psnr = PSNR()
        mae = MAE()
        N_GT_psnr, DN_GT_psnr, N_GT_mae, DN_GT_mae, N_GT_ssim, DN_GT_ssim = 0, 0, 0, 0, 0, 0
        for i, ((noise_img, _,_,noise_label,_), (clear_img,_,_,clear_label,_)) in enumerate(zip(test_noise_loader, test_clear_loader)):
            '''Gen'''
            g_noise = Gen(torch.cat((Variable(fix_batch_sample_z).cuda(),Variable(clear_img).cuda()), 1))            
            g_img = g_noise + Variable(clear_img).cuda()
            '''Dnn'''
            dnn_pred = Dnn(Variable(noise_img).cuda())
            out = Variable(noise_img).cuda() - dnn_pred
            batch_len = len(out)
            for (noise,label) in zip(Variable(noise_img).cuda(),Variable(noise_label).cuda()): 
                N_GT_psnr += psnr(noise, label)/batch_len
                #N_GT_ssim += compare_ssim(noise,label)/batch_len
                N_GT_mae += mae(noise,label)/batch_len

            for (denoise,label) in zip(out,Variable(noise_label).cuda()): 
                DN_GT_psnr += psnr(clp(denoise), label)/batch_len
                #DN_GT_ssim += compare_ssim(denoise,label)/batch_len
                DN_GT_mae += mae(clp(denoise), label)/batch_len

            if  i == 4:                
                fig = plt.figure(figsize=[8*6,8*4])
                axes = [fig.add_subplot(6, 1, r+1 ) for r in range(0, 6)]
                for ax in axes:
                    ax.axis('off')
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0) 
                axes[0].imshow(torchvision.utils.make_grid(clear_img.cpu(), nrow=8).permute(1, 2, 0))
                #torchvision.utils.save_image(clear_img.cpu(), './samples/origin_clear_ep{:02d}-{:04d}.png'.format(epoch, i))               
                axes[1].imshow(torchvision.utils.make_grid(g_noise.cpu(), nrow=8).permute(1, 2, 0))
                #torchvision.utils.save_image(g_noise.cpu(), './samples/gen_noise_ep{:02d}-{:04d}.png'.format(epoch, i))                
                axes[2].imshow(torchvision.utils.make_grid(g_img.cpu(), nrow=8).permute(1, 2, 0))
                #torchvision.utils.save_image(g_img.cpu(), './samples/gen_img_ep{:02d}-{:04d}.png'.format(epoch, i))                                                         
                axes[3].imshow(torchvision.utils.make_grid(noise_img.cpu(), nrow=8).permute(1, 2, 0))
                #torchvision.utils.save_image(noise_img.cpu(), './samples/origin_noise_ep{:02d}-{:04d}.png'.format(epoch, i))
                axes[4].imshow(torchvision.utils.make_grid(dnn_pred.cpu(), nrow=8).permute(1, 2, 0))
                #torchvision.utils.save_image(dnn_pred.cpu(),  './samples/dnn_noise_ep{:02d}-{:04d}.png'.format(epoch, i))
                axes[5].imshow(torchvision.utils.make_grid(out.cpu(), nrow=8).permute(1, 2, 0))
                #torchvision.utils.save_image(out.cpu(), './samples/denoised_img_ep{:02d}-{:04d}.png'.format(epoch, i))
                fig.savefig("results/DNN2UNet_SP/DNN2UNet_SP_ep{:02d}.png".format(epoch),bbox_inches = 'tight',pad_inches = 0)
                plt.close(fig)
                print("saving...")
        l = len(test_noise_loader)
        print("Epoch: [{:2d}], N_GT_psnr: {:.8f}, DN_GT_psnr: {:.8f}, N_GT_mae: {:.8f}, DN_GT_mae: {:.8f}, N_GT_ssim: {:.8f}, DN_GT_ssim: {:.8f}".format(
                        epoch, N_GT_psnr/l, DN_GT_psnr/l, N_GT_mae/l, DN_GT_mae/l, N_GT_ssim/l, DN_GT_ssim/l))
        metric_list = metric_list.append({'N_GT_psnr':N_GT_psnr/l,'DN_GT_psnr': DN_GT_psnr/l, 
                'N_GT_mae': N_GT_mae/l, 'DN_GT_mae': DN_GT_mae/l, 
                'N_GT_ssim': N_GT_ssim/l, 'DN_GT_ssim' : DN_GT_ssim/l}, ignore_index = True)

metric_list.to_csv('results/DNN2UNet_SP/DNN2UNet_SP_Result.csv')
        
        