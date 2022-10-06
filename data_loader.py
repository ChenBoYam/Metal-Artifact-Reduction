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
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ############### test #####################
# CT_img = Image.open('CT-MRI-dataset/pair2D/CT/006876/30_image.jpg').convert('RGB')
# CT_img.show()

# CT_BodyMask = np.load('CT-MRI-dataset/pair2D/Body_Mask/006876/30_label.npy')
# CT_BodyMask = Image.fromarray(CT_BodyMask*255)
# CT_BodyMask.show()

# CT_img2gray = np.load('CT-MRI-dataset/pair2D/CT/006876/30_image.npy')
# CT_img2gray = Image.fromarray(CT_img2gray.reshape(512,512)*255)
# CT_img2gray.show()
# ###########################################

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
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

def get_patient_info(root, patients_id_list):
    patient_info_clear = list()
    patient_info_clear = pd.DataFrame(patient_info_clear, columns = ['name', 'path', 'class']) # clear : 0
    patient_info_noise = list()
    patient_info_noise = pd.DataFrame(patient_info_noise, columns = ['name', 'path', 'class']) # noise : 1
    noise_num = 0
    clear_num = 0
    for i, patient_id in enumerate(patients_id_list):
        patient_id_path = os.path.join(root, patient_id)
        f = open(os.path.join(patient_id_path, 'MA_slice_num.txt'))
        noisy_patients_No = list()
        for line in f.read().splitlines():
            noisy_patients_No.append(line)
        for item in os.listdir(patient_id_path):
            if ('.jpg' in item and item.split('_')[0] in noisy_patients_No):
                patient_info_noise = patient_info_noise.append({'name':item,'path': patient_id_path, 'class': 1}, ignore_index = True)
                noise_num += 1
            elif ('.jpg' in item and item.split('_')[0] not in noisy_patients_No):
                patient_info_clear = patient_info_clear.append({'name':item,'path': patient_id_path, 'class': 0}, ignore_index = True)
                clear_num += 1
    return patient_info_noise, patient_info_clear, noise_num, clear_num

class CTImg(Dataset):

    def __init__(self, transform, patient_info):
        self.patient_info = patient_info
        self.name = patient_info['name'].values
        self.path = patient_info['path'].values
        self.label = patient_info['class'].values
        self.transform = transform

    def __len__(self):
        return len(self.patient_info)

    def __getitem__(self, i):
        # CT_BodyMask = np.load(os.path.join(self.path_M[i],self.name[i]+'_label.npy'))
        # image = cv2.imread(os.path.join(self.path[i],self.name[i]+'_image.jpg'))
        # result1 = image.copy()
        # result1[CT_BodyMask == 0] = 0
        # result1[CT_BodyMask != 0] = image[CT_BodyMask != 0]
        # image = self.transform(Image.fromarray(result1))
        image = self.transform(Image.open(os.path.join(self.path[i],self.name[i])).convert('RGB'))
        name = self.name[i]
        label =  self.label[i]
        return image, label, name



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
            *discriminator_block(256, 256),                                 #[batchsize,  512,  4,  4]
        )
        
        self.final = nn.Sequential(
            nn.Linear(256 * 20 * 20, 1),
            nn.Sigmoid(),
        )


    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        conv = self.model(img)
        conv = conv.view(conv.shape[0], -1)
        return self.final(conv).view(-1)

class DNN(nn.Module):
    def __init__(self, input_shape):
        super(DNN, self).__init__()

        channels, _, _ = input_shape                                        #[batchsize,   3, 64, 64]

        def dnn_block(in_filters, out_filters):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1)]
            layers.append(nn.ReLU(inplace=True))
            for i in range(12):
                layers.append(nn.Conv2d(out_filters, out_filters, 3, stride=1, padding=1))
                layers.append(nn.BatchNorm2d(out_filters))
                layers.append(nn.ReLU(inplace=True))
            return layers

        self.model = nn.Sequential(
            *dnn_block(channels, 64),                                       #[batchsize,  3, 64, 64]
        )
        self.output = nn.Sequential(
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
        )


    def forward(self, img):
        # Concatenate image and condition image by channels to produce input
        return img - self.output(self.model(img))

def diff(x,use_image_gradient ='G1'):
    if use_image_gradient == 'G1':
        g1 = x + nn.AvgPool2d(3, stride=1, padding=1)(x)
        g = torch.cat((x, g1), 1)
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

root ='CT-MRI-dataset/pair2D/CT'
patients_id_list = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]
print("Total number of patients: ", len(patients_id_list))

#####################################################
################  param  ############################
#####################################################
#same_seeds(33)  
batch_size = 4 
num_epoch = 50
lr = 2e-5
channels = 3
img_size = 320
input_shape = (channels, img_size, img_size)
######################################################
######################################################

######################################################
############### split in patients id #################
######################################################
train_id_list, test_id_list = train_test_split(patients_id_list, test_size=0.33)

train_patient_info_noise, train_patient_info_clear, train_noise_num, train_clear_num = get_patient_info(root, train_id_list)
test_patient_info_noise, test_patient_info_clear, test_noise_num, test_clear_num = get_patient_info(root, test_id_list)
print(train_patient_info_noise.sample(n = 10))
print(train_patient_info_clear.sample(n = 10))
print("Train noise: %d, Test noise: %d, Total noise: %d"%(train_noise_num,test_noise_num,train_noise_num + test_noise_num))
print("Train clear: %d, Test clear: %d, Total clear: %d"%(train_clear_num,test_clear_num,train_clear_num + test_clear_num))

train_transform = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
train_transform1 = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.RandomHorizontalFlip(p=0.45),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
train_transform2 = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.RandomHorizontalFlip(p=0.40),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
train_transform3 = transforms.Compose([transforms.Resize((img_size, img_size)),transforms.RandomHorizontalFlip(p=0.35),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

test_transform = transforms.Compose([  
    transforms.Resize((img_size, img_size)),                                 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

train_set_noise1 = CTImg(transform = train_transform, patient_info = train_patient_info_noise)
train_set_noise2 = CTImg(transform = train_transform1, patient_info = train_patient_info_noise)
train_set_noise3 = CTImg(transform = train_transform2, patient_info = train_patient_info_noise)
train_set_noise4 = CTImg(transform = train_transform3, patient_info = train_patient_info_noise)
train_set_noise = ConcatDataset([train_set_noise1, train_set_noise2, train_set_noise3, train_set_noise4])
train_set_noise = ConcatDataset([train_set_noise,train_set_noise])

train_set_clear = CTImg(transform = train_transform, patient_info = train_patient_info_clear)
test_set_noise =  CTImg(transform = test_transform, patient_info = test_patient_info_noise)
test_set_clear =  CTImg(transform = test_transform, patient_info = test_patient_info_clear)


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
Dnn = DNN(input_shape)

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
start_time = time.time()
for epoch in range(num_epoch):
    print("")   
    for i, ((noise_img, noise_label,_), (clear_img, clear_label, _)) in enumerate(zip(train_noise_loader, train_clear_loader)):
        
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
        
        real_label = Variable(noise_label.float().cuda()) #1
        fake_label = Variable(clear_label.float().cuda()) #0
        
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
        if i ==0 or i == 1:
            torchvision.utils.save_image(g_noise.cpu(), './samples/train_gen_noise_img_{:02d}_{:04d}.png'.format(epoch, i))
        g_img = g_noise + Variable(clear_img).cuda()
        if i ==0 or i == 1:
            torchvision.utils.save_image(g_img.cpu(), './samples/train_gen_img_{:02d}_{:04d}.png'.format(epoch, i))
        noisy_fake = diff(g_img)
        fake_logit = Dis(noisy_fake)
        dnn_pred = Dnn(g_img.detach())
        out = g_img.detach() - dnn_pred  
        loss_G = g_loss(fake_logit, torch.ones((len(clear_img))).cuda())
        loss_Dnn = dnn_loss(out,Variable(clear_img).cuda())        
        loss_G.backward()
        optimizer_Gen.step()       
        loss_Dnn.backward()
        optimizer_Dnn.step()

        print("Epoch: [{:2d}] [{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}, dnn_loss: {:.8f}".format(
                    epoch, i, time.time() - start_time, loss_D, loss_G, loss_Dnn),end='\r')
    with torch.no_grad():
        for i, ((noise_img, noise_label,_), (clear_img, clear_label, _)) in enumerate(zip(test_noise_loader, test_clear_loader)):
            if i == 0 or i == 1:
                dnn_pred = Dnn(Variable(noise_img).cuda())
                out = Variable(noise_img).cuda() - dnn_pred
                torchvision.utils.save_image(noise_img.cpu(), './samples/noise_img_{:02d}_{:04d}.png'.format(epoch, i))
                torchvision.utils.save_image(dnn_pred.cpu(),  './samples/dnn_noise_img_{:02d}_{:04d}.png'.format(epoch, i))
                torchvision.utils.save_image(out.cpu(), './samples/denoised_img_{:02d}_{:04d}.png'.format(epoch, i))
            else:
                break

        
        