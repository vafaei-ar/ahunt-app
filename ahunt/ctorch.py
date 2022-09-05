
from __future__ import print_function, division

import os
import numpy as np
import pandas as pd
import pylab as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
import streamlit as st
from skimage import io, transform

import torch
from torch.optim import Adam
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader
from torchvision import models,datasets
from torchvision import transforms, utils
from torchmetrics.functional import accuracy
from torch.nn.functional import cross_entropy
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix,accuracy_score


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from ssaip.alservice import ALServiceBase

TRANSFORM_TRAIN = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Lambda(lambda x: x.convert("RGB") ),
    transforms.RandomApply(torch.nn.ModuleList([transforms.CenterCrop(400),]), p=0.8),
    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomVerticalFlip(),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomHorizontalFlip(),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomRotation(degrees=30),]), p=0.9),
    transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(brightness=.5, hue=.3),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomPerspective(distortion_scale=0.6, p=1.0),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomPosterize(bits=2),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomSolarize(threshold=64.0),]), p=0.5),
    transforms.RandomApply(torch.nn.ModuleList([transforms.RandomEqualize(),]), p=0.5),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

TRANSFORM_DEPLOY = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Lambda(lambda x: x.convert("RGB") ),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225] )
    ])

class TorchCSVLoader(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataframe, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.landmarks_frame = dataframe
        self.root_dir = root_dir
        self.transform = transform
        
        cidx_pat = os.path.join(root_dir,'idx_to_class')
        if os.path.exists(cidx_pat+'npy'):
            cname_old = np.load(cidx_pat+'npy')
            cname_old = sorted(cname_old)
            cname_new = dataframe['label'].dropna().unique().tolist()
            cname_new = sorted(cname_new)
            cname_diff = np.setdiff1d(cname_new,cname_old).tolist()
            cname_diff = sorted(cname_diff)
            self.classes = cname_old+cname_diff
            self.class_to_idx = {j:i for i,j in enumerate(self.classes)}
            np.save(cidx_pat,self.classes)
        else:
            self.classes = dataframe['label'].dropna().unique().tolist()
            self.classes = sorted(self.classes)
            self.class_to_idx = {j:i for i,j in enumerate(self.classes)}
            np.save(cidx_pat,self.classes)
        
        int_labels = dataframe['label'].apply(lambda i:self.class_to_idx[i]).values
        self.imgs = [(i,j) for i,j in zip(dataframe.index,int_labels)]

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        
        # npimgs = np.array(self.imgs)
        img_name = os.path.join(self.root_dir,
                                self.imgs[idx][0])
        # image = io.imread(img_name)
        # image = read_image(img_name)
        image = Image.open(img_name)#.convert("RGB")
        
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # image = image.to(device)
        
        landmarks = int(self.imgs[idx][1])
        # landmarks = np.array([landmarks])
        # landmarks = landmarks.astype('float').reshape(-1, 2)
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'landmarks': landmarks}

        return sample



def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item[1]] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val[1]]                                  
    return weight     


class ImageClassifier(pl.LightningModule):
    def __init__(self, model_name, num_classes=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
#         self.model = models.resnet50(pretrained=True)
        exec('self.model = models.'+model_name+'(pretrained=True)')
#         self.model = model
#         for param in self.model.parameters():
#             param.requires_grad = False

#         last_layer_name = list(dict(self.model.named_modules()).keys())[-1]
#         print(last_layer_name)
#         exec('self.model.'+last_layer_name+' = torch.nn.Linear(self.model.'+last_layer_name+'.in_features, num_classes)')
        
        ll = list(self.model.children())[-1]
        if type(list(self.model.children())[-1]) is torch.nn.modules.container.Sequential:
            try:
                n_latent = list(self.model.children())[-1][0].in_features
            except:
                n_latent = list(self.model.children())[-1][1].in_features
        else:
            n_latent = list(self.model.children())[-1].in_features
        
        removed = list(self.model.children())[:-1]
        self.model= torch.nn.Sequential(*removed)        
#         self.model = torch.nn.Sequential(self.model, torch.nn.Linear(n_latent,num_classes))
        if 'Dropout' in str(self.model):
            self.model = torch.nn.Sequential(self.model, torch.nn.Flatten(), torch.nn.Linear(n_latent,num_classes))
        else:
            self.model = torch.nn.Sequential(self.model, torch.nn.Flatten(), torch.nn.Dropout(0.3), torch.nn.Linear(n_latent,num_classes))

    def training_step(self, batch, batch_idx):
        # return the loss given a batch: this has a computational graph attached to it: optimization
        x, y = batch
        preds = self.model(x)
        loss = cross_entropy(preds, y)
        self.log('train_loss', loss)  # lightning detaches your loss graph and uses its value
        self.log('train_acc', accuracy(preds, y), prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        # return optimizer
        optimizer = Adam(self.model.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def predict_datapoint(self,x,transform=None):
        if self.training:
            self.eval()
        if not transform is None:
            x = transform(x)
        x = x[None,...]
        with torch.no_grad():
            pred = self.model(x)
        pred = torch.nn.functional.softmax(pred, dim=1)
        idp = torch.argmax(pred)
        return idp.numpy()        

class ALServiceTorch(ALServiceBase):

    def train(self,als_config=None):
        
        batch_size = 2
        autotrain = False
        checkpoints_dir = os.path.join(self.root_dir,'checkpoints')
        
        if als_config:
            batch_size = als_config['batch_size']
            autotrain = als_config['autotrain']
        
        df = self.session.df
        dataframe = df[~df['label'].isna()]
        train_data = TorchCSVLoader(dataframe = dataframe,
                                    root_dir = self.root_dir,
                                    transform=TRANSFORM_TRAIN
                                    )

        # train_data = datasets.ImageFolder(root=self.csv_file, transform=TRANSFORM_IMG)

        # For unbalanced dataset we create a weighted sampler                       
        weights = make_weights_for_balanced_classes(train_data.imgs, len(train_data.classes))                                                                
        weights = torch.DoubleTensor(weights)                                       
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))       


        self.train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, 
                                    sampler = sampler, num_workers=1, pin_memory=True,
                                    persistent_workers=True)
        n_class = len(train_data.classes)

        st.write(train_data.class_to_idx)

        # for i in range(len(train_data_loader)):
        #     sample = train_data_loader[i]

        #     print(i, sample['image'].size(), sample['landmarks'].size())

        #     if i == 3:
        #         break

        # models_list = [i for i in dir(models) if i[0].islower()]
        # print(models_list)
        # i_model = 9
        model_name = 'efficientnet_b1' #models_list[i_model]

        classifier = ImageClassifier(model_name=model_name,num_classes=n_class)
        checkpoint_callback = ModelCheckpoint(dirpath=checkpoints_dir,
                                              filename='{epoch:03d}')

        trainer = pl.Trainer(gpus=1, max_epochs=10,
                            callbacks=[checkpoint_callback])  # for Colab: set refresh rate to 20 instead of 10 to avoid freezing
        trainer.fit(classifier, self.train_data_loader)  # train_loader
        
            # def predict

        # y_true = []
        # y_pred = []

        # # pbar = tqdm(total=len(glob(TRAIN_DATA_PATH+"*/*")), position=0, leave=True)
        # # for i,clas in enumerate(classes):
        # files = self.session.df['path'].value
        # y_true = self.session.df['label'].value
        # for fil in files:
        #     img = Image.open(fil)#.convert("RGB")
        #     idp = classifier.predict_datapoint(img,TRANSFORM_DEPLOY)
        #     y_pred.append(idp)
        # y_pred = np.array(y_pred)

        # y_pred_names = [train_data.classes[i] for i in y_pred]
        
        # self.session.df['predict'] = y_pred_names
        # self.session.df['score'] = y_pred_names

        # self.session.df.to_csv(os.path.join(self.root_dir,'labels.csv'))

        # list_of_files = glob(checkpoints_dir+'/*.ckpt') # * means all if need specific format then *.csv
        # latest_checkpoint = max(list_of_files, key=os.path.getctime)
        # print('The latest checkpoint is',latest_checkpoint)

        # classifier = ImageClassifier(model_name=model_name,num_classes=n_class)
        # classifier = classifier.load_from_checkpoint(latest_checkpoint);

        # dataframe = self.session.df
        # train_data = TorchCSVLoader(dataframe = dataframe,
        #                             root_dir = self.root_dir,
        #                             transform=TRANSFORM_TRAIN
        #                             )