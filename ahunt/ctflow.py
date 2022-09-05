
from __future__ import print_function, division

import os
from tabnanny import verbose
import numpy as np
import pandas as pd
import pylab as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
import streamlit as st
from skimage import io, transform
from sklearn.metrics import confusion_matrix,accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from ssaip.alservice import ALServiceBase


class ALServiceTFlow(ALServiceBase):

    def train(self):
        
        if not self.als_config:
            self.als_config = {'batch_size':32,'autotrain':False,'model_name':'VGG19'}
        
        batch_size = self.als_config['batch_size']
        autotrain = self.als_config['autotrain']
        model_name = self.als_config['model_name']
        checkpoints_dir = os.path.join(self.root_dir,'als_files','checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)
        
        # if self.als_config:
        #     batch_size = als_config['batch_size']
        #     autotrain = als_config['autotrain']
        
        df = self.session.df
        dataframe = df[~df['label'].isna()]
 
        cidx_pat = os.path.join(self.root_dir,'als_files','idx_to_class')
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
        
        train_images = [np.array(
                        Image.open(
                            os.path.join(self.root_dir,i[0])
                            ).resize((256,256))
                                   ) for i in self.imgs]
        train_labels = [i[1] for i in self.imgs]

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        print(train_images.shape,train_labels.shape)

        aug = ImageDataGenerator(rotation_range = 10,
                                width_shift_range = 0.1,
                                height_shift_range = 0.1,
                                zoom_range = 0.2,
                                fill_mode="nearest")

        y_train = tf.keras.utils.to_categorical(train_labels)
        n_class,class_labels, nums = describe_labels(y_train,verbose=1)
        train_images,y_train = balance_aug(train_images,y_train)
        n_class,class_labels, nums = describe_labels(y_train,verbose=1)
                

        loss = tf.keras.losses.CategoricalCrossentropy()
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                    initial_learning_rate=1e-3,
                                    decay_steps=50,
                                    decay_rate=0.95)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        model_path = os.path.join(checkpoints_dir,model_name+'.h5')
        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            if n_class>model.output.shape[-1]:
                
                model = add_class(model,n_class = n_class,summary=0)
            
        else:
            model,encoder,model_name = build_model(imodel=model_name,
                                                shape=train_images.shape[1:],
                                                n_class = n_class,
                                                n_latent=16,
                                                comp=False)


        model.compile(loss=loss, optimizer=opt, metrics=["accuracy"])
        
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        epoch = 10
        for iepoch in range(epoch):
            H = model.fit(train_images, y_train,
                          validation_split=0.1,
                          batch_size=batch_size,
                          epochs=1,
                          verbose = 0)
            progress_bar.progress((iepoch+1)/epoch)
            status_text.text('Training... {:4.2f}% complete.'.format(100*(iepoch+1)/epoch))
            
            metric_key = 'accuracy'
            lss1 = H.history['loss'][0]
            acc1 = H.history[metric_key][0]
            lss2 = H.history['val_loss'][0]
            acc2 = H.history['val_'+metric_key][0]
        
            df_acc = pd.DataFrame(columns=['train','valid'],data=[[acc1,acc2]])
            if iepoch==0:
                chart1 = st.sidebar.line_chart(df_acc)
                tit = metric_key.replace('_',' ')
                # col1.write(tit)
                st.sidebar.markdown(f"<h4 style='text-align: center; color: Black;'>{tit}</h4>", unsafe_allow_html=True)
            else:
                chart1.add_rows(df_acc)

            # df_lss = pd.DataFrame(columns=['train','valid'],data=[[lss1,lss2]])
            # if epoch==0:
            #     chart2 = st.sidebar.line_chart(df_lss)
            #     # col2.write('loss')
            #     st.markdown("<h4 style='text-align: center; color: Black;'>loss</h4>", unsafe_allow_html=True)
            # else:
            #     chart2.add_rows(df_lss)
            
            

        model.save(model_path)
        # for in df.index:
        #     chunk = 
        #     l = [1,2,3,4,5,6,7,8,9,10]
        #     batch_size = 3    
            
        progress_bar.progress(0/epoch)
        status_text.text('Preiction... {:4.2f}% complete.'.format(0/epoch))
        img_path = df.index.to_list()
        y_pred = []
        nimg = len(img_path)
        for i in range(0, nimg, 32):
            chunk = img_path[i:i+batch_size]
            
            all_images = [np.array(
                            Image.open(
                                os.path.join(self.root_dir,i)
                                ).resize((256,256))
                                    ) for i in chunk]
            all_images = np.array(all_images)
            y_predp = model.predict(all_images)
            y_pred.extend(list(y_predp))

            progress_bar.progress((i+1)/nimg)
            status_text.text('Preiction... {:4.2f}% complete.'.format(100*(i+1)/nimg))
            
        progress_bar.progress(100)
        status_text.text('Preiction... {:4.2f}% complete.'.format(100))
        self.session.labeler_config['trained_model'] = True
        
        y_pred = np.array(y_pred)
        preds = np.argmax(y_pred,axis=1)
        y_pred_names = [self.classes[i] for i in preds]
        self.session.df['predict'] = y_pred_names
        self.session.df['score'] = np.max(y_pred,axis=1)
        self.session.df.to_csv(os.path.join(self.root_dir,'als_files','labels.csv'))
        st.sidebar.write('Done!')


def describe_labels(y0,verbose=0):
    y = y0+0
    if y.ndim==2:
        y = np.argmax(y,axis=1)
    class_labels, nums = np.unique(y,return_counts=True)
    n_class = len(class_labels)
    if verbose:
        print('labels/numbers are:\n',*['{:5s}/{:6d}\n'.format(str(i),j) for i,j in zip(class_labels,nums)])
    return n_class,class_labels, nums

def augment(aug,x):
    aug.fit(x)
    out = []
    for i in x:
        out.append(aug.random_transform(i))
    return np.array(out)

def balance_aug(x0,y0,aug=None,mixup=False):
    x = x0+0
    y = y0+0
    n_class,class_labels, nums = describe_labels(y,verbose=0)
    nmax = max(nums)
    for i,(lbl,n0) in enumerate(zip(class_labels,nums)):
        if nmax==n0:
            continue
        delta = nmax-n0
        if y.ndim==1:
            filt = y==lbl
        elif y.ndim==2:
            filt = y[:,i].astype(bool)
        else:
            assert 0,'Unknown label shape!'
        x_sub = x[filt]
        y_sub = y[filt]
        inds = np.arange(n0)
        nrep = (nmax//len(inds))+1
        inds = np.repeat(inds, nrep)
        np.random.shuffle(inds)
        inds = inds[:delta]
        x_sub = x_sub[inds]
        y_sub = y_sub[inds]
        if not aug is None:
            x_sub = augment(aug,x_sub)
        x = np.concatenate([x,x_sub],axis=0)
        y = np.concatenate([y,y_sub],axis=0)
    return x,y

# ## Build the model
def build_model(imodel,shape, n_class, n_latent=16,comp=True):
    tf.keras.backend.clear_session()

    if False:#imodel>=100:
        from sys import path
        path.insert(0,'./ConvNeXt-TF/')
        from models.convnext_tf import get_convnext_model
        if imodel==100: 
            mdl = 'Convnext_Base'
            model = get_convnext_model(input_shape=shape,num_classes=2,depths=[3, 3, 27, 3],dims=[128, 256, 512, 1024])
        if imodel==101:
            mdl = 'Convnext_XL'
            model = get_convnext_model(input_shape=shape,num_classes=2,depths=[3, 3, 27, 3],dims=[256, 512, 1024, 2048])
        encoder = ''

    else:
        Model_list = [i for i in dir(applications) if i[0].isupper()]
    #     for mdl in Model_list:
    #         print(mdl)
    #         exec('from tensorflow.keras.applications import '+mdl+' as Model')
        if type(imodel) is int:
            mdl = Model_list[imodel]
            exec('from tensorflow.keras.applications import '+mdl+' as Model', globals())
    #     Model = __import__('tensorflow.keras.applications.'+mdl)
        elif type(imodel) is str:
            exec('from tensorflow.keras.applications import '+imodel+' as Model', globals())
            mdl = imodel
        else:
            assert 0,'Model not recognized!'
        print(mdl)

        if shape[2]==3:
            baseModel = Model(weights="imagenet", include_top=False,
                                                       input_tensor=tf.keras.layers.Input(shape=shape))

        #     baseModel = tf.keras.applications.MobileNetV3Small(alpha=1.0, minimalistic=True,
        #                                                        weights="imagenet", include_top=False,
        #                                                        input_tensor=tf.keras.layers.Input(shape=shape))


            # show a summary of the base model
            print("[INFO] summary for base model...")
            #     print(baseModel.summary())
            inputs = baseModel.input
            headModel = baseModel.output
        else:
            inputs = layers.Input(shape=shape, name="img")
            baseModel = Model(weights="imagenet", include_top=False,
                                                       input_tensor=tf.keras.layers.Input(shape=(shape[0],shape[1],3)))

        #     baseModel = tf.keras.applications.MobileNetV3Small(alpha=1.0, minimalistic=True,
        #                                                        weights="imagenet", include_top=False,
        #                                                        input_tensor=tf.keras.layers.Input(shape=shape))


            # show a summary of the base model
            print("[INFO] summary for base model...")
            #     print(baseModel.summary())
            xh = layers.Conv2D(3, (1,1), activation=None)(inputs)
            headModel = baseModel(xh)
        
        headModel = layers.Conv2D(32, 3, activation="relu",padding='same')(headModel)
        headModel = tf.keras.layers.GlobalAveragePooling2D()(headModel)
    #     headModel = tf.keras.layers.Flatten(name="flatten")(headModel)
        encoded = tf.keras.layers.Dense(n_latent, activation="relu")(headModel)
        xl = tf.keras.layers.Dropout(0.5)(encoded)
        output = tf.keras.layers.Dense(n_class, activation="softmax")(xl)
        # place the head FC model on top of the base model (this will become
        # the actual model we will train)
        encoder = keras.models.Model(inputs=inputs, outputs=encoded)
        model = keras.models.Model(inputs=inputs, outputs=output,name=mdl)
    
    if comp:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-5,
                                                                     decay_steps=10,
                                                                     decay_rate=0.95)
        opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        #opt = tf.keras.optimizers.Adagrad(learning_rate=lr_schedule)
        #opt = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)
#         loss = tf.keras.losses.CategoricalCrossentropy()
        loss = tf.keras.losses.CategoricalCrossentropy()
    
        model.compile(loss=loss, optimizer=opt,metrics=["accuracy"])
    
#         model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['acc'])
    return model,encoder,mdl

def add_class(clf,drt,n_class = None,loss=None,optimizer='adam',metrics=["accuracy"],summary=False):
    
    if n_class is None:
        n_class = clf.layers[-1].output_shape[1]+1
    if loss is None:
        loss = keras.losses.CategoricalCrossentropy()
        
    dclass = n_class-clf.layers[-1].output_shape[1]
    w,b = clf.layers[-1].get_weights()

    inp = keras.Input(shape=clf.layers[0].input_shape[0][1:], name="input")
    latent = drt(inp)
    dop = layers.Dropout(clf.layers[-2].rate)(latent)
    out = layers.Dense(n_class, activation="softmax")(dop)
    # out = layers.Dense(n_class, activation="sigmoid")(dop)

    clf2 = keras.Model(inputs=inp, outputs=out, name="Classifier2")

    #     clf.summary()

    w_new = np.concatenate([w,clf2.layers[-1].weights[0].numpy()[:,-dclass:]],axis=-1)
    b_new = np.concatenate([b,clf2.layers[-1].weights[1].numpy()[-dclass:]],axis=-1)

    clf2.layers[-1].set_weights([w_new,b_new])

    clf2.compile(
    #     loss=keras.losses.BinaryCrossentropy(),
        loss=loss,
        optimizer=optimizer,
        metrics=metrics,
    )

    if summary:
        clf2.summary()
    return clf2