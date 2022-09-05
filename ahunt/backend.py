
import os
from glob import glob
import pandas as pd
import streamlit as st
from datetime import datetime

def set_new_labels(session_state, default_labels='label1, label2, ...'):
    data_path = session_state.labeler_config['data_path']
    labels = st.text_input('Enter the labels and serapate them by comma.', default_labels)
    labels = labels.split(',')
    labels = [i.replace(' ','') for i in labels if i!='']
    if len(labels)==1:
        st.write('The number of lables have to be more than 1.')
    st.write('Selected labels are ',labels)

    images = get_images_list(data_path)
    images = [os.path.split(i)[1] for i in images]
    if data_path and len(labels)>1:
        if st.button('Approve config'):
            session_state.ishow = 0
            session_state.df = pd.DataFrame(index=images,columns=['label','predict','score','date-time'])
            session_state.df.index.name = 'path'
            # session_state.df.set_index('images', drop=True, append=False, inplace=False, verify_integrity=True)
            session_state.labeler_config['labels'] = labels
            os.mkdir(os.path.join(data_path,'als_files'))
            session_state.success = True
            st.experimental_rerun()

def get_images_list(data_path):
    images = glob(os.path.join(data_path,'*.jpg')) +\
            glob(os.path.join(data_path,'*.png'))
    images = sorted(images)
    return images

def imageshow_setstate(session_state):
    
    session_state.success = False
    data_path = st.text_input('Please Enter the path to the data directory.',
                              '/home/alireza/works/datasets/cat_dog')
    images = get_images_list(data_path)
    # images = images
    nimg = len(images)
    
    if data_path and nimg==0:
        st.write('No image is found in the given path.')
        return
    if not data_path: return
    st.write(f'{nimg} images found!')
        
    session_state.labeler_config = {'data_path' : data_path}
    session_state.labeler_config['trained_model'] = False
    
    csv_file = os.path.join(data_path,'als_files','labels.csv')
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, index_col='path')
        df.to_csv(os.path.join(data_path,'als_files','labels_backup.csv'))
        nlabeled = df['label'].dropna().shape[0]
        isresume = st.radio(
            f"Label history exists with {nlabeled} labeled images, do you want to resume?",
            ("New run" ,"Resume")
        )
         
        # df.set_index('images', drop=True, append=False, inplace=False, verify_integrity=True)
        old_labels = df['label'].dropna().unique().tolist()
        if isresume=='Resume':
            st.write('Are you sure?')
            if st.button('Yes'):
                session_state.df = df
                nlabeled = df['label'].dropna().shape[0]
                if not df['predict'].isnull().values.any():
                    session_state.labeler_config['trained_model'] = True
                session_state.ishow = nlabeled
                session_state.labeler_config['labels'] = old_labels
                session_state.als_config = None
                session_state.success = True
                st.experimental_rerun()
            
        else:
            set_new_labels(session_state, default_labels=', '.join(old_labels))
    else:
        set_new_labels(session_state)

    if False:
        from ssaip.ctorch import ALServiceTorch
        session_state.als = ALServiceTorch(root_dir = data_path,
                                        csv_file = None,
                                        session = session_state,
                                        st = None)
    else:
        from ssaip.ctflow import ALServiceTFlow
        session_state.als_config = {'batch_size':32,'autotrain':False,'model_name':'VGG19'}
        session_state.als = ALServiceTFlow(root_dir = data_path,
                                           csv_file = None,
                                           als_config = session_state.als_config,
                                           session = session_state,
                                           st = None)
    # imageshow(session_state)

def imageshow(session_state):
    data_path = session_state.labeler_config['data_path']
    images = session_state.df.index.to_list()
    col1, col2 = st.columns([.7,1.])
    with col2:
        
        predic = None
        session_state.index = 0
        if session_state.labeler_config['trained_model']:
            image = images[session_state.ishow]
            # session_state.df.loc[session_state.ishow,'images'] = image
            predic = session_state.df.loc[image,'predict']
            session_state.index = session_state.labeler_config['labels'].index(predic)
        
        with st.form("labeler"):
            if predic: st.write('Prediction: '+predic)#,session_state.index)
            label = st.selectbox(
                label = "Select the label?",
                options = session_state.labeler_config['labels'],
                index = session_state.index,
                # key = '10001'
            )
            # if predic and label!=predic:
            #     label = st.selectbox(
            #         label = "Select the label?",
            #         options = session_state.labeler_config['labels'],
            #         index = session_state.index,
            #         key = '10001'
            #     )
                # st.experimental_rerun()
            # st.write(session_state.index,label,session_state.labeler_config['labels'])

            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit")
            if submitted:
                image = images[session_state.ishow]
                # session_state.df.loc[session_state.ishow,'images'] = MODEL_LISTimage
                session_state.df.loc[image,'label'] = label
                session_state.df.loc[image,'date-time'] = \
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                session_state.df.to_csv(os.path.join(data_path,'als_files','labels.csv'))
                session_state.ishow = session_state.ishow+1
                
        with st.form("newclass"):
            new_class = st.text_input('Add a new class if you need!.', 'new class')
            submitted = st.form_submit_button("Submit")
            if submitted and new_class!='new class' and new_class not in session_state.labeler_config['labels']:
                session_state.labeler_config['labels'] += [new_class]
                image = images[session_state.ishow]
                # session_state.df.loc[image,'path'] = image
                session_state.df.loc[image,'label'] = new_class
                session_state.df.loc[image,'date-time'] = \
                    datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                session_state.df.to_csv(os.path.join(data_path,'als_files','labels.csv'),index=0)
                session_state.ishow = session_state.ishow+1
                st.experimental_rerun()
    with col1:
        st.write(session_state.ishow,images[session_state.ishow])
        image = os.path.join(data_path,images[session_state.ishow])
        # print(image)
        # results_raw = class_labeler(image)
        st.image(image)


MODEL_LIST = ['DenseNet121', 'DenseNet169', 'DenseNet201', 'EfficientNetB0', 'EfficientNetB1',
              'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5',
              'EfficientNetB6', 'EfficientNetB7', 'EfficientNetV2B0', 'EfficientNetV2B1',
              'EfficientNetV2B2', 'EfficientNetV2B3', 'EfficientNetV2L', 'EfficientNetV2M',
              'EfficientNetV2S', 'InceptionResNetV2', 'InceptionV3', 'MobileNet', 'MobileNetV2',
              'MobileNetV3Large', 'MobileNetV3Small', 'NASNetLarge', 'NASNetMobile', 'RegNetX002',
              'RegNetX004', 'RegNetX006', 'RegNetX008', 'RegNetX016', 'RegNetX032', 'RegNetX040',
              'RegNetX064', 'RegNetX080', 'RegNetX120', 'RegNetX160', 'RegNetX320', 'RegNetY002',
              'RegNetY004', 'RegNetY006', 'RegNetY008', 'RegNetY016', 'RegNetY032', 'RegNetY040',
              'RegNetY064', 'RegNetY080', 'RegNetY120', 'RegNetY160', 'RegNetY320', 'ResNet101',
              'ResNet101V2', 'ResNet152', 'ResNet152V2', 'ResNet50', 'ResNet50V2', 'ResNetRS101',
              'ResNetRS152', 'ResNetRS200', 'ResNetRS270', 'ResNetRS350', 'ResNetRS420',
              'ResNetRS50', 'VGG16', 'VGG19', 'Xception']

def imageshow_label(session_state):
    if not session_state.als_config:
        session_state.als_config = {'batch_size':32,'autotrain':False,'model_name':'VGG19'}
    if st.sidebar.button('AL-service'):
        session_state.als.train()
    ahunt_mod = st.sidebar.checkbox("Preferences", value=False)
    if ahunt_mod:
        batch_size = st.sidebar.number_input('batch size', value=32)
        autotrain = st.sidebar.checkbox("Autotrain", value=False)
        model_name = st.sidebar.selectbox(
            label = "Select the label?",
            options = MODEL_LIST,
            index = 0,
        )
        session_state.als_config['model_name'] = model_name
        session_state.als_config['batch_size'] = batch_size
        session_state.als_config['autotrain'] = autotrain
        
        if st.sidebar.button('apply'):
            from ssaip.ctflow import ALServiceTFlow
            session_state.als = ALServiceTFlow(root_dir = session_state.labeler_config['data_path'],
                                               csv_file = None,
                                               als_config = session_state.als_config,
                                               session = session_state,
                                               st = None)
            st.sidebar.write('Changes applied!')    
        
        

    nimg = session_state.df.shape[0]
    if session_state.ishow<nimg:
        
        # if session_state.new_class!='class name':
        #     st.write(session_state.new_class)
            
        #     st.write(session_state.labeler_config['labels'])
        
    
        col1, col2, col3, col4 = st.columns([.5,1.,1.5,2.0])
        with col1:
            if st.button('Undo'):
                session_state.ishow = session_state.ishow-1
        with col2:
            if st.button('Next'):
                session_state.ishow = session_state.ishow+1
        with col3:
            st.write('')
            # if st.button('Train'):    session_state.ahunt.train()
        with col4:
            st.write('')

        progress_bar = st.progress(0)
        status_text = st.empty()
        progress_bar.progress(session_state.ishow/nimg)
        status_text.text('{:4.2f}% complete.'.format(100*session_state.ishow/nimg))
        imageshow(session_state)

    else:
        st.write('All are done!')
        del session_state.ishow
        del session_state.df