import streamlit as st
import argparse
from PIL import Image


import numpy as np
import random
import time
from tqdm import tqdm

import torch
import torchvision.utils as vutils

import os, sys
import os.path as osp

if sys.version_info[0] == 2:    ##python version
    import cPickle as pickle
else:
    import pickle

ROOT_PATH = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)),  ".."))
sys.path.insert(0, ROOT_PATH)

cwd = os.getcwd()
directory = os.path.join(cwd, 'code/lib')
sys.path.append(cwd)

from code.lib.utils import mkdir_p, get_rank, merge_args_yaml, get_time_stamp, load_netG
from code.lib.utils import truncated_noise, prepare_sample_data
from code.lib.perpare import prepare_models


###########  GEN  #############

def get_tokenizer():
    from nltk.tokenize import RegexpTokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    return tokenizer

@st.cache
def tokenize(wordtoix, sentences):
    '''generate images from example sentences'''
    tokenizer = get_tokenizer()
   # a list of indices for a sentence
    captions = []
    cap_lens = []
    new_sent = []
    for sent in sentences:
        if len(sent) == 0:
            continue
        sent = sent.replace("\ufffd\ufffd", " ")
        tokens = tokenizer.tokenize(sent.lower())
        if len(tokens) == 0:
            print('sent', sent)
            continue
        rev = []
        for t in tokens:
            t = t.encode('ascii', 'ignore').decode('ascii')
            if len(t) > 0 and t in wordtoix:
                rev.append(wordtoix[t])
        captions.append(rev)
        cap_lens.append(len(rev))
        new_sent.append(sent)    
    return captions, cap_lens, new_sent

@st.cache
def sample_example(wordtoix, netG, text_encoder, args):
    batch_size, device = args.imgs_per_sent, args.device
    text_filepath, img_save_path = args.example_captions, args.samples_save_dir
    truncation, trunc_rate = args.truncation, args.trunc_rate
    z_dim = args.z_dim
    captions, cap_lens, _ = tokenize(wordtoix, text_filepath)
    # print(captions)
    # print("*******\n")
    # print(cap_lens)
    # print(device)
    sent_embs, _  = prepare_sample_data(captions, cap_lens, text_encoder, device)
    
    caption_num = sent_embs.size(0)
    # print(type(caption_num))

    # get noise
    if truncation==True:
        noise = truncated_noise(batch_size, z_dim, trunc_rate)
        noise = torch.tensor(noise, dtype=torch.float).to(device)
    else:
        noise = torch.randn(batch_size, z_dim).to(device)
    # sampling

    with torch.no_grad():
        fakes = []        
        for i in tqdm(range(caption_num)):
            sent_emb = sent_embs[i].unsqueeze(0).repeat(batch_size, 1)
            fakes = netG(noise, sent_emb)
            img_name = osp.join(img_save_path,'Sent%03d.png'%(i+1))
            vutils.save_image(fakes.data, img_name, nrow=4, range=(-1, 1), normalize=True)
            torch.cuda.empty_cache()

@st.cache
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='DF-GAN')
    parser.add_argument('--cfg', dest='cfg_file', type=str, default='./code/cfg/bird.yml',
                        help='optional config file')
    parser.add_argument('--imgs_per_sent', type=int, default=1,
                        help='the number of images per sentence')
    parser.add_argument('--imsize', type=int, default=256,
                        help='image szie')
    parser.add_argument('--cuda', type=bool, default=False,
                        help='if use GPU')
    parser.add_argument('--train', type=bool, default=False,
                        help='if training')
    parser.add_argument('--multi_gpus', type=bool, default=False,
                        help='if use multi-gpu')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='gpu id')
    parser.add_argument('--local_rank', default=-1, type=int,
        help='node rank for distributed training')
    parser.add_argument('--random_sample', action='store_true',default=True, 
        help='whether to sample the dataset with random sampler')
    args = parser.parse_args()
    return args

@st.cache
def build_word_dict(pickle_path):
    with open(pickle_path, 'rb') as f:
        x = pickle.load(f)
        wordtoix = x[3]
        del x
        n_words = len(wordtoix)
        print('Load from: ', pickle_path)
    return n_words, wordtoix



def main(args):
    st.title("Text to Image DFGAN Demo")
    st.write('\n\n')

    # selected_model = st.selectbox('Select The Model',("CUB Birds","COCO" ))

    selected_caption = st.selectbox( 'Select The Caption', 
    ('this bird has an orange bill, a white belly and white eyebrows.',
    'this bird had brown primaries, a brown crown, and white belly.',
    'this is a grey bodied bird with light grey wings and a white breast.'
    ))


    caption = st.text_input("Enter The Caption", selected_caption)
    n_copies = st.slider('Number of Generated Images', min_value=1, max_value=12, value=6, step=1)

    if st.button('Generate Image'):
        my_bar = st.progress(0)
        placeholder = st.empty()
        placeholder.info('Loading Model', icon="ℹ️")

        time_stamp = get_time_stamp()
        args.example_captions = list(caption.split('\n'))
        # print(caption)
        args.imgs_per_sent = n_copies
        args.samples_save_dir = osp.join(args.samples_save_dir, time_stamp)
        
        # if selected_model == "COCO":
        #     args.cfg_file = './cfg/coco.yml'
        #     print("d"*20)
        #     parser = argparse.ArgumentParser(description='DF-GAN')
        #     args = parser.parse_args()

            
        # else:
        #     args.cfg_file = './cfg/bird.yml'
        # print(args)
        

        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            mkdir_p(args.samples_save_dir) 
        # prepare data

        for percent_complete in range(15,30):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        # print(os.getcwd())
        pickle_path = os.path.join(args.data_dir, 'captions_DAMSM.pickle')
        
        args.vocab_size, wordtoix = build_word_dict(pickle_path)
        # prepare models
        
        
        _, text_encoder, netG, _, _ = prepare_models(args)
        
        # model_path = osp.join(ROOT_PATH, args.checkpoint)
        model_path = args.checkpoint
        
        netG = load_netG(netG, model_path, args.multi_gpus, train=False)
        netG.eval()

        
        placeholder.info('Providing Input', icon="ℹ️")
        for percent_complete in range(40,60):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('Load %s for NetG'%(args.checkpoint))
            print("************ Start sampling ************")
        
        
        start_t = time.time()
        sample_example(wordtoix, netG, text_encoder, args)
        end_t = time.time()
        
        if (args.multi_gpus==True) and (get_rank() != 0):
            None
        else:
            print('*'*40)
            print('Sampling done, %.2fs cost, saved to %s'%(end_t-start_t, args.samples_save_dir))
            print('*'*40)

        
        placeholder.info('Generating Images', icon="ℹ️")
        for percent_complete in range(70,100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)

        placeholder.empty()
        my_bar.empty()
        


        st.write("#### Output Image")
        image = Image.open(args.samples_save_dir+"/sent001.png")
        st.image(image, width = 700)



        



if __name__ == "__main__":

    st.markdown("CUB Birds")
    st.sidebar.markdown("CUB Birds")
    args = merge_args_yaml(parse_args())
   # print(args)
    
    # set seed    
    if args.manual_seed is None:
        args.manual_seed = 100
    random.seed(args.manual_seed)
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if args.cuda:
        if args.multi_gpus:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.distributed.init_process_group(backend="nccl")
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            args.device = torch.device("cuda", local_rank)
            args.local_rank = local_rank
        else:
            torch.cuda.manual_seed_all(args.manual_seed)
            torch.cuda.set_device(args.gpu_id)
            args.device = torch.device("cuda")
    else:
        args.device = torch.device('cpu')
    main(args)