"""
File: dip_template.py -- Deep Image Prior Defense
Authors: Apurva Gandhi and Shomik Jain
Date: 2/02/2020

Additional References: 
- https://github.com/DmitryUlyanov/deep-image-prior/blob/master/denoising.ipynb (original implementation)
- https://arxiv.org/abs/1711.10925 (Deep Image Prior paper)
"""

# SCRIPT PARAMETERS

IN_COLAB = False

DIP_TYPE = 'JPEG'

MODEL_NAME = 'resnet_blur'

MODEL_PATH = 'models/'

MODEL_PATH = MODEL_PATH.format(MODEL_NAME)

SOFTMAX = True

# CATEGORY (i.e. Perturbed (FGSM) Whitebox Fake-Wrong)

DIR = 'data/resnet_blur_fgsm/fake/'
CORRECT_PRED = 1

IMGS_CODES = ['whitebox_resnet_blur_fgsm_fake_wrong/']

MAX_NUM_IMGS = 10

# Imports

from __future__ import print_function

import os
import numpy as np

import torch
import torch.optim

from skimage.measure import compare_psnr
from torchvision.utils import save_image
import h5py

import cv2
from torchvision import transforms

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

from PIL import Image
import pandas as pd
import scipy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Import Models

if IN_COLAB: 
  from google.colab import drive
  drive.mount('/content/gdrive/')

#   %cd gdrive/My\ Drive/Fuzzy\ Fakes/code
  from HHReLU import HHReLU 

#   %cd deep_image_prior

  from models import *
  from utils.denoising_utils import * 

#   %cd blur_fgsm_dip

else:
  from HHReLU import HHReLU 
  from models import *
  from utils.dip_utils import * 

DEFENSE_MODEL = torch.load(MODEL_PATH, map_location=device)
DEFENSE_MODEL.cuda()

# Helper function to load an image

def load_image(file):

    # Orig Img has shape (368, 300, 3)
    orig_img = cv2.imread(file)
    assert orig_img.shape == (368, 300, 3)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    assert orig_img.shape == (368, 300, 3)

    # Resize to (512, 512, 3)
    img = cv2.resize(orig_img, (512, 512))
    assert img.shape == (512, 512, 3)

    # Convert to Channel First Format
    img = np.transpose(img, (2,0, 1))
    assert img.shape == (3, 512, 512)

    # Add Dummy Layer for Tensor and rescale to [0, 1]
    img = np.array([img])/255.
    assert img.shape == (1, 3, 512, 512)
    return img

# Helper function to get prediction for image

def defense_pred(img_in, use_softmax=SOFTMAX):
    
    assert img_in.shape == (3, 512, 512)

    # Move Channel Last
    img = np.transpose(img_in, (1, 2, 0))
    assert img.shape == (512, 512, 3)

    # Resize to (368, 300, 3)
    img = cv2.resize(img, (300, 368))
    assert img.shape == (368, 300, 3)

    # Move Channel Back to First
    img = np.transpose(img, (2, 0, 1))
    assert img.shape == (3, 368, 300)   

    img = torch.tensor(img).type(torch.FloatTensor).unsqueeze(0)
    img = img.to(device)
    if not use_softmax:
      score = scipy.special.expit(DEFENSE_MODEL(img).squeeze().cpu().detach().numpy())
    else: 
      score = torch.nn.functional.softmax(DEFENSE_MODEL(img).squeeze(), dim=0).cpu().detach().numpy()[1]
    return score

IMGS = []
count = 0
for filename in os.listdir(DIR):
    if filename.endswith(".jpg"):
      img = load_image(DIR+filename)
      unperturbed_img = load_image('../../../data/test/' + DIR[-5:] + filename)
      pred =  defense_pred(img[0])
      unperturbed_pred = defense_pred(unperturbed_img[0])
      if(DIR[-5:] == 'fake/'): 
        if(unperturbed_pred > 0.5): 
          continue
      else:
        if(unperturbed_pred < 0.5): 
          continue
      if (CORRECT_PRED == 1 and pred >= 0.5) or (CORRECT_PRED == 0 and pred < 0.5):
        IMGS.append(DIR+filename)
        count += 1
        print(filename)
        if(count == MAX_NUM_IMGS):
          break

# Commented out IPython magic to ensure Python compatibility.
# Debugging function

# %matplotlib inline
def show_torch_img(img):
    plt.imshow(np.transpose(img[0], (1,2,0)), interpolation='nearest')

def save_image(data, out_file):
    data = (data * 255).astype(np.uint8)
    data = np.transpose(data, (1, 2, 0))
    
    im = Image.fromarray(data)
    #if im.mode != 'RGB':
    #    im = im.convert('RGB')
    im.save(out_file)

def closure():
    
    global i, out_avg, psnr_input_last, last_net, net_input
    
    if reg_noise_std > 0:
        net_input = net_input_saved + (noise.normal_() * reg_noise_std)
    
    out = net(net_input)
    out_np = torch_to_np(out)
    
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)
    out_avg_np = torch_to_np(out_avg)
    
    mse_input = mse(out, input_img_torch)
    mse_sm_input = mse(out_avg, input_img_torch)
    mse_input.backward()
    
    psnr_input = compare_psnr(input_img_np[0], out.detach().cpu().numpy()[0]) 
    psnr_sm_input = compare_psnr(input_img_np[0], out_avg.detach().cpu().numpy()[0]) 
    psnr_unperturbed = compare_psnr(out.detach().cpu().numpy()[0], unperturbed_img_np[0])
    
    # Prediction at Iteration
    pred = defense_pred(out_np)
    pred_sm = defense_pred(out_avg_np)

    print(i, pred, pred_sm, mse_input.item())

    if  PLOT and i % save_every == 0:
      plot_image_grid([np.clip(out_np, 0, 1), np.clip(out_avg_np, 0, 1)], factor=figsize, nrow=1)
      save_image(out_np, IMGS_CODES[0] + path[len(DIR):-4]+ '/' + 'it_'+str(i)+'.jpg')
      save_image(out_avg_np, IMGS_CODES[0] + path[len(DIR):-4]+ '/' + 'sm_'+str(i)+'.jpg')

    # Backtracking
    if i % show_every:
        if psnr_input - psnr_input_last < -5: 
            print('Falling back to previous checkpoint.')

            for new_param, net_param in zip(last_net, net.parameters()):
                net_param.data.copy_(new_param.cuda())

            return None, None, None
        else:
            last_net = [x.detach().cpu() for x in net.parameters()]
            psnr_input_last = psnr_input

    metrics = {}
    metrics['iteration'] = i
    metrics['mse_input'] = mse_input.item()
    metrics['mse_sm_input'] = mse_sm_input.item()
    metrics['psnr_input'] = psnr_input
    metrics['psnr_sm_input'] = psnr_sm_input
    metrics['psnr_unperturbed'] = psnr_unperturbed
    metrics['pred'] = pred
    metrics['pred_sm'] = pred_sm
                
    i += 1

    return out_np, out_avg_np, metrics

for img_index, path in enumerate(IMGS):
    input_img_np = load_image(path)
    unperturbed_img_np = load_image('../../../data/test/' + DIR[-5:] + path[len(DIR):])
    os.makedirs(IMGS_CODES[0] + path[len(DIR):-4]+ '/', exist_ok=True)
    print(defense_pred(input_img_np[0]))
    print(defense_pred(unperturbed_img_np[0]))
    INPUT = 'noise' # 'meshgrid'
    pad = 'reflection'
    OPT_OVER = 'net' # 'net,input'

    reg_noise_std = 1./30. # set to 1./20. for sigma=50
    LR = 0.01

    PLOT = True

    OPTIMIZER='adam' # 'LBFGS'
    show_every = 10
    exp_weight=0.99

    num_iter = 10000
    input_depth = 32 
    figsize = 4 

    if(DIP_TYPE == 'JPEG'): 
      net = skip(
                input_depth, 3, 
                num_channels_down = [8, 16, 32, 64, 128], 
                num_channels_up   = [8, 16, 32, 64, 128],
                num_channels_skip = [0, 0, 0, 4, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU').type(dtype)
    else:
      net = get_net(input_depth, 'skip', pad,
                    skip_n33d=128, 
                    skip_n33u=128, 
                    skip_n11=4, 
                    num_scales=5,
                    upsample_mode='bilinear').type(dtype)

    net_input = get_noise(input_depth, INPUT, (512, 512)).type(dtype).detach()

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    out_avg = None
    last_net = None
    psnr_input_last = 0
    i = 0

    input_img_torch = np_to_torch(input_img_np[0]).type(dtype)

    parameters = get_params(OPT_OVER, net, net_input)

    optimizer = torch.optim.Adam(parameters, lr=LR)

    save_every = 100

    answer = pd.DataFrame(columns=['iteration', 'mse_input', 'mse_sm_input', 'psnr_input', 'psnr_sm_input', 'psnr_unperturbed', 'pred', 'pred_sm'])

    for j in range(num_iter):
        optimizer.zero_grad()
        it_out, it_avg_out, metrics = closure()
        if metrics != None:
            answer = answer.append(metrics, ignore_index=True)
        optimizer.step()

    answer.to_csv(IMGS_CODES[0] + path[len(DIR):-4]+ '/' +'results.csv', index=False)

