# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
"""

import numpy as np
import cv2
import os
from scipy import signal
from PIL import Image
import torch
from methods.Bicubic import Bicubic
from methods.Brovey import Brovey
from methods.PCA import PCA
from methods.IHS import IHS
from methods.SFIM import SFIM
from methods.GS import GS
from methods.Wavelet import Wavelet
from methods.MTF_GLP import MTF_GLP
from methods.MTF_GLP_HPM import MTF_GLP_HPM
from methods.GSA import GSA
from methods.CNMF import CNMF
from methods.GFPCA import GFPCA

from metrics import ref_evaluate, no_ref_evaluate

'''loading data'''

ms_path = r'E:/data/test/wv/lrms/151.tif'
pan_path = r'E:/data/test/wv/pan/151.tif'
gt_path = r'E:/data/test/wv/hrms/151.tif'
save_dir= r'E:/data/test/wv/wv-result/'
pnn_path = r'E:/data/test/wv/wv-result/PNN.tif'
rscnnca_path = r'E:/data/test/wv/wv-result/RSCNNCA.tif'

'''setting save parameters'''
save_images = True
save_channels = [0, 1, 2] # BGR-NIR for GF2

if save_images and (not os.path.isdir(save_dir)):
    os.makedirs(save_dir)

def save_img(img, img_name, mode):
    img = torch.tensor(img)
    save_img = img.squeeze().clamp(0, 1).numpy()
    # save img
    save_fn = save_dir + '/' + img_name
    save_img = np.uint8(save_img * 255).astype('uint8')
    save_img = Image.fromarray(save_img, mode)
    save_img.save(save_fn)

original_ms = np.array(Image.open(ms_path))
original_pan = np.expand_dims(np.array(Image.open(pan_path), dtype=np.float32), -1)
original_gt = np.array(Image.open(gt_path))

print('original ms', original_ms.shape)
print('original pan', original_pan.shape)
print('original gt', original_gt.shape)


'''normalization'''
used_ms = original_ms / 255.
used_pan = original_pan / 255.
gt = original_gt / 255.

# max_patch, min_patch = np.max(original_ms, axis=(0,1)), np.min(original_ms, axis=(0,1))
# original_msi = np.float32(original_ms-min_patch) / (max_patch - min_patch)
#
# max_patch, min_patch = np.max(original_pan, axis=(0,1)), np.min(original_pan, axis=(0,1))
# original_pan = np.float32(original_pan-min_patch) / (max_patch - min_patch)
# original_pan = np.expand_dims(original_pan, -1)
#
# max_patch, min_patch = np.max(original_gt, axis=(0,1)), np.min(original_gt, axis=(0,1))
# original_gt = np.float32(original_gt-min_patch) / (max_patch - min_patch)

gt = np.uint8(gt*255)

# used_ms = original_msi
# used_pan = original_pan




print('lrms shape: ', used_ms.shape, 'pan shape: ', used_pan.shape, 'gt', gt.shape)

'''evaluating all methods'''
ref_results={}
ref_results.update({'metrics: ':'  PSNR,     SSIM,   SAM,    ERGAS,  SCC,    Q'})
no_ref_results={}
no_ref_results.update({'metrics: ':'  D_lamda, D_s,    QNR'})

'''Bicubic method'''
print('evaluating Bicubic method')
fused_image = Bicubic(used_pan, used_ms)
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255).astype('uint8'), np.uint8(used_ms*255).astype('uint8'))
ref_results.update({'Bicubic    ':temp_ref_results})
no_ref_results.update({'Bicubic    ':temp_no_ref_results})
print('Bicubic ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'bicubic.tif', mode='CMYK')

'''Brovey method'''
print('evaluating Brovey method')
fused_image = Brovey(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255).astype('uint8'), np.uint8(used_ms*255).astype('uint8'))
ref_results.update({'Brovey     ':temp_ref_results})
no_ref_results.update({'Brovey     ':temp_no_ref_results})
print('Brovey ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'Brovey.tif', mode='CMYK')


    
'''PCA method'''
print('evaluating PCA method')
fused_image = PCA(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255).astype('uint8'), np.uint8(used_ms*255).astype('uint8'))
ref_results.update({'PCA        ':temp_ref_results})
no_ref_results.update({'PCA        ':temp_no_ref_results})
print('PCA ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'PCA.tif', mode='CMYK')



'''IHS method'''
print('evaluating IHS method')
fused_image = IHS(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255).astype('uint8'), np.uint8(used_ms*255).astype('uint8'))
ref_results.update({'IHS        ':temp_ref_results})
no_ref_results.update({'IHS        ':temp_no_ref_results})
print('IHS ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'IHS.tif', mode='CMYK')



'''SFIM method'''
print('evaluating SFIM method')
fused_image = SFIM(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255).astype('uint8'), np.uint8(used_ms*255).astype('uint8'))
ref_results.update({'SFIM       ':temp_ref_results})
no_ref_results.update({'SFIM       ':temp_no_ref_results})
print('SFIM ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'SFIM.tif', mode='CMYK')



'''GS method'''
print('evaluating GS method')
fused_image = GS(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'GS         ':temp_ref_results})
no_ref_results.update({'GS         ':temp_no_ref_results})
print('GS ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'GS.tif', mode='CMYK')

'''Wavelet method'''
print('evaluating Wavelet method')
fused_image = Wavelet(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'Wavelet    ':temp_ref_results})
no_ref_results.update({'Wavelet    ':temp_no_ref_results})
print('Wavelet ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'Wavelet.tif', mode='CMYK')


'''MTF_GLP method'''
print('evaluating MTF_GLP method')
fused_image = MTF_GLP(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'MTF_GLP    ':temp_ref_results})
no_ref_results.update({'MTF_GLP    ':temp_no_ref_results})
print('MTF_GLP ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'MTF_GLP.tif', mode='CMYK')



'''MTF_GLP_HPM method'''
print('evaluating MTF_GLP_HPM method')
fused_image = MTF_GLP_HPM(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'MTF_GLP_HPM':temp_ref_results})
no_ref_results.update({'MTF_GLP_HPM':temp_no_ref_results})

print('MTF_GLP_HPM ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'MTF_GLP_HPM.tif', mode='CMYK')



'''GSA method'''
print('evaluating GSA method')
fused_image = GSA(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'GSA        ':temp_ref_results})
no_ref_results.update({'GSA        ':temp_no_ref_results})

print('GSA ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'GSA.tif', mode='CMYK')


'''CNMF method'''
print('evaluating CNMF method')
fused_image = CNMF(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'CNMF       ':temp_ref_results})
no_ref_results.update({'CNMF       ':temp_no_ref_results})

print('CNMF ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'CNMF.tif', mode='CMYK')


'''GFPCA method'''
print('evaluating GFPCA method')
fused_image = GFPCA(used_pan[:, :, :], used_ms[:, :, :])
fused_image_uint8 = np.uint8(fused_image*255)
temp_ref_results = ref_evaluate(fused_image_uint8, gt)
temp_no_ref_results = no_ref_evaluate(fused_image_uint8, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'GFPCA      ':temp_ref_results})
no_ref_results.update({'GFPCA      ':temp_no_ref_results})

print('GFPCA ', fused_image.shape, fused_image.max(), fused_image.min())
#save
if save_images:
    save_img(fused_image, 'GFPCA.tif', mode='CMYK')

'''PNN method'''
print('evaluating PNN method')
fused_image = np.array(Image.open(pnn_path))
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'PNN        ':temp_ref_results})
no_ref_results.update({'PNN        ':temp_no_ref_results})


'''RSCNNCA method'''
print('evaluating RSCNNCA method')
fused_image = np.array(Image.open(rscnnca_path))
temp_ref_results = ref_evaluate(fused_image, gt)
temp_no_ref_results = no_ref_evaluate(fused_image, np.uint8(used_pan*255), np.uint8(used_ms*255))
ref_results.update({'RSCNNCA     ':temp_ref_results})
no_ref_results.update({'RSCNNCA     ':temp_no_ref_results})


filename = 'eval-wv.txt'
with open(filename, 'w') as f:
    ''''print result'''
    print('################## reference comparision #######################')
    for index, i in enumerate(ref_results):  # i=key
        if index == 0:
            print(i, ref_results[i])
            f.write(i + ' ' + ref_results[i] + ' ')
        else:
            print(i, [round(j, 4) for j in ref_results[i]])
            x = 0
            for j in no_ref_results[i]:
                if x == 0:
                    f.write(i + ' ' + str(round(j, 4)))
                    x += 1
                else:
                    f.write(str(round(j, 4)) + ' ')
        f.write('\n')
    print('################## reference comparision #######################')

    print()
    print()
    print()

    print('################## no reference comparision ####################')
    for index, i in enumerate(no_ref_results):
        if index == 0:
            print(i, no_ref_results[i])
            f.write(i + ' ' + no_ref_results[i] + ' ')
        else:
            print(i, [round(j, 4) for j in no_ref_results[i]])
            x = 0
            for j in no_ref_results[i]:
                if x == 0:
                    f.write(i + ' ' + str(round(j, 4)))
                    x += 1
                else:
                    f.write(str(round(j, 4)) + ' ')
        f.write('\n')
    print('################## no reference comparision ####################')

    # print(ref_results)
    # print(no_ref_results)

print('write done...')




