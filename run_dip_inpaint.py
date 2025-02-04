# Import libs
from __future__ import print_function
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.optim

from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip
from utils.inpainting_utils import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

PLOT = True
imsize = -1
dim_div_by = 64

def main():
    # Choose figure
    img_path  = 'data/inpainting/kate.png'
    mask_path = 'data/inpainting/kate_mask.png'

    NET_TYPE = 'skip_depth6'  # one of skip_depth4|skip_depth2|UNET|ResNet

    # Load mask
    img_pil, img_np = get_image(img_path, imsize)
    img_mask_pil, img_mask_np = get_image(mask_path, imsize)

    # Center crop
    img_mask_pil = crop_image(img_mask_pil, dim_div_by)
    img_pil = crop_image(img_pil, dim_div_by)

    img_np = pil_to_np(img_pil)
    img_mask_np = pil_to_np(img_mask_pil)

    # Visualize
    img_mask_var = np_to_torch(img_mask_np).type(dtype)

    plot_image_grid([img_np, img_mask_np, img_mask_np * img_np], 3, 11)

    # Setup
    pad = 'reflection'  # 'zero'
    OPT_OVER = 'net'
    OPTIMIZER = 'adam'  # Can switch to 'adamw' or 'rmsprop'
    LR = 0.005  # Reduced learning rate for smoother convergence
    reg_noise_std = 0.01  # Reduced noise regularization

    # Define parameters based on image types
    if 'kate.png' in img_path or 'peppers.png' in img_path:
        INPUT = 'noise'
        input_depth = 32
        num_iter = 15001  # Reduced number of iterations with early stopping
        param_noise = False
        show_every = 10
        figsize = 5

        net = skip(input_depth, img_np.shape[0],
                   num_channels_down=[128] * 6,  # Increased depth
                   num_channels_up=[128] * 6,    # Increased depth
                   num_channels_skip=[128] * 6,  # Increased depth
                   filter_size_up=3, filter_size_down=3,
                   upsample_mode='nearest', filter_skip_size=1,
                   need_sigmoid=True, need_bias=True, pad=pad,
                   act_fun='LeakyReLU').type(dtype)  # Removed need_batchnorm
    
    else:
        assert False

    net = net.type(dtype)
    net_input = get_noise(input_depth, INPUT, img_np.shape[1:]).type(dtype)

    # Compute number of parameters
    s = sum(np.prod(list(p.size())) for p in net.parameters())
    print('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().type(dtype)

    img_var = np_to_torch(img_np).type(dtype)
    mask_var = np_to_torch(img_mask_np).type(dtype)

    # Main loop with early stopping
    i = 0
    best_loss = float('inf')
    patience = 500  # Early stopping patience
    early_stop_counter = 0

    def closure():
        nonlocal i, best_loss, early_stop_counter
        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        out = net(net_input)

        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()

        print('Iteration %05d    Loss %f' % (i, total_loss.item()), '\r', end='')

        # Early stopping
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter > patience:
                print(f"\nEarly stopping at iteration {i}")
                return total_loss

        # Plot intermediate results
        if PLOT and i % show_every == 0:
            out_np = torch_to_np(out)
            plot_image_grid([np.clip(out_np, 0, 1)], factor=figsize, nrow=1)

        i += 1
        return total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    p = get_params(OPT_OVER, net, net_input)
    optimize(OPTIMIZER, p, closure, LR, num_iter)

    out_np = torch_to_np(net(net_input))
    plot_image_grid([out_np], factor=5)

if __name__ == "__main__":
    main()
