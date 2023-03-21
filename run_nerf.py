import os, sys
from datetime import datetime
import numpy as np
import imageio
import json
import pdb
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm, trange
import pickle

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from optimizer import MultiOptimizer
from radam import RAdam
from loss import sigma_sparsity_loss, total_variation_loss

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_scannet import load_scannet_data
from load_LINEMOD import load_LINEMOD_data
from neaf_operations import load_neaf_data, build_neaf_batch
from ir_visualization import save_ir

import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, times, fn, embed_fn, embeddirs_fn, embedtimes_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded, keep_mask = embed_fn(inputs_flat)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    if times is not None:
        input_times_flat = torch.reshape(times, [-1, 1])
        embedded_times = embedtimes_fn(input_times_flat)
        embedded = torch.cat([embedded, embedded_times], -1)
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs_flat[~keep_mask, -1] = 0 # set sigma to 0 for invalid points
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(chunk=1024*32, rays=None,
                  near=0., far=1.,
                  use_viewdirs=False, times=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """

    # use provided ray batch
    rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape  # [..., 3]

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)
    if times is not None:
        rays = torch.cat([rays, times[:, None]], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'depth_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_recs(recs, times, chunk, render_kwargs):
    rgb, _, _, _ = render(rays=recs, times=times, chunk=chunk, **render_kwargs)
    return rgb


def render_ir(recs, timesteps, i, chunk, render_kwargs):
    # need every rec at every timestep
    rec_count = recs.shape[1]
    time_vals = torch.linspace(0., 1., steps=timesteps)
    time_vals = time_vals.repeat((rec_count))  # get all timesteps for every rec
    recs_repeated = torch.repeat_interleave(recs, timesteps, dim=1)  # get every rec timestep-times
    irs = render_recs(recs_repeated, time_vals, chunk, render_kwargs)
    irs = irs.reshape((rec_count, timesteps, 3))
    irs = irs.cpu().numpy()
    return irs


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    input_ch_time = 0
    embedtimes_fn = None
    if args.use_time:
        # encode time input
        if args.i_embed_time==0:
            print("Positional encoding for time input selected")
        elif args.i_embed_time==1:
            print(f"Hash encoding for time input selected")
        embedtimes_fn, input_ch_time = get_embedder(args.multires_time, args, input_dims=1, i=args.i_embed_time)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    if args.i_embed==1:
        model = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=4,
                        hidden_dim_color=64,
                        input_ch=input_ch,
                        input_ch_views=input_ch_views,
                        input_ch_time=input_ch_time).to(device)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=4,
                        hidden_dim_color=64,
                        input_ch=input_ch,
                        input_ch_views=input_ch_views,
                        input_ch_time=input_ch_time).to(device)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, times, network_fn: run_network(inputs, viewdirs, times, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                embedtimes_fn=embedtimes_fn,
                                                                netchunk=args.netchunk)


    # Create optimizer
    if args.i_embed==1:
        optimizer = RAdam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################
    # pdb.set_trace()

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False, neaf_mode=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)

    blocking_alpha = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * blocking_alpha

    if not neaf_mode:
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
        acc_map = torch.sum(weights, -1)
    else:
        falloffs = torch.minimum(torch.ones_like(dists), 1 / dists)
        neaf_weights = alpha * blocking_alpha * falloffs
        rgb_map = torch.sum(neaf_weights[..., None] * rgb, -2)  # [N_rays, 3]
        depth_map = torch.sum(weights * z_vals, -1) / torch.sum(weights, -1)
        disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map)
        acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    # Calculate weights sparsity loss
    try:
        entropy = Categorical(probs = torch.cat([weights, 1.0-weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
    except:
        pdb.set_trace()
    sparsity_loss = entropy

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                embed_fn=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False,
                timeinterval=0.1):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,8:11] if ray_batch.shape[-1] > 8 else None
    times = ray_batch[:, -1] if ray_batch.shape[-1] > 11 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]
    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))
    z_vals = z_vals.expand([N_rays, N_samples])

    # original for dom
    # pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    # get offsets so we can calculate offset times
    rays_offsets = rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]

    pt_times = None  # [N_rays, N_samples, 1]
    if times is not None:
        sos = 343.
        offset_lengths = torch.linalg.norm(rays_offsets, dim=2)
        offset_times = offset_lengths / sos * (1 / timeinterval)
        pt_times = times[..., None] - offset_times

    pts = rays_o[..., None, :] + rays_offsets  # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, pt_times, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                white_bkgd, pytest=pytest,
                                                                                neaf_mode=times is not None)

    if N_importance > 0:
        rgb_map_0, depth_map_0, acc_map_0, sparsity_loss_0 = rgb_map, depth_map, acc_map, sparsity_loss

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        rays_offsets = rays_d[...,None,:] * z_vals[...,:,None]  # [N_rays, N_samples + N_importance, 3]

        pt_times = None
        if times is not None:
            sos = 343.
            offset_lengths = torch.linalg.norm(rays_offsets, dim=2)
            offset_times = offset_lengths / sos * (1 / timeinterval)
            pt_times = times[..., None] - offset_times

        pts = rays_o[...,None,:] + rays_offsets  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, pt_times, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std,
                                                                                    white_bkgd, pytest=pytest,
                                                                                    neaf_mode=times is not None)

    ret = {'rgb_map' : rgb_map, 'depth_map' : depth_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['depth0'] = depth_map_0
        ret['acc0'] = acc_map_0
        ret['sparsity_loss0'] = sparsity_loss_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret

# access the model without going 5 functions deep
def simple_model_query(pts, viewdirs, pt_times, network_query_fn, model):
    raw = network_query_fn(pts, viewdirs, pt_times, model)
    return raw


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--use_time", action='store_true',
                        help='create model with time input')
    parser.add_argument("--i_embed", type=int, default=1,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--i_embed_views", type=int, default=2,
                        help='set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels / rays')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## scannet flags
    parser.add_argument("--scannet_sceneID", type=str, default='scene0000_00',
                        help='sceneID to load from scannet')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    ## neaf flags
    parser.add_argument("--time_interval", type=float, default=0.1,
                        help='time interval (in s) represented by range [0, 1]')
    parser.add_argument("--multires_time", type=int, default=4,
                        help='log2 of max freq for time encoding (1D timestep)')
    parser.add_argument("--i_embed_time", type=int, default=0,
                        help="encoding for time input. 0=positional, 1=hash")
    parser.add_argument("--speed_of_sound", type=float, default=343.,
                        help='speed of sound used for neaf')
    parser.add_argument("--neaf_timesteps", type=int, default=100,
                        help='time discretion for neaf')
    parser.add_argument("--neaf_raydata", type=str, default="rays.json",
                        help="ray file for neaf gt")
    parser.add_argument("--angle_exp", type=float, default=10,
                        help="exponent applied to dot product when calculating receivers")
    parser.add_argument("--neaf_permtest_cnt", type=int, default=100,
                        help="number of permanent test listeners whose irs are visualized")

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=1000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=5000,
                        help='frequency of render_poses video saving')

    parser.add_argument("--finest_res",   type=int, default=512,
                        help='finest resolultion for hashed embedding')
    parser.add_argument("--log2_hashmap_size",   type=int, default=19,
                        help='log2 of hashmap size')
    parser.add_argument("--sparse-loss-weight", type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument("--tv-loss-weight", type=float, default=1e-6,
                        help='learning rate')

    parser.add_argument("--use_wandb", action='store_true',
                        help='upload to wandb')
    parser.add_argument("--wandb_project_name", type=str, default="neaf_optimization",
                        help='project name for wandb')
    parser.add_argument("--bootstrap_only", action='store_true',
                        help='kill process before training iteration')

    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Load data
    if args.dataset_type == 'neaf':
        print('loading precomputed rays from json')
        listener_states, i_split, source_pos, bounding_box = load_neaf_data(basedir=args.datadir, ray_file=args.neaf_raydata)
        i_train, i_test, i_val = i_split
        args.bounding_box = bounding_box
        near = 0.01
        far = 6.
    else:
        print('Unknown dataset type', args.dataset_type, 'exiting')
        return

    # Create log dir and copy the config file
    basedir = args.basedir
    if args.i_embed==1:
        args.expname += "_hashXYZ"
    elif args.i_embed==0:
        args.expname += "_posXYZ"
    if args.i_embed_views==2:
        args.expname += "_sphereVIEW"
    elif args.i_embed_views==0:
        args.expname += "_posVIEW"
    if args.i_embed_time==0:
        args.expname += "_posTIME"
    elif args.i_embed_time==1:
        args.expname += "_hashTIME"
    args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
    args.expname += "_RAdam"

    args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')
    expname = args.expname
    print(f"This experiment is named: {expname}")

    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # wandb init
    if args.use_wandb:
        config = {
            "learning_rate": args.lrate,
            "batch_size": args.N_rand,
            "timesteps": args.neaf_timesteps,
            "time_encoding": args.multires_time,
            "time_interval": args.time_interval,
            "angle_exponent": args.angle_exp,
            "data_file": args.neaf_raydata,
        }
        wandb.init(project=args.wandb_project_name, entity="neaf", config=config)

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
    global_step = start
    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    render_kwargs_train.update({'timeinterval': args.time_interval})
    render_kwargs_test.update({'timeinterval': args.time_interval})

    # get permanent recs for testing
    gt_log_dict = None
    irtestdir = os.path.join(basedir, expname, 'ir_tests')
    os.makedirs(irtestdir, exist_ok=True)
    perm_test_recs, ir_gt = build_neaf_batch(listener_states, i_test, args, reccount=args.neaf_permtest_cnt, mode='ir')
    gt_log_dict = save_ir(ir_gt, None, 'ir_groundt', savedir=irtestdir)

    # Short circuit if only rendering out from trained model
    if args.render_only:
        # TODO for neaf
        print('RENDER ONLY')
        quit()

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    use_batching = not args.no_batching

    N_iters = 20000 + 1
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    if args.use_wandb:
        wandb.config.update({"epochs": N_iters - 1})

    if args.bootstrap_only:
        print("Exiting because bootstrap only flag was set!")
        quit()

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    loss_list = []
    psnr_list = []
    time_list = []
    start = start + 1
    time0 = time.time()
    for i in trange(start, N_iters):
        # Sample random ray batch
        if use_batching:
            np.random.shuffle(i_train)
            batch_rays, target_s, times = build_neaf_batch(listener_states, i_train, args)
        else:
            # Random from one location
            # build receiver batch from precomputed rays
            listener_i = np.random.choice(i_train, (1,))
            batch_rays, target_s, times = build_neaf_batch(listener_states, listener_i, args)

        #####  Core optimization loop  #####
        rgb, depth, acc, extras = render(chunk=args.chunk, rays=batch_rays,
                                         verbose=i < 10, retraw=True, times=times,
                                         **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if 'sparsity_loss0' in extras:
            sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].sum() + extras["sparsity_loss0"].sum())
        else:
            sparsity_loss = args.sparse_loss_weight*(extras["sparsity_loss"].sum())
        loss = loss + sparsity_loss

        # add Total Variation loss
        if args.i_embed==1:
            n_levels = render_kwargs_train["embed_fn"].n_levels
            min_res = render_kwargs_train["embed_fn"].base_resolution
            max_res = render_kwargs_train["embed_fn"].finest_resolution
            log2_hashmap_size = render_kwargs_train["embed_fn"].log2_hashmap_size
            TV_loss = sum(total_variation_loss(render_kwargs_train["embed_fn"].embeddings[i], \
                                              min_res, max_res, \
                                              i, log2_hashmap_size, \
                                              n_levels=n_levels) for i in range(n_levels))
            loss = loss + args.tv_loss_weight * TV_loss
            if i>1000:
                args.tv_loss_weight = 0.0

        log_dict = {"loss": loss}

        loss.backward()
        # pdb.set_trace()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        t = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        # Rest is logging
        if i%args.i_weights==0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            if args.i_embed==1:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'embed_fn_state_dict': render_kwargs_train['embed_fn'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            else:
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
            print('Saved checkpoints at', path)
            print('Saved checkpoints at', path)

        if i%args.i_testset==0 and i > 0:
            # get loss and ir for testset of receivers
            np.random.shuffle(i_test)
            with torch.no_grad():
                recs, test_targets, times = build_neaf_batch(listener_states, i_test, args, reccount=1024)
                test_rgbs = render_recs(recs, times, args.chunk, render_kwargs_test)
                img_loss_test = img2mse(test_rgbs, test_targets)
                psnr_test = mse2psnr(img_loss_test)
                tqdm.write(f"[TEST] Iter: {i} Recs: 1024 Loss: {img_loss_test.item()}  PSNR: {psnr_test.item()}")
                log_dict.update({"test_loss": img_loss_test,
                                 "test_psnr": psnr_test})
            with torch.no_grad():
                irs = render_ir(perm_test_recs, args.neaf_timesteps, i, args.chunk, render_kwargs_test)
                extra_log_dict = save_ir(irs, perm_test_recs, f"ir_{i}", irtestdir, truth=ir_gt)
                log_dict.update(extra_log_dict)

        # Manual alpha logging for DEBUGGING
        if i%10==0 and source_pos is not None:
            # get alpha at source position if available and log to wandb
            # TODO "manually" query model for alpha
            # time and direction values are irrelevant for the alpha, just ask the model directly :D
            # need query fn, set of points, no or pseudo viewdir and pttimes, ref to model

            pt = torch.stack((source_pos, torch.Tensor([0., 0., 0.])), dim=0)[None, :, :]  # Tensor needs shape [N_rays, N_Samples, 3]
            vd = torch.Tensor([1., 0., 0.])[None, :]  # Tensor needs shape [N_rays, 3]
            ti = torch.Tensor([[.5], [.5]])[None, :, :]  # Tensor needs shape [N_rays, N_Samples, 3]

            with torch.no_grad():
                raws = simple_model_query(pt, vd, ti, render_kwargs_train['network_query_fn'],
                                          render_kwargs_train['network_fn'])
            raw_alpha_at_source = raws[0, 0, 3]
            raw_alpha_at_origin = raws[0, 1, 3]
            log_dict.update({"raw_alpha_at_source": raw_alpha_at_source,
                             "raw_alpha_at_origin": raw_alpha_at_origin})

        # log to wandb
        if args.use_wandb:
            if gt_log_dict is not None:
                log_dict.update({'ir_gt': gt_log_dict['ir']})
                gt_log_dict = None
            wandb.log(log_dict)

        # cmdln output
        if i%args.i_print==0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
            loss_list.append(loss.item())
            psnr_list.append(psnr.item())
            time_list.append(t)
            loss_psnr_time = {
                "losses": loss_list,
                "psnr": psnr_list,
                "time": time_list
            }
            with open(os.path.join(basedir, expname, "loss_vs_time.pkl"), "wb") as fp:
                pickle.dump(loss_psnr_time, fp)

        global_step += 1


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
