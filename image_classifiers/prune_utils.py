import torch 
import torch.nn as nn 
from layerwrapper import WrappedLayer 

import os
import numpy as np
import weightwatcher as ww
from esd_utils import net_esd_estimator

from torch.nn.utils import prune

def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(model):
    subset = find_layers(model, layers=[nn.Linear])
    zero_cnt = 0
    fc_params = 0
    for name in subset:
        W = subset[name].weight.data
        if W.shape[0] == 1000:
            continue 
        zero_cnt += (W==0).sum().item()
        fc_params += W.numel()
    return float(zero_cnt) / fc_params

def compute_mask(W_metric, prune_granularity, sparsity):
    thres = torch.sort(W_metric.flatten().cuda())[0][int(W_metric.numel() * sparsity)].cpu()
    W_mask = (W_metric <= thres)
    return W_mask 

def prune_vit(args, model, calib_data, device):
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])
    
    metric_stats = []
    for blk in model.blocks:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    thresh = None 
    #####################################
    inps = model.patch_embed(inps)
    cls_tokens = model.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    inps = torch.cat((cls_tokens, inps), dim=1)
    inps = inps + model.pos_embed
    inps = model.pos_drop(inps)

    for block_id, blk in enumerate(model.blocks):
        print(f"block {block_id}")
        subset = find_layers(blk)

        # wanda
        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    tmp_res.append(blk(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = blk(inps)

            for h in handles:
                h.remove()    
        
        ################# pruning ###################
        for name in subset:
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)

            subset[name].weight.data[W_mask] = 0
        ##############################################

def prune_convnext(args, model, calib_data, device):
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda"])
    
    metric_stats = []
    for block_id in range(4):
        subset = find_layers(model.stages[block_id])
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)
    ##############################################################

    thresh = None 
    for block_id in range(4):
        print(f"block {block_id}")
        subset = find_layers(model.stages[block_id])

        # wanda
        if require_forward:
            layer = model.downsample_layers[block_id]
            if bs > 1024:
                tmp_res = []
                for i1 in range(0, bs, 512):
                    j1 = min(i1+512, bs)
                    tmp_res.append(layer(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = layer(inps)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
               handles.append(subset[name].register_forward_hook(add_batch(name)))
            layer = model.stages[block_id]
            if bs > 1024:
                tmp_res = []
                for i1 in range(0, bs, 512):
                    j1 = min(i1+512, bs)
                    tmp_res.append(layer(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = layer(inps)
            for h in handles:
                h.remove()
        
        
        
        ################# pruning ###################
        for name in subset:
            
            if args.prune_metric == "wanda":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            
            
            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, args.sparsity)

            subset[name].weight.data[W_mask] = 0
        ##############################################
        
def prune_vit_ww(args, model, calib_data, device):
    layers = [find_layers(model.blocks)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    
    prunables = torch.tensor(prunables)
    
    args.metric_cache = args.metric_cache + f"/{args.model}"
    # get ww-based layerwise ratios
    if not os.path.exists(args.metric_cache):
        os.makedirs(args.metric_cache)
         
    if os.path.exists(f"{args.metric_cache}/{args.WW_metric}.npy"):
        metrics = np.load(f"{args.metric_cache}/{args.WW_metric}.npy")
    
    else:
        print("weightwatcher analysis begin!")
        watcher = ww.WeightWatcher(model=model.blocks)
        details = watcher.analyze()
        
        if args.WW_metric == 'entropy':
            metrics = np.array(details.entropy)
        elif args.WW_metric == 'alpha':
            metrics = np.array(details.alpha)
            
        elif args.WW_metric == 'alpha_mid':
            metrics = net_esd_estimator(model.blocks,
                fix_fingers='xmin_mid'
            )
            metrics = metrics['alpha']
        elif args.WW_metric == 'alpha_peak':
            metrics = net_esd_estimator(model.blocks,
                fix_fingers='xmin_peak'
            )
            metrics = metrics['alpha']
            
        elif args.WW_metric == 'mp_softrank':
            metrics = np.array(details.mp_softrank)
        elif args.WW_metric == 'stable_rank':
            metrics = np.array(details.stable_rank)
        elif args.WW_metric == 'norm':
            metrics = np.array(details.norm)
        elif args.WW_metric == 'random_distance':
            metrics = np.array(details.rand_distance)
        elif args.WW_metric == 'log_norm':
            metrics = np.array(details.log_norm)
        elif args.WW_metric == 'log_spectral_norm':
            metrics = np.array(details.log_spectral_norm)
        elif args.WW_metric == 'alpha_weighted':
            metrics = np.array(details.alpha_weighted)
        elif args.WW_metric == 'log_alpha_norm':
            metrics = np.array(details.log_alpha_norm)
        elif args.WW_metric == 'spectral_norm':
            metrics = np.array(details.spectral_norm)
        
        np.save(f"{args.metric_cache}/{args.WW_metric}.npy", metrics)

    scores = torch.tensor(metrics)
        
    # balance allocation
    alpha_max = torch.max(scores)
    alpha_min = torch.min(scores)
    layerwise_pruning_ratios = (((scores - alpha_min) / (alpha_max - alpha_min)) * (2*args.epsilon) + (1-args.epsilon))
    scaler = torch.sum(prunables) * args.sparsity / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    print(ratios)
    
    print("pruning begin!")
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda_ww"])

    metric_stats = []
    for blk in model.blocks:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    thresh = None 
    inps = model.patch_embed(inps)
    cls_tokens = model.cls_token.expand(bs, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
    inps = torch.cat((cls_tokens, inps), dim=1)
    inps = inps + model.pos_embed
    inps = model.pos_drop(inps)

    #####################################
    i = 0
    for block_id, blk in enumerate(model.blocks):
        subset = find_layers(blk)

        # wanda
        if require_forward:
            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))

            if bs > 256:
                tmp_res = []
                for i1 in range(0, bs, 256):
                    j1 = min(i1+256, bs)
                    tmp_res.append(blk(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = blk(inps)

            for h in handles:
                h.remove()     
        
        ################# pruning ###################
        for name in subset:
            if args.prune_metric == "wanda_ww":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, ratios[i])
            i+=1
            subset[name].weight.data[W_mask] = 0
    

def prune_convnext_ww(args, model, calib_data, device):
    layers = [find_layers(model.stages)]
    prunables = []
    for layer in layers:
        for name in layer:
            prunables.append(layer[name].weight.numel())
    
    prunables = torch.tensor(prunables)
    
    # get ww-based layerwise ratios
    if not os.path.exists(args.metric_cache):
        os.makedirs(args.metric_cache)
         
    if os.path.exists(f"{args.metric_cache}/{args.WW_metric}.npy"):
        metrics = np.load(f"{args.metric_cache}/{args.WW_metric}.npy")
    
    else:
        print("weightwatcher analysis begin!")
        watcher = ww.WeightWatcher(model=model.stages)
        details = watcher.analyze()
        
        if args.WW_metric == 'entropy':
            metrics = np.array(details.entropy)
        elif args.WW_metric == 'alpha':
            metrics = np.array(details.alpha)
            
        elif args.WW_metric == 'alpha_mid':
            metrics = net_esd_estimator(model.stages,
                fix_fingers='xmin_mid'
            )
            metrics = metrics['alpha']
        elif args.WW_metric == 'alpha_peak':
            metrics = net_esd_estimator(model.stages,
                fix_fingers='xmin_peak'
            )
            metrics = metrics['alpha']
            
        elif args.WW_metric == 'mp_softrank':
            metrics = np.array(details.mp_softrank)
        elif args.WW_metric == 'stable_rank':
            metrics = np.array(details.stable_rank)
        elif args.WW_metric == 'random_distance':
            metrics = np.array(details.rand_distance)
        elif args.WW_metric == 'log_norm':
            metrics = np.array(details.log_norm)
        elif args.WW_metric == 'log_spectral_norm':
            metrics = np.array(details.log_spectral_norm)
        elif args.WW_metric == 'alpha_weighted':
            metrics = np.array(details.alpha_weighted)
        elif args.WW_metric == 'log_alpha_norm':
            metrics = np.array(details.log_alpha_norm)
        elif args.WW_metric == 'spectral_norm':
            metrics.np.array(details.spectral_norm)
        
        np.save(f"{args.metric_cache}/{args.WW_metric}.npy", metrics)

    metrics = [metrics[i+1:i+3] for i in range(0, len(metrics), 3)]
    metrics = np.array(metrics).flatten()
    
    scores = torch.tensor(metrics)
        
    # balance allocation
    alpha_max = torch.max(scores)
    alpha_min = torch.min(scores)
    layerwise_pruning_ratios = (((scores - alpha_min) / (alpha_max - alpha_min)) * (2*args.epsilon) + (1-args.epsilon))
    scaler = torch.sum(prunables) * args.sparsity / (torch.sum(prunables * layerwise_pruning_ratios))  
    layerwise_pruning_ratios = layerwise_pruning_ratios * scaler
    ratios = layerwise_pruning_ratios.cpu().numpy().tolist()
    print(ratios)
    
    print("pruning begin!")
    inps = calib_data 
    bs = inps.shape[0]
    require_forward = (args.prune_metric in ["wanda_ww"])
    
    metric_stats = []
    for blk in model.stages:
        subset = find_layers(blk)
        res_per_layer = {}
        for name in subset:
            res_per_layer[name] = torch.abs(subset[name].weight.data)
        metric_stats.append(res_per_layer)

    thresh = None

    #####################################
    i = 0
    for block_id, blk in enumerate(model.stages):
        subset = find_layers(blk)

        # wanda
        if require_forward:
            layer = model.downsample_layers[block_id]
            if bs > 1024:
                tmp_res = []
                for i1 in range(0, bs, 512):
                    j1 = min(i1+512, bs)
                    tmp_res.append(layer(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = layer(inps)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = WrappedLayer(subset[name])

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp

            handles = []
            for name in wrapped_layers:
               handles.append(subset[name].register_forward_hook(add_batch(name)))
            layer = model.stages[block_id]
            if bs > 1024:
                tmp_res = []
                for i1 in range(0, bs, 512):
                    j1 = min(i1+512, bs)
                    tmp_res.append(layer(inps[i1:j1]))
                inps = torch.cat(tmp_res, dim=0)
            else:
                inps = layer(inps)
            for h in handles:
                h.remove() 
        
        ################# pruning ###################
        for name in subset:
            if args.prune_metric == "wanda_ww":
                metric_stats[block_id][name] *= torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = compute_mask(metric_stats[block_id][name], args.prune_granularity, ratios[i])
            i+=1
            subset[name].weight.data[W_mask] = 0