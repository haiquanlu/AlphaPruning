import torch
import torch.nn as nn
from operator import itemgetter
import numpy as np
import math
import weightwatcher as ww
from transformers import AutoModelForCausalLM

def get_llm(model_name, cache_dir="llm_weights", device="cpu"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True
    )
    model.to(device)  # Move model to the specified device
    model.seqlen = 2048
    return model


def get_esd_metrics(model_name, metric_name, cache_dir="llm_weights"):
    model = get_llm(model_name, cache_dir)
    model.eval()

    if "opt" in model_name:
        blocks = model.model.decoder.layers    
    else:
        blocks = model.model.layers
    
    if metric_name == 'alpha_mid':
        metrics = net_esd_estimator(blocks,
            fix_fingers='xmin_mid'
        )
        metrics = metrics['alpha']
    elif metric_name == 'alpha_peak':
        metrics = net_esd_estimator(blocks,
            fix_fingers='xmin_peak'
        )
        metrics = metrics['alpha']
    else:
        watcher = ww.WeightWatcher(model=blocks)
        details = watcher.analyze(mp_fit=True, randomize=True)
        
        if metric_name == 'entropy':
            metrics = np.array(details.entropy)
        elif metric_name == 'alpha':
            metrics = np.array(details.alpha)
        elif metric_name == 'mp_softrank':
            metrics = np.array(details.mp_softrank)
        elif metric_name == 'stable_rank':
            metrics = np.array(details.stable_rank)
        elif metric_name == 'random_distance':
            metrics = np.array(details.rand_distance)
        elif metric_name == 'log_norm':
            metrics = np.array(details.log_norm)
        elif metric_name == 'log_spectral_norm':
            metrics = np.array(details.log_spectral_norm)
        elif metric_name == 'alpha_weighted':
            metrics = np.array(details.alpha_weighted)
        elif metric_name == 'log_alpha_norm':
            metrics = np.array(details.log_alpha_norm)
        elif metric_name == 'spectral_norm':
            metrics = np.array(details.spectral_norm)
    
    torch.cuda.empty_cache()

    return metrics    

def net_esd_estimator(
            net=None,
            EVALS_THRESH=0.00001,
            bins=100,
            fix_fingers=None,
            xmin_pos=2,
            conv_norm=0.5, 
            filter_zeros=False):
    """_summary_

    Args:
        net (_type_, optional): model. Defaults to None.
        EVALS_THRESH (float, optional): eval threshold to filter near-zero. Defaults to 0.00001.
        bins (int, optional): _description_. Defaults to 100.
        fix_fingers (_type_, optional): [None, 'xmin_peak', 'xmin_mid']
        xmin_pos:   2 = middle of the spectrum selected as xmin,    larger than 2 means select smaller eigs as xmin

    Returns:
        _type_: _description_
    """
    results = {
        'alpha':[],
        'spectral_norm': [],
        'D': [],
        'longname':[],
        'eigs':[],
        'norm':[],
        'alphahat': []
        }
    print("=================================")
    print(f"fix_fingers: {fix_fingers}, xmin_pos: {xmin_pos}, conv_norm: {conv_norm}, filter_zeros: {filter_zeros}")
    print("=================================")
    # iterate through layers
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            matrix = m.weight.data.clone().cpu()
            # i have checked that the multiplication won't affect the weights value
            # normalization and tranpose Conv2d
            if isinstance(m, nn.Conv2d):
                matrix = torch.flatten(matrix, start_dim=2) * math.sqrt(conv_norm)
                matrix = matrix.transpose(1, 2).transpose(0, 1)
            matrix = matrix.to(torch.float32)
            eigs = torch.square(torch.linalg.svdvals(matrix).flatten())
            # ascending order 
            eigs, _ = torch.sort(eigs, descending=False)
            spectral_norm = eigs[-1].item()
            fnorm = torch.sum(eigs).item()
            
            if filter_zeros:
                nz_eigs = eigs[eigs > EVALS_THRESH]
                N = len(nz_eigs)
                # somethines N may equal 0, if that happens, we don't filter eigs
                if N == 0:
                    nz_eigs = eigs
                    N = len(nz_eigs)
            else:
                nz_eigs = eigs
                N = len(nz_eigs)
            log_nz_eigs  = torch.log(nz_eigs)

            if fix_fingers == 'xmin_mid':
                i = int(len(nz_eigs) / xmin_pos)    
                xmin = nz_eigs[i]
                n = float(N - i)
                seq = torch.arange(n)
                final_alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                final_D = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-final_alpha + 1) - seq / n     
                        ))
            else:
                alphas = torch.zeros(N-1)
                Ds     = torch.ones(N-1)
                if fix_fingers == 'xmin_peak':
                    hist_nz_eigs = torch.log10(nz_eigs)
                    min_e, max_e = hist_nz_eigs.min(), hist_nz_eigs.max()
                    counts = torch.histc(hist_nz_eigs, bins, min=min_e, max=max_e)
                    boundaries = torch.linspace(min_e, max_e, bins + 1)
                    h = counts, boundaries
                    ih = torch.argmax(h[0])  # 
                    xmin2 = 10 ** h[1][ih]
                    xmin_min = torch.log10(0.95 * xmin2)
                    xmin_max = 1.5 * xmin2
                
                for i, xmin in enumerate(nz_eigs[:-1]):
                    if fix_fingers == 'xmin_peak':
                        if xmin < xmin_min:
                            continue
                        if xmin > xmin_max:
                            break

                    n = float(N - i)
                    seq = torch.arange(n)
                    alpha = 1 + n / (torch.sum(log_nz_eigs[i:]) - n * log_nz_eigs[i])
                    alphas[i] = alpha
                    if alpha > 1:
                        Ds[i] = torch.max(torch.abs(
                            1 - (nz_eigs[i:] / xmin) ** (-alpha + 1) - seq / n     
                        ))

                min_D_index = torch.argmin(Ds)
                final_alpha = alphas[min_D_index]
                final_D = Ds[min_D_index]
            
            final_alpha = final_alpha.item()
            final_D = final_D.item()
            final_alphahat=final_alpha*math.log10(spectral_norm)

            results['spectral_norm'].append(spectral_norm)
            results['alphahat'].append(final_alphahat)
            results['norm'].append(fnorm)
            results['alpha'].append(final_alpha)
            results['D'].append(final_D)
            results['longname'].append(name)
            results['eigs'].append(eigs.detach().cpu().numpy())

    return results