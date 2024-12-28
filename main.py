import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from lib.prune import prune_wanda, prune_sparsegpt, prune_magnitude, prune_wanda_ww, prune_sparsegpt_ww, prune_magnitude_ww, check_sparsity 
from lib.eval import eval_ppl, eval_zero_shot
from lib.esd_utils import get_esd_metrics


def get_llm(model, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model,
        torch_dtype = torch.float16,
        cache_dir = cache_dir,
        low_cpu_mem_usage=True,
        device_map = "auto"
    )
    
    model.seqlen = 2048
    return model


def main():
   
    parser = argparse.ArgumentParser()  
    parser.add_argument('--model', type=str, help="model type")
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument('--prune_method', type=str)
    parser.add_argument('--sparsity_type', type=str, default="unstructured", help='Structured pruning for N:M')
    parser.add_argument('--cache_dir', default="llm_weights", type=str)
    parser.add_argument('--save', type=str, default=None, help='Path to save results')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the wanda paper appendix")
    
    # params for WW
    parser.add_argument("--ww_metric", default="alpha_peak", type=str, help="the WW-based metric to ues.")
    parser.add_argument("--ww_metric_cache", default="./data/llama-7b-hf")
    parser.add_argument("--epsilon", default=0.3, type=float, help="for pruning ratio allocation.")
    parser.add_argument("--mapping_type", default="block_wise", type=str, help="mapping type for pruning ratios allocation.")
    # evaluation benchmark
    parser.add_argument("--eval_zero_shot", action="store_true", help="evaluation on zero-shot tasks.")
    parser.add_argument("--eval_wikitext", type=bool, default=True, help="evaluation on wikitext.")
   
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    
    # get the layerwise metric values of the model
    if "ww" in args.prune_method and not os.path.exists(f"{args.ww_metric_cache}/{args.ww_metric}.npy"):
        metric_values = get_esd_metrics(args.model, args.ww_metric, args.cache_dir)
        np.save(f"{args.ww_metric_cache}/{args.ww_metric}.npy", metric_values)



    model_name = args.model.split("/")[-1]
    model = get_llm(args.model, args.cache_dir)
    model.eval()

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    
    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model or "70b" in args.model in args.model: # for 30b or 65b or 70b, we use device_map to load onto multiple GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        # Uniform pruning
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

        ################################################
        # Pruning with our layerwise pruning ratios
        elif args.prune_method == "wanda_ww":
            prune_wanda_ww(args, model, tokenizer, device)

        elif args.prune_method == "magnitude_ww":
            prune_magnitude_ww(args, model, tokenizer, device)

        elif args.prune_method == "sparsegpt_ww":
            prune_sparsegpt_ww(args, model, tokenizer, device)
            
    sparsity_ratio = check_sparsity(model)
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    if args.eval_wikitext:
        ppl_test = eval_ppl(args, model, tokenizer, device)
        print(f"wikitext perplexity {ppl_test}")

        save_filepath = os.path.join(args.save, f"perplexity_{args.prune_method}_sparsity_{args.sparsity_ratio}.txt")
        with open(save_filepath, "w") as f:
            print("method\tactual_sparsity\tppl_test", file=f, flush=True)
            print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)
    
    # zero-shot tasks evaluation
    if args.eval_zero_shot:
        accelerate=False
        
        if "30b" in args.model or "65b" in args.model or "70b" in args.model or 'Llama-2-13b-hf' in args.model:
            accelerate=True

        task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        num_shot = 0
        results = eval_zero_shot(args.model, model, tokenizer, task_list, num_shot, accelerate)
        print("zero_shot evaluation results")
        print(results)

        save_filepath = os.path.join(args.save, f"zero_shot_{args.prune_method}_sparsity_{args.sparsity_ratio}.txt")
        with open(save_filepath, "w") as f:
            print(f"{args.prune_method}:\n{results}", file=f, flush=True)
    
    # save model if needed.    
    if args.save_model:
        save_model_path = os.path.join(args.save_model, f"{args.prune_method}_{args.sparsity_ratio}")
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        model.save_pretrained(save_model_path)
        tokenizer.save_pretrained(save_model_path)
    
    
if __name__ == '__main__':
    main()
