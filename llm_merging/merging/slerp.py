import torch 
from llm_merging.merging.Merges import Merges
from peft import get_peft_model, set_peft_model_state_dict
import sys
import os, copy
import torch
import matplotlib.pyplot as plt
import numpy as np
import re
from collections import OrderedDict
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import subprocess
import os
import shutil
import tkinter as tk
from tkinter import filedialog
from colorama import init, Fore, Style

def lerp(t, v0, v1):
    return (1 - t) * v0 + t * v1

def slerp_func(t, v0, v1, DOT_THRESHOLD=0.9995):
    epsilon = 1e-10

    # Convert tensors to a common format, float32
    v0 = v0.to(dtype=torch.float32)
    v1 = v1.to(dtype=torch.float32)

    # Convert tensors to numpy arrays
    c = False
    if not isinstance(v0, np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1, np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()

    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)

    # Normalize the vectors to get the directions and angles    
    norm_v0 = np.linalg.norm(v0)
    norm_v1 = np.linalg.norm(v1)

    if norm_v0 > epsilon:
        v0 = v0 / norm_v0
    else:
        print(f"Warning: Norm of v0 is very small ({norm_v0}). Skipping normalization.")

    if norm_v1 > epsilon:
        v1 = v1 / norm_v1
    else:
        print(f"Warning: Norm of v1 is very small ({norm_v1}). Skipping normalization.")

    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy

    del v0_copy, v1_copy
    del v1

    if c:
        res = torch.from_numpy(v2)
    else:
        res = v2
    return res

# Check and pad vocabularies if necessary using state dictionaries
def pad_state_dicts_with_different_tensors(primary_state_dict, secondary_state_dict):
    with torch.no_grad(): 
        # Get common keys
        common_keys = set(primary_state_dict.keys()).intersection(set(secondary_state_dict.keys()))
        
        for key in common_keys:
            tensor1 = primary_state_dict[key]
            tensor2 = secondary_state_dict[key]
            
            if tensor1.size() != tensor2.size():
                if tensor1.size(0) < tensor2.size(0):
                    # Pad the first tensor to match the size of the second tensor
                    padding_size = tensor2.size(0) - tensor1.size(0)
                    padding = torch.zeros((padding_size,) + tensor1.size()[1:], device=tensor1.device, dtype=tensor1.dtype)
                    primary_state_dict[key] = torch.cat([tensor1, padding], dim=0)
                elif tensor1.size(0) > tensor2.size(0):
                    # Pad the second tensor to match the size of the first tensor
                    padding_size = tensor1.size(0) - tensor2.size(0)
                    padding = torch.zeros((padding_size,) + tensor2.size()[1:], device=tensor2.device, dtype=tensor2.dtype)
                    secondary_state_dict[key] = torch.cat([tensor2, padding], dim=0)

        # For keys that are not common, add the missing parameters from one model to the other with appropriate zero padding
        for key in primary_state_dict:
            if key not in secondary_state_dict:
                tensor = primary_state_dict[key]
                padding = torch.zeros_like(tensor)
                secondary_state_dict[key] = padding

        for key in secondary_state_dict:
            if key not in primary_state_dict:
                tensor = secondary_state_dict[key]
                padding = torch.zeros_like(tensor)
                primary_state_dict[key] = padding

        # Ensure vocab sizes match in both models
        primary_vocab_size = primary_state_dict['model.embed_tokens.weight'].size(0)
        secondary_vocab_size = secondary_state_dict['model.embed_tokens.weight'].size(0)
        assert primary_vocab_size == secondary_vocab_size, "Vocab sizes do not match even after padding!"

class slerp(Merges):
    def __init__(self, name):
        super().__init__(name)

        '''
        # These values are meant to be modified by the user.
        # '''
        # # Give a list of models to load for the merge. Each element is the list a is a tuple of (model, revision_id). We recommend specifying a revision id to ensure the model was not modified after May 31 
        # self.list_models = [("EmbeddedLLM/Mistral-7B-Merge-14-v0.2", "85495c5f428c3efa6f91639818c6403dbb18559c"),
        #                     ("mlabonne/NeuralHermes-2.5-Mistral-7B","6f2b949fbb84997e62db9aef4f05f84b95344c90")]

        # # Hyperparameters 
        self.base_model_name = "mistralai/Mistral-7B-v0.1"
        #     #use llama to test on mac
    
        # # We recommend specifying a revision id to ensure the model was not modified after May 31 
        self.base_model_revision_id = "7231864981174d9bee8c7687c24c8344414eae6b"
        # model_rte = AutoModelForSeq2SeqLM.from_pretrained("PavanNeerudu/t5-base-finetuned-rte").to("cpu").state_dict()


        # self.max_seq_len = None
        # self.max_gen_len = 64
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # # Architecture must match base model. 
        # self.architecture = "decoder"
        # '''
        # These are variables used later in the code and not intended to be set, but feel free to adapt to your use case.  
        # '''
        # # Loaded models and configs 
        # self.loaded_models = {}
        # self.loaded_configs = {}

        # # Merged model parameters
        # self.merged_model = {}



    # Implement merge function 
    def merge(
        self,
    ):

        '''
        # 1) Load HuggingFace checkpoints and configs 
        # '''
        # super()._load_huggingface_models_and_configs()
        primary_model_name = "mistralai/Mistral-7B-v0.1"
        secondary_model_name = "EmbeddedLLM/Mistral-7B-Merge-14-v0.2"
        primary_model = AutoModelForCausalLM.from_pretrained(primary_model_name).to("cpu").state_dict()

        secondary_model = AutoModelForCausalLM.from_pretrained(secondary_model_name).to("cpu").state_dict()
        # model_mnli = AutoModelForCausalLM.from_pretrained("mlabonne/NeuralHermes-2.5-Mistral-7B").to("cpu").state_dict()
        pad_state_dicts_with_different_tensors(primary_model['state_dict'], secondary_model['state_dict'])
        v0 = primary_model['state_dict']
        v1 = secondary_model['state_dict']

        for key in set(v0.keys()).union(set(v1.keys())):
            if key in v0 and key in v1:
                # Check if both values are tensors
                if isinstance(v0[key], torch.Tensor) and isinstance(v1[key], torch.Tensor):
                    v0[key] = slerp_func((float(1.0) - 0.5), v0[key], v1[key])
                else:
                    print(f"Skipping key {key} because it does not point to tensors.")
            if key in v1 and key not in v0:
                v0[key] = v1[key]
                del v1[key]
        del secondary_model
        blended_model_savedir = "models"
        print(f"{Fore.YELLOW}\nCopying necessary files and saving blended model to: {blended_model_savedir}{Style.RESET_ALL}")
        for key, value in v0.items():
            if isinstance(value, np.ndarray):
                v0[key] = torch.tensor(value)

        resulting_vocab_size = primary_model['state_dict']['model.embed_tokens.weight'].size(0)

        config = AutoConfig.from_pretrained(primary_model_name)
        if config.vocab_size != resulting_vocab_size:
            print(f"Updating config vocab size from {config.vocab_size} to {resulting_vocab_size}")
            config.vocab_size = resulting_vocab_size

        self.merged_model = AutoModelForCausalLM.from_config(config)
        self.merged_model.load_state_dict(primary_model['state_dict'])
        self.base_model.load(self.merged_model)
        self.base_model.eval()


        # '''
        # 2) Merge checkpoints  
        # '''
        # parameter_lambdas = [0.5, 0.3]

        # # Get individual models 
        # all_models = list(self.loaded_models.values())

        # # Get all the parameters names (uses the first model and assume all the models have the same parameter)
        # all_parameter_names = all_models[0].keys()

        # for parameter_name in all_parameter_names:
        #     merged_parameter = None
        #     for parameter_lambda, model in zip(parameter_lambdas, all_models):
        #         parameter = model[parameter_name]
        #         if merged_parameter is None:
        #             merged_parameter = torch.clone(parameter) * parameter_lambda
        #         else:
        #             # first model has rank 16 and second model has rank 8, so we expand the second model to rank 16 by adding zeros
        #             if "A" in parameter_name:
        #                 parameter = torch.cat([torch.zeros_like(parameter), parameter], dim=0)
        #             else:
        #                 assert "B" in parameter_name
        #                 parameter = torch.cat([torch.zeros_like(parameter), parameter], dim=1)
        #             merged_parameter += parameter * parameter_lambda
        #     self.merged_model[parameter_name] = merged_parameter

        # '''
        # 3) Load base model and tokenizer
        # '''
        # self._load_base_model()
        # self._load_tokenizer()

        # '''
        # 4) Load merged model into base model 
        # '''
        # # Modify the base model. This is needed for Peft, which wraps the base_model in a Peft wrapper. 
        # huggingface_config = list(self.loaded_configs.values())[0]
        # if huggingface_config is not None:
        #     self.base_model = get_peft_model(self.base_model, huggingface_config)
        #     set_peft_model_state_dict(self.base_model, self.merged_model)
        
        # else:
        #     self.base_model.load(self.merged_model)

        # # Requires to make results deterministic. If not set, we will just run once and use the results from the first pass. 
        # self.base_model.eval()

        # return self.base_model