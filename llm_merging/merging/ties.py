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


class ties(Merges):
    def __init__(self, name):
        super().__init__(name)

        '''
        # These values are meant to be modified by the user.
        # '''
        # # Give a list of models to load for the merge. Each element is the list a is a tuple of (model, revision_id). We recommend specifying a revision id to ensure the model was not modified after May 31 
        self.list_models = [("EmbeddedLLM/Mistral-7B-Merge-14-v0.2", "85495c5f428c3efa6f91639818c6403dbb18559c"),
                            ("mlabonne/NeuralHermes-2.5-Mistral-7B","6f2b949fbb84997e62db9aef4f05f84b95344c90")]

        # # Hyperparameters 
        self.base_model_name = "mistralai/Mistral-7B-v0.1"
        #     #use llama to test on mac
    
        # # We recommend specifying a revision id to ensure the model was not modified after May 31 
        self.base_model_revision_id = "7231864981174d9bee8c7687c24c8344414eae6b"
        model_rte = AutoModelForSeq2SeqLM.from_pretrained("PavanNeerudu/t5-base-finetuned-rte").to("cpu").state_dict()


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
        model_rte = AutoModelForCausalLM.from_pretrained("EmbeddedLLM/Mistral-7B-Merge-14-v0.2").to("cpu").state_dict()
        model_mnli = AutoModelForCausalLM.from_pretrained("mlabonne/NeuralHermes-2.5-Mistral-7B").to("cpu").state_dict()
        model_sst2 = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to("cpu").state_dict()

        # '''
        # 2) Merge checkpoints  
        # '''
        parameter_lambdas = [0.5, 0.3]

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