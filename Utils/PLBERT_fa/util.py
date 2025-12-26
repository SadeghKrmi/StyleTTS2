import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel, BertConfig, BertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state

class CustomBert(BertModel):
    def forward(self, *args, **kwargs):
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state

def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    if not os.path.exists(config_path):
        config_path = os.path.join(log_dir, "config_fa.yml")
    
    plbert_config = yaml.safe_load(open(config_path))
    
    try:
        albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
        bert = CustomAlbert(albert_base_configuration)
    except Exception:
        # Fallback to BERT if ALBERT config fails (e.g. unknown params or explicit choice)
        bert_base_configuration = BertConfig(**plbert_config['model_params'])
        bert = CustomBert(bert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = []
    for f in ckpts:
        if not os.path.isfile(os.path.join(log_dir, f)):
            continue
        try:
            # Try to parse "step_123..." -> 123
            # Split by '_' and take the second element (index 1) which should be the number
            parts = f.split('_')
            if len(parts) > 1 and parts[1].isdigit():
                iters.append(int(parts[1]))
            else:
                # Fallback to original logic if format is different but still tries to be robust
                iters.append(int(f.split('_')[-1].split('.')[0]))
        except ValueError:
            continue
            
    if not iters:
        raise FileNotFoundError(f"No valid checkpoints found in {log_dir}")
        
    iters = sorted(iters)[-1]
    
    # Find the file corresponding to this iteration
    # If there are multiple files with the same step, pick one (usually there shouldn't be, or we need to match exact pattern)
    # The original code constructed the filename: log_dir + "/step_" + str(iters) + ".t7"
    # But now we might have suffixes. We need to find the file that starts with step_{iters}_ or exactly step_{iters}.t7
    
    candidates = [f for f in ckpts if f.startswith(f"step_{iters}_") or f == f"step_{iters}.t7"]
    if not candidates:
         checkpoint_path = log_dir + "/step_" + str(iters) + ".t7" # Fallback to constructing it
    else:
         checkpoint_path = os.path.join(log_dir, candidates[0])

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.encoder.'): # Fix for some nested encoders potentially
             name = name[16:]
        elif name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            
        # Remove leading dot if present (fix for the issue seen in verification)
        if name.startswith('.'):
            name = name[1:]
            
        new_state_dict[name] = v
    
    if "embeddings.position_ids" in new_state_dict:
        del new_state_dict["embeddings.position_ids"]
        
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert
