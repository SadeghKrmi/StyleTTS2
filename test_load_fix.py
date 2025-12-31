
import torch
import torch.nn as nn
from models import load_checkpoint
import os

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

# Mock DataParallel to create the target structure
class MockDataParallel(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    # Forward just to mimic behavior if needed, but we care about state_dict loading
    def forward(self, x):
        return self.module(x)

def test_loading():
    # 1. Create a "Single GPU" model and save it
    single_model = SimpleModel()
    torch.save({'net': {'my_component': single_model.state_dict()}, 'epoch': 0, 'iters': 0, 'optimizer': {}}, 'test_ckpt.pth')
    print("Saved single GPU checkpoint.")

    # 2. Create a "DataParallel" model (target)
    # The target expects keys like 'module.fc.weight', etc.
    target_model = MockDataParallel(SimpleModel())
    
    # Wrap in a dict as the real code uses Munch/dict
    models_dict = {'my_component': target_model}
    
    # 3. Try to load using the fixed function
    print("Attempting to load Single GPU checkpoint into DataParallel model...")
    class MockOptimizer:
        def load_state_dict(self, state): pass
    
    try:
        load_checkpoint(models_dict, MockOptimizer(), 'test_ckpt.pth', load_only_params=True)
        print("SUCCESS: Load completed without error.")
    except Exception as e:
        print(f"FAILURE: Load failed with error: {e}")
        raise e
    
    # 4. Verify parameter values match
    # Load original state dict
    original_sd = torch.load('test_ckpt.pth')['net']['my_component']
    # Check target model parameters
    # Target model state dict (if unwrapped) or access via .module
    target_sd = target_model.module.state_dict()
    
    for k in original_sd:
        if not torch.allclose(original_sd[k], target_sd[k]):
            print(f"FAILURE: Parameter mismatch for {k}")
            return

    print("SUCCESS: Parameters match verified.")

    # Cleanup
    if os.path.exists('test_ckpt.pth'):
        os.remove('test_ckpt.pth')

if __name__ == "__main__":
    test_loading()
