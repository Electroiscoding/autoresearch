import time
import torch
import torch.nn as nn
import numpy as np
import sys
import importlib.util

# Load module with numeric prefix (invalid for direct import)
spec = importlib.util.spec_from_file_location("engine", "2_engine.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
custom_train_step = engine.custom_train_step

# --- 1. THE TOY MODEL BASELINE ---
# A strictly CPU-bound 1-Million parameter network.
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.fc1 = nn.Linear(1000, 1000) # 1M Params
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 10)
    
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def run_pytorch_baseline(inputs, targets, iterations=100):
    model = ToyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    for _ in range(iterations):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    total_time = time.time() - start_time
    return total_time, loss.item()

def run_custom_engine(inputs, targets, iterations=100):
    # Extract raw numpy arrays to feed the custom engine
    np_inputs = inputs.numpy()
    np_targets = targets.numpy()
    
    # Initialize random weights matching the 1M param architecture
    w1 = np.random.randn(1000, 1000).astype(np.float32) * 0.01
    w2 = np.random.randn(1000, 10).astype(np.float32) * 0.01
    weights = [w1, w2]
    
    start_time = time.time()
    try:
        for _ in range(iterations):
            # Beta's code is executed here
            weights = custom_train_step(weights, np_inputs, np_targets)
    except Exception as e:
        print(f"CRITICAL ENGINE FAILURE: {str(e)}", file=sys.stderr)
        sys.exit(1)
        
    total_time = time.time() - start_time
    return total_time, weights

if __name__ == "__main__":
    print(">>> INITIATING TRINITY BENCHMARK <<<")
    
    # Generate static dummy data for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)
    inputs = torch.randn(128, 1000) # Batch of 128
    targets = torch.randint(0, 10, (128,))
    
    print("Running PyTorch/Adam Baseline (100 iterations)...")
    pt_time, pt_loss = run_pytorch_baseline(inputs, targets)
    
    print("Running Custom Engine (100 iterations)...")
    custom_time, _ = run_custom_engine(inputs, targets)
    
    print("\n--- RESULTS ---")
    print(f"PyTorch Time:      {pt_time:.4f} seconds")
    print(f"Custom Time:       {custom_time:.4f} seconds")
    
    speed_factor = pt_time / custom_time if custom_time > 0 else float('inf')
    print(f"Speed Multiplier:  {speed_factor:.2f}x")
    
    # THE IRONCLAD 100x RULE
    if speed_factor < 100:
        error_msg = f"BENCHMARK FAILED: Reached {speed_factor:.2f}x speedup. Target is 100x."
        print(error_msg, file=sys.stderr)
        sys.exit(1)
        
    print("BENCHMARK PASSED: 100x SPEEDUP ACHIEVED.")
    sys.exit(0)