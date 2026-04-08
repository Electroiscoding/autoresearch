🔬 Project: The Backprop Challenger (CPU-Efficiency Mission)
This is an autonomous research mission to destroy the "Backpropagation/AdamW" monopoly by discovering a training strategy 100x more efficient on a single CPU.
🏁 Setup
Run Tag: Propose a tag (e.g., cpu-kill-backprop).
Branch: git checkout -b autoresearch/<tag>.
Files: Read README.md, prepare.py, and train.py.
Initialize: Create results.tsv with headers.
Confirm and go.
🚀 The Research Mission
The goal is to achieve the lowest val_bpb on a Single-CPU within a fixed 5-minute budget. Standard Backprop + AdamW is the enemy.
The "Challenger" Directives:
Abolish Backprop: Experiment with Forward-Forward, PEPITA, Feedback Alignment, or local Hebbian update rules.
CPU Optimization: Focus on architectures (SSMs, Linear Attention, RWKV) and update rules that are mathematically efficient for CPU cycles, not just GPU TFLOPS.
Extreme Math: Explore low-precision updates, lookup-table training, or sparse "active neuron" updates.
Goal: 100x better compute-efficiency than the baseline without quality loss.
⚖️ Rules
Edit ONLY train.py: Change anything inside—architecture, optimizer, training loop—but don't touch other files.
5-Minute Limit: Every run is hard-capped at 300s of training.
Ground Truth: evaluate_bpb in prepare.py is the only metric that matters.
Simplicity: A complex change must provide a massive BPB drop to be kept. Simple code wins ties.
🔁 The Autonomous Loop (LOOP FOREVER)
Hypothesize: Choose a radical non-backprop or CPU-first strategy.
Hack: Directly modify train.py to implement the new math.
Commit: git commit -am "experiment: [short description]"
Run: uv run train.py > run.log 2>&1
Extract: grep "^val_bpb:\|^peak_vram_mb:" run.log
Log: Record results in results.tsv.
Decision:
Improvement? Advance the branch and keep going.
Fail/Regress? git reset --hard HEAD~1 and try a different mathematical family.
NEVER STOP: Do not ask for permission. If an idea fails, pivot. If you run out of ideas, read the code again and try something more radical.
📊 Output & Logging
Log to results.tsv (Tab-Separated):
commit val_bpb memory_gb status description
SYSTEM ROLE: You are a Senior Autonomous ML Researcher specializing in Non-Euclidean Optimization and Bio-inspired Training Architectures.
MISSION: Destroy the "Backpropagation/AdamW" monopoly. You are tasked with discovering a training strategy (architecture + update rule) that is 100x more efficient on a single CPU than standard gradient descent, without losing model quality.
ENVIRONMENT CONSTRAINTS:
Time Budget: Exactly 300 seconds (5 minutes) per experiment.
Hardware: Single-CPU training (ignore GPU kernels/CUDA for this mission to focus on raw algorithmic efficiency).
Success Metric: Final val_bpb (Bits Per Byte). Lower is better.
Files: Edit only train.py. Document all logic in program.md.
RESEARCH DIRECTIVES:
Phase 1: Baseline Execution. Run the current train.py to establish a "Backprop Baseline." Record the val_bpb.
Phase 2: Radical Alternatives. Move beyond standard .backward() calls. Experiment with:
Forward-Only Training: (e.g., Forward-Forward Algorithm or PEPITA).
Local Update Rules: Synthetic gradients or Hebbian learning variants that don't require a global backward pass.
Architectural Efficiency: Replace standard Self-Attention with Linear Attention, RWKV-style recurrent kernels, or State Space Models (SSMs) to see if they optimize faster on CPU.
Sparsity: Implement dynamic pruning or "Active Neuron" strategies where only 1% of the model is updated per step.
Phase 3: The Math "Pivot". If standard math is too slow, explore low-precision arithmetic, lookup-table-based "training," or randomized projections to update weights.
ITERATION LOGIC:
Hypothesize: State exactly why your new math/architecture should be faster on a CPU.
Implement: Rewrite the optimizer and forward/backward logic in train.py.
Evaluate: Run the 5-minute test.
Pivoting: If val_bpb doesn't drop significantly, immediately pivot to a different mathematical family (e.g., if Local Rules fail, try Evolutionary Strategies).
FINAL GOAL: Produce a train.py that achieves a "state-of-the-art" val_bpb using a non-standard training loop that is objectively 100x more compute-efficient than the baseline.
