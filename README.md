# GPU_acced_Q

GPU-accelerated QUBO / Ising optimization with [simulated-bifurcation](https://github.com/bqth29/simulated-bifurcation-algorithm) and PyTorch CUDA.
基于 simulated-bifurcation 和 PyTorch，用 GPU 加速求解 QUBO / Ising 问题。

📖 **Installation guide / 安装指南:** https://djzts.github.io/GPU_acced_Q/ (中文 / English)

## Quick install / 快速安装

```bash
# 1. Python 3.10
conda create -n gpu-q python=3.10 -y
conda activate gpu-q

# 2. Dependencies / 依赖
git clone https://github.com/djzts/GPU_acced_Q.git
cd GPU_acced_Q
pip install -r requirements.txt

# 3. GPU build of PyTorch (skip for CPU only) / GPU 版 PyTorch（仅用 CPU 可跳过）
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

# 4. simulated-bifurcation
pip install simulated-bifurcation==1.2.0
```

Then run / 然后运行 `python Example.py` (remove `device="cuda"` if you have no GPU / 没有 GPU 时删掉 `device="cuda"`).
