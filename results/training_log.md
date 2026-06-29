# 📊 Training Log — Qwen3-14B Trigonometry LoRA

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen/Qwen3-14B |
| Quantization | 4-bit (bitsandbytes NF4) |
| LoRA Method | Unsloth optimized |
| Trainable Parameters | 64M / 14.8B (0.43%) |
| Learning Rate | 1e-4 |
| Batch Size | 8 (1 × 8 gradient accumulation) |
| Max Sequence Length | 2048 tokens |
| Epochs | 1 |
| Total Steps | 625 |
| Platform | Google Colab (T4 GPU, 15GB VRAM) |

## Dataset Summary

- **Source:** NuminaMath-CoT (HuggingFace: AI-MO/NuminaMath-CoT)
- **Filter:** Trigonometry-related problems
- **Total Filtered:** 25,204 problems
- **Problem Sources:** olympiads, aops_forum, amc_aime, math

| Split | Count |
|-------|-------|
| Train | 5,000 |
| Validation | 500 |
| Test | 500 |

## Training Metrics

| Step | Training Loss | Validation Loss | Δ Val Loss | Cumulative Δ |
|:----:|:------------:|:---------------:|:----------:|:------------:|
| 75   | 0.497        | 0.487           | —          | —            |
| 150  | 0.472        | 0.476           | −0.011     | −0.011       |
| 225  | 0.455        | 0.472           | −0.004     | −0.015       |
| 300  | 0.435        | 0.469           | −0.003     | −0.018       |
| 375  | 0.474        | 0.467           | −0.002     | −0.020       |
| 450  | 0.469        | 0.466           | −0.001     | −0.021       |
| 525  | 0.418        | 0.465           | −0.001     | −0.022       |
| 600  | 0.411        | 0.464           | −0.001     | −0.023       |

## Analysis

### Convergence Behavior
- **Total validation loss reduction:** 4.72% (0.487 → 0.464)
- **Training loss reduction:** 17.3% (0.497 → 0.411)
- **Loss trajectory:** Monotonically decreasing validation loss across all checkpoints
- **Overfitting indicator:** None — validation loss continues to decrease at final checkpoint

### Training Stability
- The model exhibited excellent training stability despite being trained across multiple Colab sessions with checkpoint resuming
- Gradient accumulation (8 steps) effectively simulated a larger batch size, contributing to stable loss curves
- The slight fluctuations in training loss (e.g., step 375: 0.474) are expected with gradient accumulation and do not indicate instability

### Efficiency
- Training on free-tier T4 GPU demonstrates the practical accessibility of LoRA fine-tuning
- Only 0.43% of parameters required updates, making the approach memory-efficient
- 4-bit quantization enabled fitting a 14.8B parameter model within 15GB VRAM constraints
