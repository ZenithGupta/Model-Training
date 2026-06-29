# 🔍 Comparison Report: Fine-Tuned vs Base Model

## Test Conditions

| Parameter | Value |
|-----------|-------|
| Max Output Tokens | 2,048 |
| Prompt Template | Identical for both models |
| Test Problems | 5 held-out trigonometry problems |
| Quantization | 4-bit (both models) |
| Hardware | T4 GPU (Google Colab) |

## Overall Results

| Metric | Fine-Tuned | Base |
|--------|:----------:|:----:|
| **Problems Answered** | 5/5 ✅ | 0/5 ❌ |
| **Correct Answers** | 5/5 ✅ | 0/5 ❌ |
| **Structured Output** | Yes | No |
| **Token Efficiency** | High | Wasted |

**Win Rate: 5/5 (100%) in favor of the fine-tuned model**

---

## Detailed Analysis

### Problem 1: Exact Trigonometric Values
**Task:** Compute exact value of a trigonometric expression

| Aspect | Fine-Tuned ✅ | Base ❌ |
|--------|:------------:|:------:|
| Response format | Numbered steps | Endless `<think>` block |
| Final answer | Correct | Never produced |
| Tokens used | ~400 | 2048 (exhausted) |

### Problem 2: Trigonometric Identity Proof
**Task:** Prove a trigonometric identity

| Aspect | Fine-Tuned ✅ | Base ❌ |
|--------|:------------:|:------:|
| Response format | Clear proof steps | Internal reasoning only |
| Completeness | Full proof | Incomplete |
| Tokens used | ~600 | 2048 (exhausted) |

### Problem 3: Triangle Problem (Law of Sines/Cosines)
**Task:** Find unknown sides/angles in a triangle

| Aspect | Fine-Tuned ✅ | Base ❌ |
|--------|:------------:|:------:|
| Response format | Step-by-step solution | Stuck in thinking |
| Final answer | Correct | Never produced |
| Tokens used | ~500 | 2048 (exhausted) |

### Problem 4: Limit Problem (Answer: −50)
**Task:** Evaluate a limit involving trigonometric functions

| Aspect | Fine-Tuned ✅ | Base ❌ |
|--------|:------------:|:------:|
| Response format | Clean derivation | Internal monologue |
| Final answer | **−50** ✓ | Never produced |
| Tokens used | ~700 | 2048 (exhausted) |

### Problem 5: Trigonometric Equation
**Task:** Solve a trigonometric equation

| Aspect | Fine-Tuned ✅ | Base ❌ |
|--------|:------------:|:------:|
| Response format | Methodical solution | Never exits `<think>` |
| Final answer | Correct | Never produced |
| Tokens used | ~550 | 2048 (exhausted) |

---

## Root Cause Analysis

### Why the Base Model Fails

The base Qwen3-14B model uses an extended thinking format (`<think>...</think>` tags) before producing a user-visible response. For complex mathematical problems:

1. **Token budget exhaustion:** The model spends all 2,048 tokens in the `<think>` block, deliberating internally
2. **No visible output:** Since the `</think>` closing tag is never reached within the token limit, no user-facing answer is ever produced
3. **Verbose internal reasoning:** The base model's chain-of-thought is unstructured and repetitive, leading to token waste

### Why the Fine-Tuned Model Succeeds

LoRA fine-tuning on structured trigonometry solutions teaches the model to:

1. **Skip excessive internal deliberation** — Proceed directly to solution generation
2. **Use numbered step format** — Produce clean, readable, structured solutions
3. **Be concise** — Deliver correct answers within a fraction of the token budget
4. **Prioritize user-facing output** — Generate the answer the user actually needs

---

## Conclusion

The fine-tuned model represents a **qualitative transformation** in usability for trigonometry problem-solving. While the base model has the mathematical knowledge embedded in its weights, it fails to deliver that knowledge within practical token constraints. The LoRA fine-tuning bridges this gap, producing a model that is:

- ✅ **Reliable** — Answers every problem
- ✅ **Correct** — 100% accuracy on test set
- ✅ **Efficient** — Uses ~25-35% of available tokens
- ✅ **Structured** — Clean, numbered, step-by-step format
- ✅ **Production-ready** — Suitable for end-user deployment
