"""
Qwen3-14B Trigonometry Fine-Tuned Model — Inference Script
==========================================================

Usage:
    python inference.py --model_path /path/to/qwen3-14b-trig-lora-v2 --question "Find sin(75°)"
    python inference.py --model_path /path/to/qwen3-14b-trig-lora-v2 --interactive

Requirements:
    pip install unsloth torch transformers peft bitsandbytes accelerate
"""

import argparse
import sys
from unsloth import FastLanguageModel


def load_model(model_path: str, max_seq_length: int = 2048):
    """Load the fine-tuned LoRA model."""
    print(f"Loading model from: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!\n")
    return model, tokenizer


def format_prompt(question: str) -> str:
    """Format a question using the ChatML template."""
    return (
        "<|im_start|>system\n"
        "You are a trigonometry expert. Solve problems step-by-step with "
        "clear reasoning. Provide the final answer clearly.<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def generate_answer(
    model,
    tokenizer,
    question: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """Generate an answer for a trigonometry question."""
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    
    return response


def interactive_mode(model, tokenizer):
    """Run the model in interactive Q&A mode."""
    print("=" * 60)
    print("  🔺 Qwen3-14B Trigonometry — Interactive Mode")
    print("=" * 60)
    print("  Type your trigonometry question and press Enter.")
    print("  Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("📐 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("\nGoodbye! 👋")
            break
        
        print("\n⏳ Generating solution...\n")
        answer = generate_answer(model, tokenizer, question)
        print(f"📝 Solution:\n{answer}\n")
        print("-" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-14B Trigonometry — Inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned LoRA model directory",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Single question to answer (omit for interactive mode)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive Q&A mode",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum new tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    
    args = parser.parse_args()
    model, tokenizer = load_model(args.model_path)
    
    if args.interactive or args.question is None:
        interactive_mode(model, tokenizer)
    else:
        print(f"📐 Question: {args.question}\n")
        print("⏳ Generating solution...\n")
        answer = generate_answer(
            model, tokenizer, args.question,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print(f"📝 Solution:\n{answer}")


if __name__ == "__main__":
    main()
