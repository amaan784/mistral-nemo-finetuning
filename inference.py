"""
============================================================
Agentic World — Inference Pipeline
============================================================
Loads the fine-tuned Nemotron Nano 9B v2 adapter and generates
behavioral profiles for any website, then formats the output
as AgentQL-compatible action plans.

Usage:
    python inference.py --url "https://example.com" --description "An e-commerce site..."
    python inference.py --interactive  # Interactive mode
    
Requires: The trained adapter in outputs/nemotron-behavioral-lora/
============================================================
"""

import argparse
import json
import os
import torch
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

BASE_MODEL = "unsloth/NVIDIA-Nemotron-Nano-9B-v2-bnb-4bit"
ADAPTER_DIR = "outputs/nemotron-behavioral-lora"
MAX_SEQ_LENGTH = 4096

SYSTEM_PROMPT = """You are a behavioral simulation model. Given a website description, generate a detailed behavioral profile describing how a user would interact with the website. Include: navigation pattern, reading behavior, engagement style, interaction speed, content preferences, typing behavior, feature discovery, and session flow with specific timings."""

# ============================================================
# LOAD MODEL + ADAPTER
# ============================================================

def load_model():
    """Load base model with fine-tuned LoRA adapter."""
    print(f"Loading base model: {BASE_MODEL}")
    print(f"Loading adapter: {ADAPTER_DIR}")

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_DIR,  # This loads base + adapter
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)
    print(f"Model loaded and ready for inference")
    return model, tokenizer


# ============================================================
# GENERATE BEHAVIORAL PROFILE
# ============================================================

def generate_behavioral_profile(
    model,
    tokenizer,
    url: str,
    description: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a behavioral profile for a given website."""

    user_content = f"Website: {url}\nDescription: {description}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response


# ============================================================
# CONVERT TO AGENTQL ACTION PLAN
# ============================================================

def behavioral_profile_to_agentql_plan(profile: str, url: str) -> dict:
    """
    Parse the behavioral profile and structure it as an AgentQL
    action plan. This is the bridge between the fine-tuned model
    output and AgentQL execution.
    
    AgentQL expects:
    - A target URL
    - A sequence of actions with types and parameters
    - Timing constraints
    """

    action_plan = {
        "target_url": url,
        "generated_at": datetime.now().isoformat(),
        "behavioral_profile": profile,
        "agentql_config": {
            "headless": False,  # Show browser for demo
            "viewport": {"width": 1920, "height": 1080},
            "timeout_ms": 30000,
        },
        # The behavioral profile text is passed directly to AgentQL
        # as the instruction set. AgentQL's LLM layer interprets
        # the behavioral descriptions and executes accordingly.
        "execution_prompt": f"""You are browsing the website at {url}.

Follow this behavioral profile EXACTLY. Do not deviate from the 
described patterns. The timings, hesitations, scroll speeds, and 
interaction patterns must match what is described below.

BEHAVIORAL PROFILE:
{profile}

Execute each step of the session flow described above. Between 
actions, wait the specified durations. Scroll at the specified 
speeds. Hesitate before clicks as described. Follow the exact 
navigation pattern and content preferences outlined.""",
    }

    return action_plan


# ============================================================
# BATCH PROCESSING
# ============================================================

def process_websites_from_file(model, tokenizer, input_file: str, output_file: str):
    """
    Process multiple websites from a JSON file.
    Input format: [{"url": "...", "description": "..."}, ...]
    """
    with open(input_file) as f:
        websites = json.load(f)

    results = []
    for i, site in enumerate(websites):
        print(f"\n[{i+1}/{len(websites)}] Processing: {site['url']}")
        profile = generate_behavioral_profile(
            model, tokenizer, site["url"], site["description"]
        )
        plan = behavioral_profile_to_agentql_plan(profile, site["url"])
        results.append(plan)
        print(f"  Generated {len(profile)} chars")

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} action plans to {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate behavioral profiles for websites")
    parser.add_argument("--url", type=str, help="Website URL")
    parser.add_argument("--description", type=str, help="Website description")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", type=str, help="Path to JSON file with multiple websites")
    parser.add_argument("--output", type=str, default="action_plans.json", help="Output file for batch mode")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    model, tokenizer = load_model()

    if args.batch:
        process_websites_from_file(model, tokenizer, args.batch, args.output)

    elif args.interactive:
        print("\nInteractive mode -- Enter website details (Ctrl+C to exit)")
        while True:
            try:
                print("\n" + "=" * 60)
                url = input("URL: ").strip()
                desc = input("Description: ").strip()
                if not url or not desc:
                    print("Both URL and description are required.")
                    continue

                print("\nGenerating behavioral profile...")
                profile = generate_behavioral_profile(
                    model, tokenizer, url, desc,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )

                print(f"\n{'='*60}")
                print("BEHAVIORAL PROFILE")
                print(f"{'='*60}")
                print(profile)

                plan = behavioral_profile_to_agentql_plan(profile, url)
                plan_file = f"plan_{url.replace('https://', '').replace('/', '_')[:30]}.json"
                with open(plan_file, "w") as f:
                    json.dump(plan, f, indent=2)
                print(f"\nAction plan saved to {plan_file}")

            except KeyboardInterrupt:
                print("\n\nExiting")
                break

    elif args.url and args.description:
        print(f"\nGenerating behavioral profile for {args.url}...")
        profile = generate_behavioral_profile(
            model, tokenizer, args.url, args.description,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        print(f"\n{'='*60}")
        print("BEHAVIORAL PROFILE")
        print(f"{'='*60}")
        print(profile)

        plan = behavioral_profile_to_agentql_plan(profile, args.url)
        with open(args.output, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"\nAction plan saved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
