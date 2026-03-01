"""
============================================================
Agentic World — Inference Pipeline (Policy Generation)
============================================================
Loads a fine-tuned LoRA adapter and generates a structured
JSON policy that an autonomous browser agent can execute
using AgentQL + Playwright.

Usage:
    # Generate a policy (print JSON)
    python inference.py --url https://fun-city-xi.vercel.app/ \
        --description "A Reddit-style NYC discovery board..."

    # Generate + execute in browser
    python inference.py --url https://fun-city-xi.vercel.app/ \
        --description "..." --execute

    # Generate with user profile
    python inference.py --url ... --description "..." \
        --user-profile '{"age_group":"25-34","country":"US"}'

    # Save policy to file
    python inference.py --url ... --description "..." --output policy.json

    # Use a specific policy file (skip generation)
    python inference.py --policy policy.json --execute
============================================================
"""

import os
import sys
import json
import argparse
import torch
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
DEFAULT_ADAPTER = "outputs/mistral-nemo-behavioral-lora"

SYSTEM_PROMPT = """You are a browser automation agent policy generator. Given a website description and a user profile, output a structured JSON policy that an autonomous browser agent can execute using AgentQL and Playwright.

The policy must contain:
- session_target_duration_s: estimated session length in seconds
- navigation_style: one of "exploratory", "focused", "scattered", "linear"
- browsing_speed: one of "fast", "medium", "slow"
- feed_behavior: how to scan the feed (scroll pattern, sort preference, max posts)
- post_interaction: comment/vote probabilities, typing speed, read time
- subreddit_exploration: whether to explore categories, which ones
- auth_behavior: whether to sign up or log in, and when
- engagement_arc: early/mid/late session behavior
- action_sequence: ordered list of actions the agent should execute

Valid actions for action_sequence:
- scan_feed: scroll through the homepage feed
- open_post: click on a post to view details
- signup: create a new account
- login: log in with existing credentials
- write_comment: write and submit a comment
- vote_on_post: upvote or downvote
- return_to_feed: navigate back to homepage
- browse_subreddit: explore a category/subreddit
- open_related_post: click on a trending/related post
- create_post: create a new post

Output ONLY valid JSON. No markdown, no explanation."""

# ============================================================
# PARSE ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Generate browser agent policies and optionally execute them")
parser.add_argument("--url", type=str, help="Website URL")
parser.add_argument("--description", type=str, help="Website description")
parser.add_argument("--user-profile", type=str, default="{}", help="User profile JSON string")
parser.add_argument("--adapter", type=str, default=DEFAULT_ADAPTER, help="Path to LoRA adapter")
parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model name")
parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens")
parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
parser.add_argument("--output", type=str, help="Save policy JSON to file")
parser.add_argument("--policy", type=str, help="Load existing policy JSON (skip generation)")
parser.add_argument("--execute", action="store_true", help="Execute the policy in a real browser via AgentQL")
parser.add_argument("--sandbox-url", type=str, default=None,
                    help="Sandbox URL for execution (defaults to --url)")
args = parser.parse_args()

# ============================================================
# POLICY GENERATION (unless --policy is provided)
# ============================================================

policy = None

if args.policy:
    # Load existing policy from file
    print(f"📂 Loading policy from {args.policy}")
    with open(args.policy) as f:
        policy = json.load(f)
    print(f"  Actions: {len(policy.get('action_sequence', []))}")
    print(f"  Style: {policy.get('navigation_style')}, Speed: {policy.get('browsing_speed')}")

else:
    # Generate policy using finetuned model
    if not args.url or not args.description:
        print("Usage:")
        print("  python inference.py --url URL --description 'Site description...'")
        print("  python inference.py --policy policy.json --execute")
        sys.exit(1)

    print("=" * 60)
    print("AGENTIC WORLD — POLICY GENERATION")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n📦 Loading base model: {args.model}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load LoRA adapter if it exists
    if os.path.exists(args.adapter):
        print(f"🔧 Loading LoRA adapter: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"🔀 Merging adapter into base model for fast inference...")
        model = model.merge_and_unload()
        print(f"✅ Adapter merged")
    else:
        print(f"⚠️  No adapter found at {args.adapter}, using base model")

    model.eval()
    print(f"✅ Model ready for inference\n")

    # Build user prompt
    user_content = f"Website: {args.url}\nDescription: {args.description}"
    try:
        user_profile = json.loads(args.user_profile)
        if user_profile:
            user_content += f"\n\nUser Profile:\n{json.dumps(user_profile, indent=2)}"
    except json.JSONDecodeError:
        print(f"⚠️  Invalid --user-profile JSON, ignoring")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    print(f"⏳ Generating policy for {args.url}...")

    input_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=0.9,
            do_sample=True,
        )

    raw_response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    # Parse JSON from model output
    try:
        policy = json.loads(raw_response)
    except json.JSONDecodeError:
        # Try to extract JSON from the response (model might add extra text)
        import re
        json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if json_match:
            try:
                policy = json.loads(json_match.group())
            except json.JSONDecodeError:
                print(f"\n❌ Could not parse policy JSON from model output:")
                print(raw_response[:2000])
                sys.exit(1)
        else:
            print(f"\n❌ Model did not produce valid JSON:")
            print(raw_response[:2000])
            sys.exit(1)

    # Print policy summary
    print(f"\n{'='*60}")
    print("GENERATED POLICY")
    print("=" * 60)
    print(json.dumps(policy, indent=2))
    print(f"\n  Actions: {len(policy.get('action_sequence', []))}")
    print(f"  Style: {policy.get('navigation_style')}, Speed: {policy.get('browsing_speed')}")
    print(f"  Sequence: {policy.get('action_sequence', [])}")

# ============================================================
# SAVE POLICY
# ============================================================

if args.output and policy:
    with open(args.output, "w") as f:
        json.dump(policy, f, indent=2)
    print(f"\n💾 Policy saved to {args.output}")

# ============================================================
# EXECUTE POLICY (optional)
# ============================================================

if args.execute and policy:
    sandbox_url = args.sandbox_url or args.url
    if not sandbox_url:
        print("❌ --execute requires --url or --sandbox-url")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("EXECUTING POLICY IN BROWSER")
    print("=" * 60)
    print(f"  Target: {sandbox_url}")
    print(f"  Actions: {policy.get('action_sequence', [])}")

    # Try to import BehavioralAgent from PosthogAgent
    posthog_agent_path = os.path.join(os.path.dirname(__file__), "..", "PosthogAgent")
    sys.path.insert(0, os.path.abspath(posthog_agent_path))

    try:
        from pipeline.stage5_execute import BehavioralAgent
        from feedback.session_logger import SessionLogger
    except ImportError:
        # Fallback: look in sibling directory
        alt_path = os.path.join(os.path.dirname(__file__), "..", "PosthogAgent")
        sys.path.insert(0, os.path.abspath(alt_path))
        try:
            from pipeline.stage5_execute import BehavioralAgent
            from feedback.session_logger import SessionLogger
        except ImportError as e:
            print(f"\n❌ Cannot import BehavioralAgent: {e}")
            print("   Make sure PosthogAgent is in a sibling directory.")
            print("   Or install agentql + playwright in your environment.")
            sys.exit(1)

    # Get Mistral API key for comment generation
    mistral_key = os.environ.get("MISTRAL_API_KEY", "")
    if not mistral_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            mistral_key = os.environ.get("MISTRAL_API_KEY", "")
        except ImportError:
            pass

    session_id = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = SessionLogger(session_id=session_id)

    agent = BehavioralAgent(
        policy=policy,
        sandbox_url=sandbox_url,
        mistral_api_key=mistral_key,
        session_logger=logger,
    )

    print(f"\n� Launching agent...")
    agent.run()

    # Save agent log
    log_path = f"agent_log_{session_id}.json"
    logger.save(log_path)
    summary = logger.get_summary()

    print(f"\n{'='*60}")
    print("EXECUTION COMPLETE")
    print("=" * 60)
    print(f"  Duration: {summary['total_duration_s']}s")
    print(f"  Actions: {summary['total_actions']} "
          f"({summary['successful_actions']} ok, {summary['failed_actions']} failed)")
    print(f"  Log: {log_path}")

elif not args.execute and policy:
    print(f"\nTo execute this policy in a browser, re-run with --execute")
    print(f"  python inference.py --policy {args.output or 'policy.json'} --execute --url {args.url or 'TARGET_URL'}")
