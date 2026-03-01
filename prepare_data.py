"""
============================================================
Agentic World — Data Preparation (Policy Edition)
============================================================
Builds fine-tuning data for a browser-agent model.

Input:  Parsed sessions + policies from PosthogAgent/data/
Output: train.jsonl + eval.jsonl in chat completion format

Each example becomes:
    system: browser agent policy prompt
    user:   website description + user profile JSON
    assistant: policy JSON (action_sequence + behavioral params)

The finetuned model will output executable policy JSONs that
can be fed directly to the BehavioralAgent (AgentQL + Playwright).

Usage:
    python prepare_data.py --policies-dir ../PosthogAgent/data/policies \
                           --parsed-dir ../PosthogAgent/data/parsed \
                           --output ./ --eval-count 7
============================================================
"""

import os
import json
import glob
import random
import argparse

# ============================================================
# CONFIG
# ============================================================

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

SITE_DESCRIPTION = """Website: https://fun-city-xi.vercel.app/
Description: FunCity is a Reddit-style NYC discovery board where users browse posts organized by NYC boroughs (The Bronx, Brooklyn, Manhattan, Queens, Staten Island) and topics (Art & Culture, Food & Eats, Hidden Gems, Nature & Parks, Nightlife). The homepage shows a feed of user posts sorted by Hot, New, or Top tabs. Each post card displays a borough tag, username, timestamp, title, body preview, upvote/downvote arrows with score, and comment count. The right sidebar contains borough filter buttons, topic filter buttons, and a Trending section showing top 5 posts. Users can click posts to see the full post detail page with a comments thread (each comment has upvote/downvote). There is a Sign Up button (top right) with a modal collecting username, password, age group, country, and NYC familiarity. Logged-in users see a "+ New Post" button and can comment and vote."""

# ============================================================
# PARSE ARGS
# ============================================================

parser = argparse.ArgumentParser(description="Prepare fine-tuning data from policies + parsed sessions")
parser.add_argument("--policies-dir", type=str,
                    default="../PosthogAgent/data/policies",
                    help="Directory of policy_*.json files")
parser.add_argument("--parsed-dir", type=str,
                    default="../PosthogAgent/data/parsed",
                    help="Directory of parsed_*.json files (for user profiles)")
parser.add_argument("--output", type=str, default="./", help="Output directory for JSONL files")
parser.add_argument("--eval-count", type=int, default=7, help="Number of eval examples")
parser.add_argument("--seed", type=int, default=42, help="Random seed for train/eval split")
args = parser.parse_args()

# ============================================================
# LOAD AND PROCESS FILES
# ============================================================

print(f"📂 Loading policies from {args.policies_dir}")
print(f"📂 Loading parsed sessions from {args.parsed_dir}")

policy_files = sorted(glob.glob(os.path.join(args.policies_dir, "policy_*.json")))
if not policy_files:
    print(f"❌ No policy_*.json files found in {args.policies_dir}")
    exit(1)

examples = []
for policy_path in policy_files:
    session_id = os.path.basename(policy_path).replace("policy_", "").replace(".json", "")

    # Load policy
    with open(policy_path, "r") as f:
        policy = json.load(f)

    # Validate policy has an action_sequence
    if not policy.get("action_sequence") or len(policy["action_sequence"]) < 1:
        print(f"   ⚠️  Skipping {session_id} (no action_sequence)")
        continue

    # Load user profile from parsed session
    parsed_path = os.path.join(args.parsed_dir, f"parsed_{session_id}.json")
    user_profile = {}
    if os.path.exists(parsed_path):
        with open(parsed_path, "r") as f:
            parsed = json.load(f)
        user_profile = parsed.get("user_profile", {})

    # Build user message: website description + user profile
    user_content = SITE_DESCRIPTION
    if user_profile:
        profile_summary = {
            "age_group": user_profile.get("age_group", "unknown"),
            "country": user_profile.get("country", "unknown"),
            "nyc_familiarity": user_profile.get("nyc_familiarity", "unknown"),
            "device_type": user_profile.get("device_type", "unknown"),
            "browser": user_profile.get("browser", "unknown"),
            "os": user_profile.get("os", "unknown"),
        }
        user_content += f"\n\nUser Profile:\n{json.dumps(profile_summary, indent=2)}"

    # Assistant response: the policy JSON
    assistant_content = json.dumps(policy, indent=2)

    example = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }
    examples.append(example)
    n_actions = len(policy["action_sequence"])
    style = policy.get("navigation_style", "?")
    print(f"   ✅ {session_id} — {n_actions} actions, style={style}")

print(f"\n📊 Total examples: {len(examples)}")

# ============================================================
# TRAIN/EVAL SPLIT
# ============================================================

random.seed(args.seed)
random.shuffle(examples)

eval_count = min(args.eval_count, len(examples) // 4)  # Max 25% for eval
eval_examples = examples[:eval_count]
train_examples = examples[eval_count:]

print(f"   Train: {len(train_examples)}")
print(f"   Eval:  {len(eval_examples)}")

# ============================================================
# WRITE JSONL
# ============================================================

train_path = os.path.join(args.output, "train.jsonl")
eval_path = os.path.join(args.output, "eval.jsonl")

with open(train_path, "w") as f:
    for ex in train_examples:
        f.write(json.dumps(ex) + "\n")

with open(eval_path, "w") as f:
    for ex in eval_examples:
        f.write(json.dumps(ex) + "\n")

print(f"\n💾 Saved:")
print(f"   {train_path} ({os.path.getsize(train_path) / 1024:.1f} KB)")
print(f"   {eval_path} ({os.path.getsize(eval_path) / 1024:.1f} KB)")
print(f"\nReady for: python finetune_job.py")
