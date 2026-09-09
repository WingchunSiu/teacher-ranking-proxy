#!/usr/bin/env python3
"""
Detailed analysis of recovery papers.
Shows grep hits with full context for manual review.
"""

import json
import os
import subprocess
from pathlib import Path

def key_to_filename(key):
    return key.replace(':', '_').replace('/', '_')

def get_text_path(key, fulltext_dir):
    filename = key_to_filename(key)
    return os.path.join(fulltext_dir, f"{filename}.txt")

def read_text(text_path):
    if not os.path.exists(text_path):
        return None
    try:
        with open(text_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except:
        return None

def grep_for_pattern(text_path, pattern):
    """Get grep output with context"""
    try:
        result = subprocess.run(
            ['grep', '-n', '-i', '-A3', '-B3', pattern, text_path],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout if result.stdout else None
    except:
        return None

def print_paper_analysis(key, fulltext_dir):
    """Print detailed analysis for one paper"""
    text_path = get_text_path(key, fulltext_dir)
    text = read_text(text_path)

    if not text or len(text) < 3000:
        print(f"  RETRIEVAL FAILURE")
        return

    print(f"\n{'='*70}")
    print(f"Paper: {key}")
    print(f"{'='*70}")

    # Look for key patterns
    key_patterns = [
        'ablat',  # Ablation
        'teacher',  # Teacher model
        'student',  # Student model
        'trajectories from',  # Data generation
        'data source',  # Data source
        'sft',  # SFT
    ]

    print("\nGREP RESULTS:")
    for pattern in key_patterns:
        grep_out = grep_for_pattern(text_path, pattern)
        if grep_out:
            print(f"\n--- {pattern.upper()} ---")
            lines = grep_out.split('\n')[:8]  # Show first 8 lines
            for line in lines[:8]:
                if line.strip():
                    print(f"  {line[:120]}")

def main():
    litmine_dir = "/data/cat/ws/frwe188h-otagent/OpenThoughts-Agent-trp/data/teacher_ranking_proxy/litmine"
    fulltext_dir = os.path.join(litmine_dir, "work", "state", "fulltext")
    recovery_dir = os.path.join(litmine_dir, "work", "state", "recovery_agents")

    assignment_file = os.path.join(recovery_dir, "rec_005.json")

    with open(assignment_file, 'r') as f:
        keys = json.load(f)

    # Just analyze first few
    for key in keys[:5]:
        print_paper_analysis(key, fulltext_dir)

if __name__ == "__main__":
    main()
