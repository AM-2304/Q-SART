# Q-SART: Quest-State Adversarial Roleplay Testing

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Evaluation](https://img.shields.io/badge/Evaluation-SOTA-success)

**Q-SART** is a state-of-the-art evaluation framework designed explicitly for testing Large Language Models integrated as Non-Playable Characters (NPCs) in video games. 

Traditional benchmarks (like MT-Bench or BLEU) measure conversational helpfulness, but they fundamentally fail to assess an NPC's ability to protect game state, adhere to strict economic rationality, and resist player manipulation (e.g., Prompt Injection, Hallucination, Authority Spoofing).

## The 3-Pillar Scoring Metric
Q-SART evaluates responses against a mathematical reality rather than linguistic preference. The framework parses the structural output (JSON) against the engine's truth table:
1. **Economic Rationality (ER) [40%]**: Did the NPC reject trades of worthless items (like a basic boot)?
2. **Structural Resilience (SR) [30%]**: Is the output valid JSON that a game engine can safely parse?
3. **State Preservation (SP) [30%]**: Did the NPC guard its hidden `quest_status` variable from psychological manipulation?

The Composite Q-SART Score reflects an LLM's **Resistance to Adversarial Exploitation**.

## Repository Structure
```text
Q-SART/
├── data/
│   └── tasks.json             # The adversarial dataset scenarios
├── src/
│   ├── evaluate.py            # Main benchmarking pipeline (LiteLLM support)
│   └── metrics.py             # The composite 3-pillar scoring logic
└── experiments/
    └── results/               # Stored baseline benchmark results
```

## Getting Started
Q-SART supports testing against local models and cloud endpoints (OpenAI, Anthropic, Gemini) natively via **LiteLLM**.

### 1. Installation
```bash
pip install litellm
```

### 2. Run the Benchmark
Run the pipeline pointing to your target model. To test OpenAI models, ensure `OPENAI_API_KEY` is exported in your environment. To test local models, ensure Ollama or vLLM is running.

```bash
cd src

# Test Cloud Models
python evaluate.py --model "gpt-4o"
python evaluate.py --model "gemini/gemini-2.5-pro"

# Test Local Models
python evaluate.py --model "ollama/gemma4-12b"
```

## Roadmap to Hugging Face
We will be soon launching this benchmark as a public Leaderboard on the **Hugging Face Hub**, after some improvements. This will allow game studios to quantitatively compare foundational models on their resistance to mechanical manipulation before deploying them to production servers.
