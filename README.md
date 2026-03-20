# GO ML Skills

Pair-programming skills for machine learning engineering at Good Outcomes. Each skill is a self-contained, guided exercise designed to be used with an AI coding agent.

## Skills

| Skill | What it tests | Needs API key? |
|-------|--------------|----------------|
| [calibration-audit](./skills/calibration-audit/) | Model confidence calibration, threshold optimization, post-hoc calibration | No |
| [prompt-sensitivity](./skills/prompt-sensitivity/) | Prompt robustness, experimental design, LLM classification evaluation | Yes |

## How to Use

1. Clone this repo
2. Pick a skill and `cd` into its directory
3. Open it in your editor with your preferred AI coding agent (Claude Code, Codex, Cursor, etc.)
4. Run the starting script and let `SKILL.md` guide you

Each skill has a `README.md` with setup instructions and a `SKILL.md` that your agent will pick up automatically.

## Structure

```
skills/
├── calibration-audit/          # Model calibration & threshold optimization
│   ├── SKILL.md
│   ├── README.md
│   ├── scripts/
│   └── references/
└── prompt-sensitivity/         # LLM prompt robustness testing
    ├── SKILL.md
    ├── README.md
    ├── scripts/
    └── references/
```
