#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = ROOT / "SKILL"
LEGACY_CODEX_SKILL_ROOT = ROOT / ".codex" / "skills"
EXPECTED = {
    "tilemem-environment-setup": [
        "tools/tilemem doctor",
        "tools/tilemem verify --quick",
        "Do not download large checkpoints",
    ],
    "tilemem-acceleration-path": [
        "tools/tilemem checkpoint prepare",
        "tools/report_tilepo_ablation",
        "Do not benchmark TilePO against KT with different expert budgets",
    ],
    "tilemem-backend-precision-path": [
        "TileMEM owns",
        "External backend owners provide",
        "Do not claim FP8/F6/F4 model-quality safety",
    ],
}


def _frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    assert match, "missing YAML frontmatter"
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip():
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def test_tilemem_agent_skills_are_packaged_and_discoverable() -> None:
    assert not LEGACY_CODEX_SKILL_ROOT.exists(), LEGACY_CODEX_SKILL_ROOT
    for skill_name, required_phrases in EXPECTED.items():
        skill_dir = SKILL_ROOT / skill_name
        skill_md = skill_dir / "SKILL.md"
        openai_yaml = skill_dir / "agents" / "openai.yaml"
        assert skill_md.exists(), skill_md
        assert openai_yaml.exists(), openai_yaml

        text = skill_md.read_text()
        fields = _frontmatter(text)
        assert fields["name"] == skill_name
        assert fields["description"].startswith("Use when")
        assert "TODO" not in text
        for phrase in required_phrases:
            assert phrase in text, f"{phrase!r} missing from {skill_md}"

        metadata = openai_yaml.read_text()
        assert f"Use ${skill_name}" in metadata
        assert "Use -" not in metadata


def main() -> None:
    test_tilemem_agent_skills_are_packaged_and_discoverable()
    print("TileMEM agent skill tests passed")


if __name__ == "__main__":
    main()
