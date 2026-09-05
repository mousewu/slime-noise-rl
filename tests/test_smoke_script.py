import subprocess
from pathlib import Path

from noise_rl.config import load_config

ROOT = Path(__file__).parents[1]


def test_two_gpu_smoke_script_and_short_budget_config():
    script = ROOT / "scripts/smoke_2gpu.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text(encoding="utf-8")
    for fragment in (
        '"${TASK_PROJECT_DIR}[alfworld,tracking]"',
        "--gpus 2",
        "--tensor-parallel 2",
        "--engine-gpus 2",
        "--batch-size 1",
        "--dry-run",
    ):
        assert fragment in text
    assert "pip install slime" not in text.lower()
    assert "rm -rf" not in text

    config = load_config(ROOT / "configs/smoke_2gpu.yaml")
    assert config.group_size == 4
    assert config.scenarios == 2
    assert config.max_turns == 8
    assert config.max_context_tokens == 4096
