"""The gate in front of the test split (src/eval/freeze.py), on throwaway git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.eval import freeze
from src.eval.freeze import NotFrozen


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
        encoding="utf-8",
    )
    return done.stdout.strip()


def commit(repo: Path, path: str, text: str) -> str:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    git(repo, "add", path)
    git(repo, "commit", "-q", "-m", f"change {path}")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    commit(tmp_path, "src/model.py", "x = 1\n")
    commit(tmp_path, "configs/config.yaml", "alpha: 0.5\n")
    return tmp_path


def test_without_the_tag_the_split_stays_closed(repo):
    with pytest.raises(NotFrozen, match="does not exist"):
        freeze.require_freeze(repo)


def test_a_frozen_clean_tree_opens(repo):
    git(repo, "tag", freeze.FREEZE_TAG)
    out = freeze.require_freeze(repo)
    assert out["commit"] == git(repo, "rev-parse", "HEAD")
    assert out["tag_commit"] == out["commit"]
    assert out["changed_since_freeze"] == []


def test_a_dirty_tree_is_refused(repo):
    git(repo, "tag", freeze.FREEZE_TAG)
    (repo / "scratch.py").write_text("print(1)\n", encoding="utf-8")
    with pytest.raises(NotFrozen, match="not clean"):
        freeze.require_freeze(repo)
    (repo / "scratch.py").unlink()
    (repo / "src" / "model.py").write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(NotFrozen, match="not clean"):
        freeze.require_freeze(repo)


def test_a_tag_that_is_not_an_ancestor_is_refused(repo):
    git(repo, "checkout", "-q", "-b", "other")
    commit(repo, "src/other.py", "y = 1\n")
    git(repo, "tag", freeze.FREEZE_TAG)
    git(repo, "checkout", "-q", "-")
    with pytest.raises(NotFrozen, match="not an ancestor"):
        freeze.require_freeze(repo)


def test_changing_what_the_freeze_fixed_closes_the_split_again(repo):
    git(repo, "tag", freeze.FREEZE_TAG)
    commit(repo, "configs/config.yaml", "alpha: 0.7\n")
    with pytest.raises(NotFrozen, match="changed since"):
        freeze.require_freeze(repo)


def test_a_fix_to_the_evaluation_code_after_the_freeze_is_allowed_and_listed(repo):
    commit(repo, "src/eval/metrics.py", "a = 1\n")
    git(repo, "tag", freeze.FREEZE_TAG)
    commit(repo, "src/eval/metrics.py", "a = 2\n")
    commit(repo, "scripts/run.py", "print(1)\n")
    out = freeze.require_freeze(repo)
    assert out["changed_since_freeze"] == ["scripts/run.py", "src/eval/metrics.py"]


def test_the_first_opening_is_recorded_and_a_second_must_give_a_reason(tmp_path):
    marker = tmp_path / "data" / "interim" / "test_split_opened.json"
    assert freeze.previous_openings(marker) == []
    freeze.record_opening(marker, "abc123")
    assert [o["commit"] for o in freeze.previous_openings(marker)] == ["abc123"]
    with pytest.raises(NotFrozen, match="already opened"):
        freeze.record_opening(marker, "def456")
    freeze.record_opening(marker, "def456", reopen_because="crashed after loading; nothing changed")
    openings = freeze.previous_openings(marker)
    assert [o["commit"] for o in openings] == ["abc123", "def456"]
    assert openings[1]["reopen_because"].startswith("crashed")
