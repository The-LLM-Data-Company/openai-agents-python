from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest

from agents.extensions.sandbox.blaxel import sandbox as blaxel
from agents.extensions.sandbox.e2b import sandbox as e2b
from agents.sandbox import Manifest
from agents.sandbox.snapshot import NoopSnapshot
from agents.sandbox.types import ExecResult


@pytest.fixture
def poisoned_workspace(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pythonpath = tmp_path / "pythonpath"
    pythonpath.mkdir()
    marker = tmp_path / "untrusted-import"
    poison = (
        f"open({str(marker)!r}, 'w').write('executed')\n"
        "raise RuntimeError('untrusted workspace import')\n"
    )
    for module in ("json", "signal", "subprocess"):
        (workspace / f"{module}.py").write_text(poison)
    (pythonpath / "sitecustomize.py").write_text(poison)
    return workspace, {**os.environ, "PYTHONPATH": str(pythonpath)}, marker


@pytest.mark.skipif(sys.platform != "linux", reason="provider process ownership uses /proc")
@pytest.mark.parametrize("provider", ["blaxel", "e2b"])
@pytest.mark.parametrize("helper", ["supervisor", "terminator"])
def test_trusted_process_helpers_ignore_workspace_python(
    poisoned_workspace: tuple[Path, dict[str, str], Path],
    provider: Literal["blaxel", "e2b"],
    helper: Literal["supervisor", "terminator"],
) -> None:
    workspace, environment, marker = poisoned_workspace
    if helper == "supervisor":
        builder = (
            blaxel._blaxel_supervised_command
            if provider == "blaxel"
            else e2b._e2b_supervised_command
        )
        command = builder(["/bin/echo", "trusted child"])
    else:
        terminator = (
            blaxel._blaxel_process_group_termination_command
            if provider == "blaxel"
            else e2b._e2b_process_group_termination_command
        )
        command = terminator(uuid.uuid4().hex, start_polls=1)
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=workspace,
        env=environment,
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=10,
    )

    assert not marker.exists(), result.stderr
    assert result.returncode == 0, result.stderr
    if helper == "supervisor":
        assert result.stdout == "trusted child\n"


@pytest.mark.skipif(sys.platform != "linux", reason="provider process ownership uses /proc")
@pytest.mark.parametrize("provider", ["blaxel", "e2b"])
def test_supervisor_preserves_child_python_imports(
    tmp_path: Path, provider: Literal["blaxel", "e2b"]
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pythonpath = tmp_path / "pythonpath"
    pythonpath.mkdir()
    (workspace / "agent_local.py").write_text("VALUE = 'workspace'\n")
    (pythonpath / "agent_dependency.py").write_text("VALUE = 'pythonpath'\n")
    builder = (
        blaxel._blaxel_supervised_command if provider == "blaxel" else e2b._e2b_supervised_command
    )
    command = builder(
        [
            sys.executable,
            "-c",
            "import agent_local, agent_dependency; "
            "print(agent_local.VALUE, agent_dependency.VALUE)",
        ]
    )
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=workspace,
        env={**os.environ, "PYTHONPATH": str(pythonpath)},
        capture_output=True,
        text=True,
        start_new_session=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "workspace pythonpath\n"


@pytest.mark.skipif(sys.platform == "win32", reason="provider stat helpers use POSIX metadata")
@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["blaxel", "e2b"])
async def test_trusted_stat_helper_ignores_workspace_python(
    poisoned_workspace: tuple[Path, dict[str, str], Path],
    provider: Literal["blaxel", "e2b"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, environment, marker = poisoned_workspace
    target = workspace / "output.txt"
    target.write_bytes(b"deliverable")
    session: blaxel.BlaxelSandboxSession | e2b.E2BSandboxSession
    if provider == "blaxel":
        session = blaxel.BlaxelSandboxSession(
            state=blaxel.BlaxelSandboxSessionState(
                session_id=uuid.uuid4(),
                manifest=Manifest(root=str(workspace)),
                snapshot=NoopSnapshot(id="test"),
                sandbox_name="test",
            ),
            sandbox=object(),
        )
    else:
        session = e2b.E2BSandboxSession(
            state=e2b.E2BSandboxSessionState(
                session_id=uuid.uuid4(),
                manifest=Manifest(root=str(workspace)),
                snapshot=NoopSnapshot(id="test"),
                sandbox_id="test",
            ),
            sandbox=object(),
        )

    def run_locally(*command: str | Path, **kwargs: object) -> ExecResult:
        result = subprocess.run(
            [sys.executable, *command[1:]],
            cwd=workspace,
            env=environment,
            capture_output=True,
            timeout=10,
        )
        return ExecResult(stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode)

    monkeypatch.setattr(session, "exec", AsyncMock(side_effect=run_locally))
    entry = await session.stat("output.txt")

    assert not marker.exists()
    assert entry is not None
    assert entry.size == len(b"deliverable")
