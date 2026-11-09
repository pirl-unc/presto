"""Bounded, non-overwriting local stages with durable preflight and exit receipts."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import subprocess
import sys
import time

import psutil


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def available_memory():
    if sys.platform == "darwin":
        page = int(subprocess.check_output(["/usr/sbin/sysctl", "-n", "hw.pagesize"]))
        stats = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
        counts = {}
        for key in ("free", "speculative"):
            match = re.search(r"^Pages " + key + r":\s+(\d+)\.", stats, re.M)
            if match is None:
                raise RuntimeError("Missing macOS memory counter: " + key)
            counts[key] = int(match.group(1))
        return {"metric": "macos_free_plus_speculative", "bytes": sum(counts.values()) * page}
    if sys.platform.startswith("linux"):
        match = re.search(r"^MemAvailable:\s+(\d+) kB", Path("/proc/meminfo").read_text(), re.M)
        if match is None:
            raise RuntimeError("Missing Linux MemAvailable")
        return {"metric": "linux_MemAvailable", "bytes": int(match.group(1)) * 1024}
    raise RuntimeError("Unqualified memory-admission platform")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("test", "admit", "audit", "fit", "verify"))
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    assert args.attempt > 0
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "reproduce/launch.json").read_text())
    label = f"{args.stage}-attempt{args.attempt}"
    invocation = root / "reproduce" / (label + "-launch.json")
    receipt = root / "results" / (label + "-runtime.json")
    log_path = root / "launch_logs" / (label + ".log")
    if receipt.exists() or log_path.exists():
        raise FileExistsError("Preserve every existing attempt")
    result = {
        "stage": args.stage,
        "attempt": args.attempt,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preflight",
        "child_launched": False,
        "exit_code": None,
        "resource_violation": None,
    }
    with invocation.open("x") as stream:
        json.dump(
            {
                "argv": sys.argv,
                "config": config,
                "presto_runtime_head": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "dirty_tracked": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain", "--untracked-files=no"], text=True
                    ).strip()
                ),
            },
            stream,
            indent=2,
        )
    limits = config["limits"]
    try:
        result["memory_admission"] = available_memory()
        result["disk_free_before_bytes"] = shutil.disk_usage(root).free
        if result["memory_admission"]["bytes"] < limits["startup_memory_bytes"]:
            raise RuntimeError("startup_memory_floor")
        if (
            result["disk_free_before_bytes"]
            < limits["disk_floor_bytes"] + limits["start_margin_bytes"]
        ):
            raise RuntimeError("startup_disk_floor")
        if args.stage != "test":
            assert json.loads((root / "results/boundary-tests.json").read_text())["passed"]
        assert (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=config["hitlist_checkout"], text=True
            ).strip()
            == config["hitlist_commit"]
        )
        assert not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=config["hitlist_checkout"],
            text=True,
        ).strip()
        for relative, checksum in config["frozen_code_sha256"].items():
            assert digest(root / relative) == checksum, relative
        assert digest(root / "reproduce/plan.md") == config["plan_sha256"]
        for relative, checksum in config["hitlist_source_sha256"].items():
            assert digest(Path(config["hitlist_checkout"]) / relative) == checksum, relative
        for path, checksum in config["input_sha256"].items():
            assert digest(path) == checksum, path
        dependencies = {
            "admit": ("test", "boundary-tests.json"),
            "audit": ("admit", "admission.json"),
            "fit": ("audit", "admission-verification.json"),
            "verify": ("fit", "summary.json"),
        }
        if args.stage in dependencies:
            prior, output = dependencies[args.stage]
            assert (root / "results" / output).exists()
            assert any(
                json.loads(p.read_text())["status"] == "passed"
                for p in (root / "results").glob(prior + "-attempt*-runtime.json")
            )
        if args.stage in ("fit", "verify"):
            assert json.loads((root / "results/admission-verification.json").read_text())["passed"]
            for name, expected in json.loads((root / "results/admission.json").read_text())[
                "files"
            ].items():
                assert digest(root / "results" / name) == expected
        probe = psutil.Process()
        probe.memory_info()
        probe.children(recursive=True)
    except Exception as error:
        result.update(status="preflight_refused", refusal=str(error))
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return 2
    environment = {**os.environ, **config["environment_overrides"]}
    result["environment"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "pyarrow", "psutil")
        },
        "overrides": config["environment_overrides"],
    }
    command = [config["python"], "-B", str(root / "code" / (args.stage + ".py"))]
    started = time.monotonic()
    peak, violation = 0, None
    with log_path.open("x") as log:
        child = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        result.update(child_launched=True, child_pid=child.pid, argv=command)
        (root / "results" / (label + "-process.json")).write_text(json.dumps(result) + "\n")
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                rss = sum(p.memory_info().rss for p in [process, *process.children(recursive=True)])
                peak = max(peak, rss)
            except psutil.NoSuchProcess:
                rss = 0
            except Exception as error:
                violation, rss = "monitor_error:" + repr(error), 0
            if time.monotonic() - started > limits["stage_seconds"]:
                violation = "stage_time_limit"
            elif rss > limits["rss_bytes"]:
                violation = "process_tree_rss_limit"
            elif shutil.disk_usage(root).free < limits["disk_floor_bytes"]:
                violation = "disk_reserve"
            if violation:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            time.sleep(0.05)
        returncode = child.wait()
    peak_os = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    peak_os = int(peak_os if sys.platform == "darwin" else peak_os * 1024)
    if peak_os > limits["rss_bytes"]:
        violation = "terminal_child_peak_rss_limit"
    result.update(
        status="passed" if returncode == 0 and violation is None else "failed",
        elapsed_seconds=time.monotonic() - started,
        peak_observed_process_tree_rss_bytes=peak,
        peak_child_process_rss_bytes=peak_os,
        rss_poll_interval_seconds=0.05,
        exit_code=returncode,
        resource_violation=violation,
        disk_free_after_bytes=shutil.disk_usage(root).free,
    )
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
