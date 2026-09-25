"""E0. Record the exact hardware and software environment the experiments ran on.

Output: results/final_eval/e0_environment.json
Usage:  python experiments/e0_environment.py
"""
import importlib.metadata as md
import platform
import subprocess
import sys

from _common import banner, git_commit, ollama_get, ollama_post, run_meta, save_json


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:
        return f"unavailable ({e})"


def pkg(name):
    try:
        return md.version(name)
    except Exception:
        return None


def main():
    banner("E0  Environment")
    env = run_meta()
    env["macos"] = platform.mac_ver()[0] or None
    env["cpu"] = sh(["sysctl", "-n", "machdep.cpu.brand_string"])
    env["physical_cores"] = sh(["sysctl", "-n", "hw.physicalcpu"])
    env["logical_cores"] = sh(["sysctl", "-n", "hw.logicalcpu"])
    mem = sh(["sysctl", "-n", "hw.memsize"])
    env["ram_gb"] = round(int(mem) / 1024 ** 3, 1) if mem.isdigit() else mem
    env["machine_model"] = sh(["sysctl", "-n", "hw.model"])
    env["python_executable"] = sys.executable
    env["packages"] = {
        p: pkg(p) for p in [
            "torch", "sentence-transformers", "transformers", "openai-whisper",
            "numpy", "scipy", "requests", "flask", "pytest", "sounddevice", "tokenizers",
        ]
    }
    env["git_status_short"] = sh(["git", "status", "--short"])
    env["git_commit"] = git_commit()

    try:
        env["ollama_version"] = ollama_get("/api/version").get("version")
        models = ollama_get("/api/tags").get("models", [])
        env["ollama_models"] = []
        for m in models:
            entry = {"name": m.get("name"), "size_bytes": m.get("size"), "digest": m.get("digest")}
            try:
                d = ollama_post("/api/show", {"model": m["name"]}).get("details", {})
                entry.update({
                    "family": d.get("family"),
                    "parameter_size": d.get("parameter_size"),
                    "quantization_level": d.get("quantization_level"),
                    "format": d.get("format"),
                })
            except Exception as e:
                entry["show_error"] = str(e)
            env["ollama_models"].append(entry)
    except Exception as e:
        env["ollama_error"] = f"Ollama not reachable: {e}"

    for k, v in env.items():
        if k != "packages":
            print(f"  {k}: {v}")
    print(f"  packages: {env['packages']}")
    save_json("e0_environment.json", env)


if __name__ == "__main__":
    main()
