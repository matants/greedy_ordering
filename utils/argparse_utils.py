import argparse
import json
import os
from typing import Any, Dict, List, Union

try:
    import yaml  # optional

    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


def str2bool(v: Union[str, bool]) -> bool:
    if isinstance(v, bool):
        return v
    s = v.lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {v!r}")


def parse_list(s: Union[str, List[Any]]) -> List[Any]:
    """Accept JSON-style lists (e.g. '[64]') or comma-separated (e.g. '32,32,3')."""
    if isinstance(s, list):
        return s
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            val = json.loads(s)
            if not isinstance(val, list):
                raise ValueError
            return val
        except Exception:
            raise argparse.ArgumentTypeError(f"Could not parse JSON list: {s}")
    # fallback: comma-separated, cast to int when possible
    out = []
    for item in filter(None, (x.strip() for x in s.split(","))):
        try:
            out.append(int(item))
        except ValueError:
            try:
                out.append(float(item))
            except ValueError:
                out.append(item)
    return out


def int_or_literal_factory(literal: str = "num_tasks"):
    def int_or_literal(s: str) -> Union[int, str]:
        """Allow an int or the literal."""
        if s == literal:
            return s
        try:
            return int(s)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Must be an integer or {literal}")

    return int_or_literal


def load_config_file(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    _, ext = os.path.splitext(path.lower())
    with open(path, "r", encoding="utf-8") as f:
        if ext in (".yml", ".yaml"):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML not installed but a YAML config was provided.")
            data = yaml.safe_load(f)
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a top-level JSON/YAML object (dict).")
    return data


def save_config_file(path: str, cfg: Dict[str, Any]) -> None:
    _, ext = os.path.splitext(path.lower())
    with open(path, "w", encoding="utf-8") as f:
        if ext in (".yml", ".yaml"):
            if not _HAS_YAML:
                raise RuntimeError("PyYAML not installed; cannot write YAML.")
            yaml.safe_dump(cfg, f, sort_keys=False)
        else:
            json.dump(cfg, f, indent=2)


def merge_params(base: Dict[str, Any], file_cfg: Dict[str, Any], cli_overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = {**base, **file_cfg, **cli_overrides}
    # normalize a couple fields
    if isinstance(cfg.get("input_size"), tuple):
        cfg["input_size"] = list(cfg["input_size"])
    # coerce num_training_iterations
    nti = cfg.get("num_training_iterations")
    if isinstance(nti, str) and nti != "num_tasks":
        # if someone put "40" in a file as a string, try to coerce
        try:
            cfg["num_training_iterations"] = int(nti)
        except Exception:
            raise ValueError("num_training_iterations must be an int or 'num_tasks'")
    return cfg
