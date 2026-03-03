"""Model weights CLI commands.

Supports listing known model artifacts and downloading checkpoints
from either a registry entry or a direct URL.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "presto" / "weights"
BUILTIN_WEIGHT_REGISTRY: Dict[str, Dict[str, Dict[str, Any]]] = {"models": {}}


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def _normalize_registry(payload: Any) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Normalize registry payload into {'models': {name: metadata}}."""
    if not isinstance(payload, dict):
        raise ValueError("Registry must be a JSON object")
    models = payload.get("models")
    if models is None:
        models = payload
    if not isinstance(models, dict):
        raise ValueError("Registry models must be a JSON object")
    return {"models": models}


def load_weight_registry(source: Optional[str] = None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load a model weight registry from built-in defaults, file, or URL."""
    if source is None:
        return {"models": dict(BUILTIN_WEIGHT_REGISTRY["models"])}

    if _is_url(source):
        request = Request(source, headers={"User-Agent": "presto-weights-cli"})
        with urlopen(request) as resp:
            text = resp.read().decode("utf-8")
        return _normalize_registry(json.loads(text))

    path = Path(source).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Registry not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return _normalize_registry(json.load(f))


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def download_file(
    url: str,
    output_path: Path,
    expected_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Download a file and optionally verify SHA256."""
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hasher = hashlib.sha256()
    n_bytes = 0

    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=output_path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            request = Request(url, headers={"User-Agent": "presto-weights-cli"})
            with urlopen(request) as resp:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    tmp.write(chunk)
                    hasher.update(chunk)
                    n_bytes += len(chunk)
            tmp.flush()
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    digest = hasher.hexdigest()
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"SHA256 mismatch for {url}: expected {expected_sha256}, got {digest}"
        )

    tmp_path.replace(output_path)
    return {"bytes": n_bytes, "sha256": digest, "path": str(output_path)}


def cmd_weights_list(args: Any) -> int:
    """Handle `presto weights list`."""
    try:
        registry = load_weight_registry(args.registry)
    except Exception as exc:
        print(f"Error loading weight registry: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(registry, indent=2))
        return 0

    models = registry.get("models", {})
    if not models:
        print("No registered model weights.")
        if args.registry is None:
            print("Provide --registry with a JSON file or URL to list published artifacts.")
        return 0

    print("Available model weights:")
    for name, spec in sorted(models.items()):
        description = str(spec.get("description", "")).strip()
        url = str(spec.get("url", "")).strip()
        filename = str(spec.get("filename", "")).strip()
        print(f"  {name}")
        if description:
            print(f"    {description}")
        if filename:
            print(f"    file: {filename}")
        if url:
            print(f"    url: {url}")
    return 0


def _default_filename_for_url(url: str, model_name: Optional[str]) -> str:
    path_name = Path(urlparse(url).path).name
    if path_name:
        return path_name
    if model_name:
        return f"{model_name}.pt"
    return "presto_model.pt"


def cmd_weights_download(args: Any) -> int:
    """Handle `presto weights download`."""
    if not args.name and not args.url:
        print("Provide --name or --url.", file=sys.stderr)
        return 1

    spec: Dict[str, Any] = {}
    model_name: Optional[str] = args.name
    url: Optional[str] = args.url

    if url is None:
        try:
            registry = load_weight_registry(args.registry)
        except Exception as exc:
            print(f"Error loading weight registry: {exc}", file=sys.stderr)
            return 1
        models = registry.get("models", {})
        if model_name not in models:
            print(
                f"Model '{model_name}' not found in registry. "
                "Use 'presto weights list --registry ...' to inspect entries.",
                file=sys.stderr,
            )
            return 1
        spec = models[model_name]
        url = spec.get("url")
        if not url:
            print(f"Registry entry '{model_name}' is missing a URL.", file=sys.stderr)
            return 1

    expected_sha256 = spec.get("sha256")
    output: Path
    if args.output:
        output = Path(args.output)
    else:
        cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else DEFAULT_CACHE_DIR
        filename = spec.get("filename") or _default_filename_for_url(url, model_name)
        output = cache_dir / filename

    output = output.expanduser()

    if output.exists() and not args.force:
        if expected_sha256:
            existing_hash = _sha256_file(output)
            if existing_hash.lower() == str(expected_sha256).lower():
                print(f"Using existing weights: {output}")
                return 0
            print(
                "Existing file hash does not match expected SHA256. "
                "Use --force to re-download.",
                file=sys.stderr,
            )
            return 1
        print(f"Using existing weights: {output}")
        return 0

    try:
        result = download_file(url=url, output_path=output, expected_sha256=expected_sha256)
    except Exception as exc:
        print(f"Error downloading weights: {exc}", file=sys.stderr)
        return 1

    result_path = result.get("path", str(output))
    result_sha256 = result.get("sha256", _sha256_file(output))
    print(f"Downloaded weights to {result_path}")
    print(f"SHA256: {result_sha256}")
    return 0
