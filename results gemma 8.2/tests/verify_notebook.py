"""Execute the real single-file notebook idle path, without network/GPU work."""
import ast
import importlib.util
import shutil
import tempfile
from pathlib import Path


def main():
    original = Path(__file__).resolve().parents[1] / "dist" / "qspec_hybrid_molab.py"
    with tempfile.TemporaryDirectory(prefix="qspec-notebook-check-") as directory:
        file = Path(directory) / "notebook.py"
        shutil.copyfile(original, file)
        spec = importlib.util.spec_from_file_location("qspec_notebook_idle_check", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        outputs, definitions = module.app.run()
        root = definitions["ROOT"]
        assert (root / "qspec_hybrid/core.py").is_file()
        assert (root / "notebook.py").is_file(), "Recovery must include the notebook"
        ast.parse((root / "notebook.py").read_text())
        assert not (root / "job.json").exists(), "Opening the app must never launch a worker"
        assert not (root / ".venv-hybrid").exists(), "Opening the app must never install dependencies"
        assert not (root / "models").exists(), "Opening the app must never download model weights"
        assert definitions["pilot_button"].value is False
        assert definitions["full_button"].value is False
        assert len(outputs) >= 8
        print("PASS: real standalone notebook idle execution, embedded extraction, recovery notebook, inactive run buttons; no worker/setup/model download.")


if __name__ == "__main__":
    main()
