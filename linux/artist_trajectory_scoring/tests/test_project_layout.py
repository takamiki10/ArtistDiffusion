"""Regression checks for script relocation and checkpoint import compatibility."""
import importlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dependencies import PROJECT_ROOT, script_path


class ProjectLayoutTests(unittest.TestCase):
    def test_scripts_resolve_to_their_functional_folders(self):
        expected = {
            "generate_joint_trajectory_diffusion_v8_1.py": PROJECT_ROOT,
            "validate_diffusion_v8_1_deployment_output.py": PROJECT_ROOT / "evaluation",
            "generate_ik_seed_path.py": PROJECT_ROOT / "dependencies",
            "benchmark_ik_mlp_pipeline_smoothness.py": PROJECT_ROOT / "benchmark",
        }
        for name, parent in expected.items():
            with self.subTest(name=name):
                self.assertEqual(script_path(name).parent, parent)
        with self.assertRaises(FileNotFoundError):
            script_path("not_a_pipeline_script.py")

    def test_checkpoint_model_class_name_is_preserved(self):
        trainer = importlib.import_module(
            "train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet"
        )
        model_class, class_path = trainer.locate_v5_model_class()
        self.assertEqual(
            class_path,
            "train_conditional_diffusion_trajectory_v5_residual_unet.LocalResidualConditionalUNet1D",
        )
        self.assertEqual(model_class.__module__, class_path.rsplit(".", 1)[0])

    def test_relocated_validator_runs_outside_the_repository_without_pythonpath(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, str(script_path("validate_diffusion_v8_1_deployment_output.py")), "--help"],
                cwd=cwd, env=env, capture_output=True, text=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output_dir", result.stdout)

    def test_worker_process_can_unpickle_relocated_types(self):
        # A fresh process must resolve the original module name used by workers.
        program = (
            "import dependencies, pickle; "
            "from train_conditional_diffusion_trajectory_v5_residual_unet "
            "import LocalResidualConditionalUNet1D as model; "
            "assert pickle.loads(pickle.dumps(model)) is model"
        )
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
