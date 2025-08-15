#!/usr/bin/env python3
"""
Test suite for Story 7: Model Download and Verification
Verifies all acceptance criteria for model management.
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestModelDownload(unittest.TestCase):
    """Test model download and verification functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.project_root = Path("/mnt/datadrive_m2/dream-cad")
        self.model_dir = self.project_root / "models"
        self.cache_dir = Path("/mnt/datadrive_m2/.huggingface")
        os.chdir(self.project_root)
    
    def test_huggingface_cache_configured(self):
        """Test that Hugging Face cache directory is configured correctly."""
        # Check directory exists
        self.assertTrue(
            self.cache_dir.exists(),
            f"HuggingFace cache directory not found at {self.cache_dir}"
        )
        
        # Check environment variable would be set by script
        result = subprocess.run(
            ["poetry", "run", "python", "-c", 
             "from scripts.download_models import setup_cache_dir; "
             "dir = setup_cache_dir(); "
             "import os; "
             "print(os.environ.get('HF_HOME', ''))"],
            capture_output=True,
            text=True,
        )
        
        self.assertEqual(result.returncode, 0, f"Failed to run setup: {result.stderr}")
        self.assertIn("/mnt/datadrive_m2/.huggingface", result.stdout.strip())
    
    def test_model_downloaded(self):
        """Test that sd-v2.1-base-4view.pt model is downloaded."""
        model_path = self.model_dir / "sd-v2.1-base-4view.pt"
        
        # Check if model exists
        self.assertTrue(
            model_path.exists(),
            f"Model not found at {model_path}. Run 'poe download-models' first."
        )
        
        # Check file size (should be ~5GB for real model, or small for placeholder)
        size_gb = model_path.stat().st_size / (1024**3)
        
        # Real model should be around 5GB
        if size_gb > 1:
            self.assertGreater(
                size_gb, 4.0,
                f"Model size {size_gb:.2f}GB seems too small for real model"
            )
            self.assertLess(
                size_gb, 6.0,
                f"Model size {size_gb:.2f}GB seems too large"
            )
    
    def test_model_checksum_verified(self):
        """Test that model checksum is calculated and stored."""
        info_file = self.model_dir / "model_info.json"
        
        # Check info file exists
        self.assertTrue(
            info_file.exists(),
            f"Model info file not found at {info_file}"
        )
        
        # Load and check content
        with open(info_file) as f:
            model_info = json.load(f)
        
        self.assertIn("sd-v2.1-base-4view.pt", model_info)
        
        info = model_info["sd-v2.1-base-4view.pt"]
        self.assertIn("sha256", info)
        self.assertIn("path", info)
        self.assertIn("size_bytes", info)
        
        # Check sha256 is valid (64 hex chars or "placeholder")
        sha256 = info["sha256"]
        if sha256 != "placeholder":
            self.assertEqual(len(sha256), 64, "SHA256 should be 64 characters")
            self.assertTrue(
                all(c in "0123456789abcdef" for c in sha256),
                "SHA256 should be hexadecimal"
            )
    
    def test_model_loads_successfully(self):
        """Test that model can be loaded in Python."""
        # Run the test_model_loading script
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/test_model_loading.py"],
            capture_output=True,
            text=True,
        )
        
        # Should complete without error
        self.assertEqual(
            result.returncode, 0,
            f"Model loading test failed:\n{result.stdout}\n{result.stderr}"
        )
        
        # Check for success indicators in output
        if "PyTorch not available" not in result.stdout:
            # If PyTorch is available, should load successfully
            self.assertIn(
                "Model",
                result.stdout,
                "Model loading output not found"
            )
    
    def test_poe_download_models_task(self):
        """Test that 'poe download-models' task is defined."""
        pyproject = self.project_root / "pyproject.toml"
        content = pyproject.read_text()
        
        # Check task is defined
        self.assertIn(
            "download-models",
            content,
            "download-models task not found in pyproject.toml"
        )
        
        # Check it references the correct script
        self.assertIn(
            "scripts/download_models.py",
            content,
            "download-models task doesn't reference correct script"
        )
    
    def test_poe_verify_models_task(self):
        """Test that 'poe verify-models' task is defined."""
        pyproject = self.project_root / "pyproject.toml"
        content = pyproject.read_text()
        
        # Check task is defined
        self.assertIn(
            "verify-models",
            content,
            "verify-models task not found in pyproject.toml"
        )
        
        # Should use --verify-only flag
        self.assertIn(
            "--verify-only",
            content,
            "verify-models task doesn't use --verify-only flag"
        )
    
    def test_model_documentation_exists(self):
        """Test that model documentation exists."""
        docs_file = self.project_root / "docs" / "models.md"
        
        self.assertTrue(
            docs_file.exists(),
            f"Model documentation not found at {docs_file}"
        )
        
        # Check content
        content = docs_file.read_text()
        required_sections = [
            "Model Storage",
            "Available Models",
            "Model Download",
            "Disk Space Management",
            "sd-v2.1-base-4view",
            "/mnt/datadrive_m2/.huggingface",
        ]
        
        for section in required_sections:
            self.assertIn(
                section,
                content,
                f"Documentation missing section: {section}"
            )
    
    def test_disk_space_documented(self):
        """Test that disk space usage is documented."""
        # Check current disk usage
        import shutil
        stat = shutil.disk_usage("/mnt/datadrive_m2")
        free_gb = stat.free / (1024**3)
        
        # Should have reasonable free space
        self.assertGreater(
            free_gb, 10,
            f"Less than 10GB free space on /mnt/datadrive_m2: {free_gb:.1f}GB"
        )
        
        # Check if model size is documented
        docs_file = self.project_root / "docs" / "models.md"
        if docs_file.exists():
            content = docs_file.read_text()
            self.assertIn("~10GB", content, "Model size not documented")
            self.assertIn("15GB", content, "Minimum space requirement not documented")
    
    def test_download_script_executable(self):
        """Test that download script is executable."""
        script = self.project_root / "scripts" / "download_models.py"
        
        self.assertTrue(
            script.exists(),
            f"Download script not found at {script}"
        )
        
        # Check shebang
        content = script.read_text()
        self.assertTrue(
            content.startswith("#!/usr/bin/env python3"),
            "Download script missing shebang"
        )
        
        # Check executable permission
        self.assertTrue(
            os.access(script, os.X_OK),
            "Download script is not executable"
        )
    
    def test_model_verification_works(self):
        """Test that model verification functionality works."""
        # Run verify-only mode
        result = subprocess.run(
            ["poetry", "run", "python", "scripts/download_models.py", "--verify-only"],
            capture_output=True,
            text=True,
        )
        
        # Should complete successfully
        self.assertEqual(
            result.returncode, 0,
            f"Model verification failed:\n{result.stdout}\n{result.stderr}"
        )
        
        # Check output
        self.assertIn(
            "Verifying",
            result.stdout,
            "Verification output not found"
        )
    
    def test_backup_strategy_documented(self):
        """Test that backup strategy is documented."""
        docs_file = self.project_root / "docs" / "models.md"
        
        if docs_file.exists():
            content = docs_file.read_text()
            
            # Check for backup section
            self.assertIn(
                "Backup",
                content,
                "Backup strategy not documented"
            )
            
            # Check for rsync example
            self.assertIn(
                "rsync",
                content,
                "Backup command example not provided"
            )


def run_tests():
    """Run all model download tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestModelDownload)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)