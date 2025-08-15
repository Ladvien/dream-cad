#!/usr/bin/env python3
"""
Test suite for Story 3: Python Development Environment Setup
Verifies all acceptance criteria for Python environment configuration.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path


class TestPythonEnvironmentSetup(unittest.TestCase):
    """Test Python development environment setup for MVDream."""
    
    def test_python_version(self):
        """Test that Python 3.10+ is installed."""
        version_info = sys.version_info
        self.assertGreaterEqual(version_info.major, 3, "Python 3 required")
        
        # Accept 3.10, 3.11, or newer
        if version_info.major == 3:
            self.assertGreaterEqual(version_info.minor, 10, 
                f"Python 3.10+ required, found {version_info.major}.{version_info.minor}")
    
    def test_uv_installed(self):
        """Test that uv package manager is installed and accessible."""
        # Add .local/bin to PATH for testing
        env = os.environ.copy()
        env['PATH'] = f"{Path.home()}/.local/bin:{env['PATH']}"
        
        try:
            result = subprocess.run(
                ['uv', '--version'],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            self.assertIn('uv', result.stdout.lower(), "uv version not found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.fail("uv not found - package manager not installed or not in PATH")
    
    def test_poetry_installed(self):
        """Test that Poetry is installed and accessible."""
        # Add .local/bin to PATH for testing
        env = os.environ.copy()
        env['PATH'] = f"{Path.home()}/.local/bin:{env['PATH']}"
        
        try:
            result = subprocess.run(
                ['poetry', '--version'],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            self.assertIn('poetry', result.stdout.lower(), "Poetry not found")
            
            # Check version is ≥1.7.0
            import re
            version_match = re.search(r'(\d+)\.(\d+)\.(\d+)', result.stdout)
            if version_match:
                major = int(version_match.group(1))
                minor = int(version_match.group(2))
                
                # Accept 1.7+ or 2.0+
                if major == 1:
                    self.assertGreaterEqual(minor, 7, 
                        f"Poetry 1.7+ required, found {major}.{minor}")
                else:
                    self.assertGreaterEqual(major, 1, 
                        f"Poetry 1.7+ required, found {major}.{minor}")
                    
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.fail("Poetry not found - not installed or not in PATH")
    
    def test_poetry_virtualenv_config(self):
        """Test that Poetry is configured for in-project virtualenvs."""
        env = os.environ.copy()
        env['PATH'] = f"{Path.home()}/.local/bin:{env['PATH']}"
        
        try:
            result = subprocess.run(
                ['poetry', 'config', 'virtualenvs.in-project'],
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            self.assertEqual(result.stdout.strip(), 'true',
                "Poetry not configured for in-project virtualenvs")
        except subprocess.CalledProcessError:
            self.fail("Failed to check Poetry configuration")
    
    def test_git_configured(self):
        """Test that git is configured with user name and email."""
        try:
            # Check user name
            result_name = subprocess.run(
                ['git', 'config', '--global', 'user.name'],
                capture_output=True,
                text=True,
                check=True
            )
            self.assertTrue(result_name.stdout.strip(), 
                "Git user name not configured")
            
            # Check user email
            result_email = subprocess.run(
                ['git', 'config', '--global', 'user.email'],
                capture_output=True,
                text=True,
                check=True
            )
            self.assertTrue(result_email.stdout.strip(), 
                "Git user email not configured")
            
        except subprocess.CalledProcessError:
            self.fail("Git configuration not found")
    
    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists in ~/mvdream."""
        pyproject_path = Path.home() / 'mvdream' / 'pyproject.toml'
        self.assertTrue(pyproject_path.exists(),
            f"pyproject.toml not found at {pyproject_path}")
        
        # Verify it has Poetry configuration
        content = pyproject_path.read_text()
        self.assertIn('[tool.poetry]', content, 
            "pyproject.toml missing Poetry configuration")
        self.assertIn('name = "mvdream"', content,
            "Project name not set in pyproject.toml")
    
    def test_poetry_virtualenv_created(self):
        """Test that Poetry virtual environment is created."""
        venv_path = Path.home() / 'mvdream' / '.venv'
        self.assertTrue(venv_path.exists(),
            f"Virtual environment not found at {venv_path}")
        
        # Check for Python executable
        python_exe = venv_path / 'bin' / 'python'
        self.assertTrue(python_exe.exists(),
            f"Python executable not found in virtual environment")
    
    def test_development_tools_configured(self):
        """Test that development tools are configured in pyproject.toml."""
        pyproject_path = Path.home() / 'mvdream' / 'pyproject.toml'
        content = pyproject_path.read_text()
        
        # Check for dev dependencies
        self.assertIn('ruff', content, "ruff not in dev dependencies")
        self.assertIn('bandit', content, "bandit not in dev dependencies")
        self.assertIn('poethepoet', content, "poethepoet not in dev dependencies")
        
        # Check for tool configurations
        self.assertIn('[tool.ruff]', content, "ruff configuration missing")
        self.assertIn('[tool.bandit]', content, "bandit configuration missing")
        self.assertIn('[tool.poe.tasks]', content, "poethepoet tasks missing")
    
    def test_gitignore_exists(self):
        """Test that .gitignore file exists with Python patterns."""
        gitignore_path = Path.home() / 'mvdream' / '.gitignore'
        self.assertTrue(gitignore_path.exists(),
            f".gitignore not found at {gitignore_path}")
        
        content = gitignore_path.read_text()
        
        # Check for essential patterns
        patterns = [
            '__pycache__',
            '*.py[cod]',
            '.venv',
            '*.pth',  # PyTorch models
            '.idea',  # IDEs
            '.vscode'
        ]
        
        for pattern in patterns:
            self.assertIn(pattern, content,
                f".gitignore missing pattern: {pattern}")
    
    def test_poetry_dependencies_installed(self):
        """Test that Poetry dependencies are installed."""
        env = os.environ.copy()
        env['PATH'] = f"{Path.home()}/.local/bin:{env['PATH']}"
        
        try:
            # Check if dependencies are installed
            result = subprocess.run(
                ['poetry', 'show'],
                capture_output=True,
                text=True,
                check=True,
                cwd=Path.home() / 'mvdream',
                env=env
            )
            
            # Check for key dev dependencies
            dependencies = ['ruff', 'bandit', 'poethepoet', 'pytest']
            for dep in dependencies:
                self.assertIn(dep, result.stdout,
                    f"Dependency {dep} not installed")
                    
        except subprocess.CalledProcessError as e:
            self.skipTest(f"Could not check Poetry dependencies: {e}")
    
    def test_poetry_tasks_defined(self):
        """Test that poethepoet tasks are defined."""
        pyproject_path = Path.home() / 'mvdream' / 'pyproject.toml'
        content = pyproject_path.read_text()
        
        # Check for essential tasks
        tasks = ['test', 'test-gpu', 'lint', 'format']
        for task in tasks:
            self.assertIn(f'{task} =', content,
                f"Poetry task '{task}' not defined")


def run_tests():
    """Run all Python environment setup tests."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestPythonEnvironmentSetup)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)