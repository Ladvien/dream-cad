"""Tests for documentation completeness and validity."""

import os
import re
from pathlib import Path
from typing import List, Set
import pytest


class TestDocumentation:
    """Test documentation files exist and are valid."""
    
    @pytest.fixture
    def docs_dir(self):
        """Get documentation directory."""
        return Path("/mnt/datadrive_m2/dream-cad/docs")
    
    @pytest.fixture
    def root_dir(self):
        """Get project root directory."""
        return Path("/mnt/datadrive_m2/dream-cad")
    
    def test_readme_exists(self, root_dir):
        """Test that README.md exists and has content."""
        readme = root_dir / "README.md"
        assert readme.exists(), "README.md not found"
        
        content = readme.read_text()
        assert len(content) > 1000, "README.md seems too short"
        
        # Check for required sections
        assert "## Overview" in content or "## 🎯 Overview" in content
        assert "## Installation" in content or "## 📦 Installation" in content
        assert "## Usage" in content or "## 🎮 Usage" in content
        assert "Supported Models" in content
    
    def test_required_documentation_files(self, docs_dir):
        """Test that all required documentation files exist."""
        required_docs = [
            "model_comparison.md",
            "hardware_requirements.md",
            "troubleshooting_models.md",
            "configuration_examples.md",
            "licensing.md",
            "api_reference.md",
            "performance_tuning.md",
        ]
        
        for doc in required_docs:
            doc_path = docs_dir / doc
            assert doc_path.exists(), f"Required documentation {doc} not found"
            
            # Check file has substantial content
            content = doc_path.read_text()
            assert len(content) > 500, f"{doc} seems too short"
    
    def test_model_comparison_content(self, docs_dir):
        """Test model comparison guide has all models."""
        doc = docs_dir / "model_comparison.md"
        content = doc.read_text()
        
        models = ["MVDream", "TripoSR", "Stable-Fast-3D", "TRELLIS", "Hunyuan3D"]
        for model in models:
            assert model in content, f"{model} not documented in model comparison"
        
        # Check for comparison table
        assert "| Feature |" in content or "| Model |" in content
        assert "Generation Speed" in content
        assert "Quality Score" in content
        assert "Min VRAM" in content or "VRAM" in content
    
    def test_hardware_requirements_content(self, docs_dir):
        """Test hardware requirements documentation."""
        doc = docs_dir / "hardware_requirements.md"
        content = doc.read_text()
        
        # Check for GPU requirements
        assert "RTX 3090" in content
        assert "VRAM" in content
        assert "CUDA" in content
        
        # Check for per-model requirements
        models = ["TripoSR", "Stable-Fast-3D", "TRELLIS", "Hunyuan3D", "MVDream"]
        for model in models:
            assert model in content, f"{model} hardware requirements not documented"
    
    def test_troubleshooting_content(self, docs_dir):
        """Test troubleshooting guide completeness."""
        doc = docs_dir / "troubleshooting_models.md"
        content = doc.read_text()
        
        # Check for common issues
        issues = [
            "CUDA Out of Memory",
            "Model Download",
            "Slow Generation",
            "Quality",
        ]
        
        for issue in issues:
            assert issue in content or issue.lower() in content.lower(), \
                f"Troubleshooting for '{issue}' not found"
        
        # Check for model-specific sections
        models = ["TripoSR", "Stable-Fast-3D", "TRELLIS", "Hunyuan3D", "MVDream"]
        for model in models:
            assert f"{model} Issues" in content or model in content, \
                f"{model} troubleshooting not documented"
    
    def test_configuration_examples(self, docs_dir):
        """Test configuration examples documentation."""
        doc = docs_dir / "configuration_examples.md"
        content = doc.read_text()
        
        # Check for code examples
        assert "```python" in content, "No Python code examples found"
        assert "```yaml" in content or "```json" in content, "No config file examples"
        
        # Check for different use cases
        use_cases = [
            "Game Asset",
            "3D Printing",
            "Architectural",
            "Research",
            "Production",
        ]
        
        for use_case in use_cases:
            assert use_case in content or use_case.lower() in content.lower(), \
                f"Configuration for '{use_case}' not found"
    
    def test_licensing_documentation(self, docs_dir):
        """Test licensing documentation completeness."""
        doc = docs_dir / "licensing.md"
        content = doc.read_text()
        
        # Check for all model licenses
        models_licenses = {
            "MVDream": "Apache",
            "TripoSR": "MIT",
            "Stable-Fast-3D": "Community",
            "TRELLIS": "Apache",
            "Hunyuan3D": "Commercial",
        }
        
        for model, license_type in models_licenses.items():
            assert model in content, f"{model} license not documented"
            # License type might be mentioned differently
            assert license_type in content or license_type.lower() in content.lower(), \
                f"{model} license type not mentioned"
        
        # Check for commercial use guidance
        assert "Commercial Use" in content
        assert "Attribution" in content
    
    def test_api_reference(self, docs_dir):
        """Test API reference documentation."""
        doc = docs_dir / "api_reference.md"
        content = doc.read_text()
        
        # Check for main classes
        classes = [
            "ModelFactory",
            "Model3D",
            "GenerationResult",
            "ModelCapabilities",
            "JobQueue",
            "BatchProcessor",
            "ModelBenchmark",
            "QualityAssessor",
        ]
        
        for cls in classes:
            assert cls in content, f"Class {cls} not documented in API reference"
        
        # Check for code examples
        assert "```python" in content
        assert "from dream_cad" in content
        
        # Check for method documentation
        assert "generate_from_text" in content
        assert "generate_from_image" in content
    
    def test_performance_tuning(self, docs_dir):
        """Test performance tuning documentation."""
        doc = docs_dir / "performance_tuning.md"
        content = doc.read_text()
        
        # Check for optimization techniques
        techniques = [
            "FP16",
            "GPU Cache",
            "Batch Size",
            "Memory",
            "CUDA",
            "PyTorch",
        ]
        
        for technique in techniques:
            assert technique in content or technique.lower() in content.lower(), \
                f"Optimization technique '{technique}' not documented"
        
        # Check for model-specific optimizations
        models = ["MVDream", "TripoSR", "Stable-Fast-3D", "TRELLIS", "Hunyuan3D"]
        for model in models:
            assert f"{model} Optimization" in content or model in content, \
                f"{model} optimization not documented"
    
    def test_links_in_readme(self, root_dir, docs_dir):
        """Test that links in README point to existing files."""
        readme = root_dir / "README.md"
        content = readme.read_text()
        
        # Find all markdown links
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = re.findall(link_pattern, content)
        
        for link_text, link_url in links:
            # Skip external URLs
            if link_url.startswith("http"):
                continue
            
            # Skip anchors
            if link_url.startswith("#"):
                continue
            
            # Check if file exists
            if link_url.startswith("docs/"):
                file_path = root_dir / link_url
                assert file_path.exists(), f"Broken link in README: {link_url}"
    
    def test_code_blocks_syntax(self, docs_dir):
        """Test that code blocks have proper syntax highlighting."""
        docs = list(docs_dir.glob("*.md"))
        
        for doc in docs:
            content = doc.read_text()
            
            # Find code blocks
            code_blocks = re.findall(r'```(\w*)\n(.*?)```', content, re.DOTALL)
            
            for lang, code in code_blocks:
                if not lang:
                    # Some code blocks might not need language specification
                    continue
                
                # Check common languages are spelled correctly
                valid_langs = [
                    "python", "bash", "yaml", "json", "dockerfile",
                    "text", "shell", "sh", "py", "toml", "ini", "xml"
                ]
                
                assert lang.lower() in valid_langs or lang == "", \
                    f"Unknown language '{lang}' in {doc.name}"
    
    def test_tables_formatting(self, docs_dir):
        """Test that markdown tables are properly formatted."""
        docs = list(docs_dir.glob("*.md"))
        
        for doc in docs:
            content = doc.read_text()
            lines = content.split('\n')
            
            in_table = False
            for i, line in enumerate(lines):
                if '|' in line:
                    # Check if it's a table row
                    if line.strip().startswith('|') and line.strip().endswith('|'):
                        in_table = True
                        
                        # Count columns
                        cols = len([c for c in line.split('|') if c.strip()])
                        
                        # Check separator line
                        if i + 1 < len(lines) and '---' in lines[i + 1]:
                            sep_cols = len([c for c in lines[i + 1].split('|') if c.strip()])
                            assert cols == sep_cols, \
                                f"Table column mismatch in {doc.name} at line {i}"
                elif in_table and line.strip() == "":
                    in_table = False
    
    def test_no_broken_internal_references(self, docs_dir):
        """Test that internal references are valid."""
        docs = list(docs_dir.glob("*.md"))
        
        all_headers: Set[str] = set()
        
        # Collect all headers
        for doc in docs:
            content = doc.read_text()
            headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
            all_headers.update(headers)
        
        # Check references
        for doc in docs:
            content = doc.read_text()
            
            # Find anchor links
            anchor_links = re.findall(r'\[([^\]]+)\]\(#([^)]+)\)', content)
            
            for link_text, anchor in anchor_links:
                # Convert anchor to expected header format
                header = anchor.replace('-', ' ').title()
                
                # This is a simplified check - anchors are complex to validate perfectly
                # Just ensure the document has some related content
                assert any(h for h in all_headers if anchor.replace('-', '') in h.lower().replace(' ', '')), \
                    f"Possible broken anchor link: #{anchor} in {doc.name}"
    
    def test_documentation_coverage(self, root_dir):
        """Test that key project components are documented."""
        # Check if main Python modules have corresponding documentation
        modules = [
            "dream_cad/models",
            "dream_cad/queue",
            "dream_cad/benchmark",
            "dream_cad/ui",
        ]
        
        readme = (root_dir / "README.md").read_text()
        api_ref = (root_dir / "docs" / "api_reference.md").read_text()
        
        for module in modules:
            module_name = module.split('/')[-1]
            assert module_name in readme or module_name in api_ref, \
                f"Module {module} not documented"
    
    def test_consistent_model_names(self, docs_dir):
        """Test that model names are used consistently."""
        docs = list(docs_dir.glob("*.md"))
        
        # Standard model names
        standard_names = {
            "mvdream": "MVDream",
            "triposr": "TripoSR",
            "stable-fast-3d": "Stable-Fast-3D",
            "trellis": "TRELLIS",
            "hunyuan3d-mini": "Hunyuan3D-Mini",
        }
        
        for doc in docs:
            content = doc.read_text()
            
            # Check that we use consistent capitalization
            for key, proper_name in standard_names.items():
                # Allow both the key and proper name
                if key in content.lower():
                    # Make sure the proper name appears somewhere
                    assert proper_name in content or key in content, \
                        f"Inconsistent model naming for {proper_name} in {doc.name}"


def test_documentation_exists():
    """Quick test that documentation directory exists."""
    docs_dir = Path("/mnt/datadrive_m2/dream-cad/docs")
    assert docs_dir.exists(), "Documentation directory not found"
    assert docs_dir.is_dir(), "docs is not a directory"
    
    # Check that we have at least some documentation
    md_files = list(docs_dir.glob("*.md"))
    assert len(md_files) > 5, "Too few documentation files"


if __name__ == "__main__":
    # Run basic test
    test_documentation_exists()
    print("✓ Basic documentation test passed")
    
    # Run full test suite
    tester = TestDocumentation()
    docs_dir = Path("/mnt/datadrive_m2/dream-cad/docs")
    root_dir = Path("/mnt/datadrive_m2/dream-cad")
    
    try:
        tester.test_readme_exists(root_dir)
        print("✓ README test passed")
        
        tester.test_required_documentation_files(docs_dir)
        print("✓ Required files test passed")
        
        tester.test_model_comparison_content(docs_dir)
        print("✓ Model comparison test passed")
        
        tester.test_hardware_requirements_content(docs_dir)
        print("✓ Hardware requirements test passed")
        
        tester.test_troubleshooting_content(docs_dir)
        print("✓ Troubleshooting test passed")
        
        tester.test_configuration_examples(docs_dir)
        print("✓ Configuration examples test passed")
        
        tester.test_licensing_documentation(docs_dir)
        print("✓ Licensing test passed")
        
        tester.test_api_reference(docs_dir)
        print("✓ API reference test passed")
        
        tester.test_performance_tuning(docs_dir)
        print("✓ Performance tuning test passed")
        
        tester.test_links_in_readme(root_dir, docs_dir)
        print("✓ README links test passed")
        
        tester.test_documentation_coverage(root_dir)
        print("✓ Documentation coverage test passed")
        
        print("\n✅ All documentation tests passed!")
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise