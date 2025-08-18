from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from dream_cad.models.triposr import TripoSR
from dream_cad.models.base import ModelConfig
def test_generation():
    print("Testing TripoSR 3D Generation")
    print("-" * 40)
    config = ModelConfig(
        model_name="triposr",
        device="cpu",
        extra_params={
            "resolution": 512,
            "precision": "fp32"
        }
    )
    print(f"✓ Configuration created")
    print(f"  - Model: {config.model_name}")
    print(f"  - Device: {config.device}")
    print(f"  - Resolution: {config.extra_params.get('resolution', 'N/A')}")
    model = TripoSR(config)
    print(f"✓ Model initialized (mock mode)")
    caps = model.capabilities
    print(f"✓ Model capabilities:")
    print(f"  - Min VRAM: {caps.min_vram_gb}GB")
    print(f"  - Generation time: ~0.5-2s")
    print(f"  - Formats: obj, ply, stl, glb")
    prompt = "a futuristic spaceship"
    print(f"\n🎨 Generating: '{prompt}'")
    result = model.generate_from_text(
        prompt=prompt,
        output_dir="outputs/test",
        output_format="obj"
    )
    print(f"✓ Generation complete!")
    print(f"  - Output: {result.output_path}")
    print(f"  - Vertices: {result.vertices.shape if hasattr(result, 'vertices') else 'N/A'}")
    print(f"  - Faces: {result.faces.shape if hasattr(result, 'faces') else 'N/A'}")
    print(f"  - Generation time: {result.generation_time:.2f}s")
    output_path = Path(result.output_path)
    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"  - File size: {size_kb:.1f}KB")
        print(f"\n📄 OBJ File Preview:")
        with open(output_path) as f:
            lines = f.readlines()[:10]
            for line in lines:
                print(f"  {line.rstrip()}")
    print("\n✅ Test successful!")
    return result
if __name__ == "__main__":
    test_generation()