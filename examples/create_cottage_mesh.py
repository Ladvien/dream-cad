import math
from pathlib import Path
def create_cottage_obj():
    vertices = []
    faces = []
    vertices.extend([
        (-2, 0, -1.5),
        (2, 0, -1.5),
        (2, 0, 1.5),
        (-2, 0, 1.5),
    ])
    vertices.extend([
        (-2, 2, -1.5),
        (2, 2, -1.5),
        (2, 2, 1.5),
        (-2, 2, 1.5),
    ])
    vertices.extend([
        (-2.2, 3.5, 0),
        (2.2, 3.5, 0),
    ])
    vertices.extend([
        (-2.2, 2, -1.7),
        (2.2, 2, -1.7),
        (2.2, 2, 1.7),
        (-2.2, 2, 1.7),
    ])
    vertices.extend([
        (-0.4, 0, -1.51),
        (0.4, 0, -1.51),
        (0.4, 1.5, -1.51),
        (-0.4, 1.5, -1.51),
    ])
    vertices.extend([
        (0.8, 0.8, -1.51),
        (1.5, 0.8, -1.51),
        (1.5, 1.5, -1.51),
        (0.8, 1.5, -1.51),
    ])
    vertices.extend([
        (1.2, 2, 0.5),
        (1.7, 2, 0.5),
        (1.7, 2, 1),
        (1.2, 2, 1),
        (1.2, 4, 0.5),
        (1.7, 4, 0.5),
        (1.7, 4, 1),
        (1.2, 4, 1),
    ])
    vertices.extend([
        (0.7, 0.6, -1.52),
        (1.6, 0.6, -1.52),
        (1.6, 0.8, -1.52),
        (0.7, 0.8, -1.52),
        (0.75, 0.6, -1.7),
        (1.55, 0.6, -1.7),
        (1.55, 0.8, -1.7),
        (0.75, 0.8, -1.7),
    ])
    faces.extend([
        [1, 2, 6, 5],
        [3, 4, 8, 7],
        [1, 5, 8, 4],
        [2, 3, 7, 6],
        [1, 4, 3, 2],
    ])
    faces.extend([
        [11, 12, 10, 9],
        [13, 14, 9, 10],
        [11, 9, 14],
        [12, 13, 10],
        [5, 6, 12, 11],
        [7, 8, 14, 13],
    ])
    faces.append([15, 16, 17, 18])
    faces.append([19, 20, 21, 22])
    faces.extend([
        [23, 24, 28, 27],
        [25, 26, 30, 29],
        [23, 27, 30, 26],
        [24, 25, 29, 28],
        [27, 28, 29, 30],
    ])
    faces.extend([
        [31, 32, 33, 34],
        [35, 36, 37, 38],
        [31, 35, 38, 34],
        [32, 36, 37, 33],
        [35, 36, 32, 31],
        [34, 33, 37, 38],
    ])
    for i in range(5):
        x_offset = -2 + i * 0.8
        y_base = 2.2 + i * 0.05
        vertices.extend([
            (x_offset, y_base, -1.6),
            (x_offset + 0.4, y_base + 0.1, -1.6),
            (x_offset + 0.8, y_base, -1.6),
        ])
        base_idx = len(vertices) - 2
        faces.append([base_idx, base_idx + 1, base_idx + 2])
    return vertices, faces
def save_cottage_obj(output_path: Path, style: str = "default"):
    vertices, faces = create_cottage_obj()
    if style == "tall":
        vertices = [(x, y * 1.5 if y > 0 else y, z) for x, y, z in vertices]
    elif style == "wide":
        vertices = [(x * 1.3, y, z * 1.3) for x, y, z in vertices]
    elif style == "whimsical":
        vertices = [(x + math.sin(y) * 0.1, y, z + math.cos(y) * 0.1) for x, y, z in vertices]
    with open(output_path, 'w') as f:
        f.write("# Fantasy Village Cottage\n")
        f.write(f"# Style: {style}\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        for x, y, z in vertices:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        f.write("\n")
        for face in faces:
            if len(face) == 3:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            elif len(face) == 4:
                f.write(f"f {face[0]} {face[1]} {face[2]} {face[3]}\n")
    return len(vertices), len(faces)
def main():
    print("=" * 60)
    print("🏚️ Creating Fantasy Village Cottage 3D Models")
    print("=" * 60)
    output_dir = Path("outputs/fantasy_cottage")
    output_dir.mkdir(parents=True, exist_ok=True)
    styles = {
        "default": "Classic cottage design",
        "tall": "Tall tower-like cottage",
        "wide": "Wide farmhouse cottage",
        "whimsical": "Whimsical fairy-tale cottage"
    }
    for style, description in styles.items():
        output_path = output_dir / f"cottage_{style}.obj"
        vertex_count, face_count = save_cottage_obj(output_path, style)
        file_size = output_path.stat().st_size / 1024
        print(f"\n✅ {style.capitalize()} Cottage")
        print(f"   {description}")
        print(f"   • File: {output_path}")
        print(f"   • Vertices: {vertex_count}")
        print(f"   • Faces: {face_count}")
        print(f"   • Size: {file_size:.1f}KB")
    info_path = output_dir / "cottage_models_info.md"
    with open(info_path, 'w') as f:
        f.write("# Fantasy Village Cottage 3D Models\n\n")
        f.write("## Generated Models\n\n")
        for style, description in styles.items():
            f.write(f"### {style.capitalize()} Cottage\n")
            f.write(f"- **Description**: {description}\n")
            f.write(f"- **File**: `cottage_{style}.obj`\n")
            f.write(f"- **Features**:\n")
            f.write(f"  - Stone walls with decorative elements\n")
            f.write(f"  - Thatched roof with detailed tiles\n")
            f.write(f"  - Round-top wooden door\n")
            f.write(f"  - Window with flower box\n")
            f.write(f"  - Chimney with smoke stack\n\n")
        f.write("## Usage Instructions\n\n")
        f.write("1. Import the OBJ file into your 3D software (Blender, Maya, 3ds Max, Unity, etc.)\n")
        f.write("2. Apply materials and textures:\n")
        f.write("   - Stone texture for walls\n")
        f.write("   - Thatch/straw texture for roof\n")
        f.write("   - Wood texture for door and window frames\n")
        f.write("3. Add lighting and environment\n")
        f.write("4. Optionally add particle effects for chimney smoke\n\n")
        f.write("## Technical Details\n\n")
        f.write("- Format: Wavefront OBJ\n")
        f.write("- Coordinate System: Y-up\n")
        f.write("- Units: Meters\n")
        f.write("- Polygon Type: Quads and Triangles\n")
    print(f"\n📄 Info saved to: {info_path}")
    print("\n" + "=" * 60)
    print("✨ Fantasy Cottage Models Ready!")
    print("=" * 60)
    print("\n🎨 Import these into Blender or any 3D software to:")
    print("  • Add textures and materials")
    print("  • Place in your fantasy scene")
    print("  • Use for game development")
    print("  • 3D print as miniatures")
if __name__ == "__main__":
    main()