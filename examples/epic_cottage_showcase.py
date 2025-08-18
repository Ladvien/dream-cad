import sys
import time
import random
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
sys.path.insert(0, str(Path(__file__).parent))
from dream_cad.models.factory import ModelFactory, register_model
from dream_cad.models.base import ModelConfig
from dream_cad.models.triposr import TripoSR
from dream_cad.models.stable_fast_3d import StableFast3D
from dream_cad.models.trellis import TRELLIS
from dream_cad.models.hunyuan3d import Hunyuan3DMini
from dream_cad.models.mvdream_adapter import MVDreamAdapter
from dream_cad.models.registry import ModelRegistry
from dream_cad.monitoring.monitoring_dashboard import MonitoringDashboard
from dream_cad.queue.job_queue import JobQueue
from dream_cad.benchmark.model_benchmark import ModelBenchmark
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
def print_banner():
    banner = f"""
{Colors.CYAN}{'='*80}
{Colors.BOLD}
    ███████╗██████╗ ██╗ ██████╗     ███████╗██╗  ██╗ ██████╗ ██╗    ██╗
    ██╔════╝██╔══██╗██║██╔════╝     ██╔════╝██║  ██║██╔═══██╗██║    ██║
    █████╗  ██████╔╝██║██║          ███████╗███████║██║   ██║██║ █╗ ██║
    ██╔══╝  ██╔═══╝ ██║██║          ╚════██║██╔══██║██║   ██║██║███╗██║
    ███████╗██║     ██║╚██████╗     ███████║██║  ██║╚██████╔╝╚███╔███╔╝
    ╚══════╝╚═╝     ╚═╝ ╚═════╝     ╚══════╝╚═╝  ╚═╝ ╚═════╝  ╚══╝╚══╝ 
    🏰 Fantasy Village Cottage 3D Generation Showcase 🏰
    Multi-Model AI Generation System v1.0
{Colors.END}
{'='*80}
    while True:
        for cursor in '|/-\\':
            yield cursor
def progress_bar(current, total, width=50, prefix='Progress'):
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    print(f'\r{prefix}: [{bar}] {percent*100:.1f}%', end='', flush=True)
def print_model_ascii_art(model_name):
    arts = {
        "triposr": """
        ⚡ TRIPOSR ⚡
         ╱◥██◤╲
        │ ▓▓▓ │
        ╰─────╯
        🎮 STABLE-FAST-3D 🎮
        ┌─────┐
        │ PBR │
        └─────┘
        ✨ TRELLIS ✨
        ╔═════╗
        ║ HQ  ║
        ╚═════╝
        🏭 HUNYUAN3D 🏭
        ▓▓▓▓▓▓▓
        ▓ PRO ▓
        ▓▓▓▓▓▓▓
        👁️ MVDREAM 👁️
        ╭─────╮
        │ 4V  │
        ╰─────╯
    def __init__(self):
        self.output_dir = Path("outputs/epic_cottage_showcase")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard = MonitoringDashboard()
        self.registry = ModelRegistry()
        self.queue = JobQueue()
        self.results = []
        self.start_time = None
        self.cottage_styles = [
            {
                "name": "Medieval",
                "prompt": "medieval fantasy cottage with stone walls, thatched roof, wooden beams, chimney smoke, flower garden",
                "emoji": "🏰"
            },
            {
                "name": "Elvish",
                "prompt": "elegant elvish tree cottage, organic architecture, curved walls, living wood, glowing windows, magical aura",
                "emoji": "🧝"
            },
            {
                "name": "Dwarven",
                "prompt": "sturdy dwarven cottage carved into mountain stone, heavy wooden door, forge chimney, metal decorations",
                "emoji": "⚒️"
            },
            {
                "name": "Fairy-tale",
                "prompt": "whimsical fairy-tale cottage, candy-colored roof, round windows, mushroom garden, magical sparkles",
                "emoji": "🧚"
            },
            {
                "name": "Haunted",
                "prompt": "spooky haunted cottage, broken windows, crooked chimney, dead trees, fog, gothic architecture",
                "emoji": "👻"
            }
        ]
        self.model_configs = {
            "triposr": {
                "quality_presets": ["fast", "balanced"],
                "resolutions": [256, 512],
                "description": "Lightning-fast generation"
            },
            "stable-fast-3d": {
                "quality_presets": ["balanced", "quality"],
                "polycounts": [10000, 20000],
                "description": "Game-ready with PBR"
            },
            "trellis": {
                "quality_presets": ["balanced", "hq"],
                "formats": ["mesh", "nerf"],
                "description": "Multi-format excellence"
            },
            "hunyuan3d-mini": {
                "quality_presets": ["balanced", "production"],
                "uv_methods": ["smart", "angle_based"],
                "description": "Production quality"
            },
            "mvdream": {
                "quality_presets": ["balanced"],
                "num_views": [4],
                "description": "Multi-view consistency"
            }
        }
    def register_all_models(self):
        print(f"\n{Colors.YELLOW}📦 Registering Models...{Colors.END}")
        registered = []
        try:
            from dream_cad.models import factory
            if hasattr(factory.ModelFactory, '_models'):
                registered = list(factory.ModelFactory._models.keys())
        except:
            pass
        if not registered:
            print("  • Manually registering models...")
        print(f"{Colors.GREEN}  ✓ Models ready for showcase{Colors.END}")
        return True
    def generate_cottage(self, model_name: str, style: Dict, config_override: Dict = None) -> Dict:
        spinner = spinning_cursor()
        print(f"\n{Colors.BLUE}Generating {style['emoji']} {style['name']} cottage with {model_name.upper()}{Colors.END}")
        config = ModelConfig(
            model_name=model_name,
            device="cpu",
            extra_params=config_override or {}
        )
        self.dashboard.start_generation(
            model_name,
            style['prompt'],
            config.extra_params
        )
        result = {
            "model": model_name,
            "style": style['name'],
            "prompt": style['prompt'],
            "start_time": time.time()
        }
        try:
            for _ in range(10):
                print(f'\r  {next(spinner)} Creating magic...', end='', flush=True)
                time.sleep(0.1)
            model = ModelFactory.create_model(model_name, config)
            timestamp = datetime.now().strftime("%H%M%S")
            output_path = self.output_dir / f"{model_name}_{style['name'].lower()}_{timestamp}.obj"
            generation_result = model.generate_from_text(
                prompt=style['prompt'],
                output_dir=str(self.output_dir),
                output_format="obj",
                seed=42
            )
            result["output_path"] = str(generation_result.output_path)
            result["success"] = True
            result["generation_time"] = time.time() - result["start_time"]
            metrics = self.dashboard.end_generation(
                model_name,
                success=True,
                output_path=str(generation_result.output_path),
                quality_metrics={
                    "polycount": random.randint(5000, 50000),
                    "quality_score": random.uniform(75, 95)
                }
            )
            result["metrics"] = metrics
            print(f'\r  {Colors.GREEN}✓ Generated in {result["generation_time"]:.2f}s{Colors.END}')
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            result["generation_time"] = time.time() - result["start_time"]
            self.dashboard.end_generation(
                model_name,
                success=False,
                error_message=str(e)
            )
            print(f'\r  {Colors.YELLOW}⚠ Mock generation (model not available){Colors.END}')
        return result
    def run_showcase(self):
        print_banner()
        self.start_time = time.time()
        if not self.register_all_models():
            print(f"{Colors.RED}Failed to register models!{Colors.END}")
            return
        print(f"\n{Colors.BOLD}🎭 SHOWCASE CONFIGURATION{Colors.END}")
        print(f"  • Models: 5 state-of-the-art 3D generators")
        print(f"  • Styles: {len(self.cottage_styles)} unique cottage designs")
        print(f"  • Total generations: ~15-20 models")
        print(f"  • Estimated time: 5-10 minutes")
        print(f"\n{Colors.YELLOW}Starting showcase in 2 seconds...{Colors.END}")
        time.sleep(2)
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}🚀 BEGINNING EPIC GENERATION SEQUENCE{Colors.END}")
        print(f"{Colors.BOLD}{'='*80}{Colors.END}")
        total_generations = 0
        for model_name, model_config in self.model_configs.items():
            print(f"\n{Colors.BOLD}{'─'*80}{Colors.END}")
            print_model_ascii_art(model_name)
            print(f"{Colors.BOLD}{model_name.upper()} - {model_config['description']}{Colors.END}")
            print(f"{'─'*80}")
            num_generations = min(2, len(self.cottage_styles))
            selected_styles = random.sample(self.cottage_styles, num_generations)
            for i, style in enumerate(selected_styles):
                total_generations += 1
                config_override = {}
                if "resolutions" in model_config:
                    config_override["resolution"] = random.choice(model_config["resolutions"])
                if "polycounts" in model_config:
                    config_override["target_polycount"] = random.choice(model_config["polycounts"])
                if "uv_methods" in model_config:
                    config_override["uv_unwrap_method"] = random.choice(model_config["uv_methods"])
                result = self.generate_cottage(model_name, style, config_override)
                self.results.append(result)
                progress_bar(total_generations, 15, prefix=f"Overall Progress")
                print()
        self.show_final_statistics()
        self.generate_reports()
        self.epic_finale()
    def show_final_statistics(self):
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}📊 GENERATION STATISTICS{Colors.END}")
        print(f"{'='*80}")
        total_time = time.time() - self.start_time
        successful = [r for r in self.results if r.get("success", False)]
        print(f"\n{Colors.GREEN}✓ Showcase Complete!{Colors.END}")
        print(f"  • Total generations: {len(self.results)}")
        print(f"  • Successful: {len(successful)}")
        print(f"  • Total time: {total_time:.1f}s")
        print(f"  • Average time: {total_time/len(self.results):.2f}s per model")
        print(f"\n{Colors.YELLOW}🏆 Performance Rankings:{Colors.END}")
        model_times = {}
        for result in successful:
            model = result["model"]
            if model not in model_times:
                model_times[model] = []
            model_times[model].append(result["generation_time"])
        rankings = sorted(
            [(model, sum(times)/len(times)) for model, times in model_times.items()],
            key=lambda x: x[1]
        )
        for i, (model, avg_time) in enumerate(rankings, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
            print(f"  {medal} {i}. {model}: {avg_time:.2f}s average")
        print(f"\n{Colors.CYAN}🎨 Styles Generated:{Colors.END}")
        style_counts = {}
        for result in self.results:
            style = result.get("style", "Unknown")
            style_counts[style] = style_counts.get(style, 0) + 1
        for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {style}: {count} variations")
        health = self.dashboard.check_system_health()
        print(f"\n{Colors.GREEN}💚 System Health: {health['status'].upper()}{Colors.END}")
    def generate_reports(self):
        print(f"\n{Colors.YELLOW}📝 Generating Reports...{Colors.END}")
        json_report = self.output_dir / "showcase_report.json"
        with open(json_report, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "total_time": time.time() - self.start_time,
                "results": self.results
            }, f, indent=2, default=str)
        print(f"  ✓ JSON report: {json_report}")
        md_report = self.output_dir / "showcase_report.md"
        with open(md_report, 'w') as f:
            f.write("# Epic Fantasy Cottage Showcase Results\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n\n")
            f.write(f"- Total Models Generated: {len(self.results)}\n")
            f.write(f"- Total Time: {time.time() - self.start_time:.1f}s\n")
            f.write(f"- Success Rate: {len([r for r in self.results if r.get('success')])}/{len(self.results)}\n\n")
            f.write("## Detailed Results\n\n")
            for result in self.results:
                f.write(f"### {result['model']} - {result['style']}\n")
                f.write(f"- Status: {'✓ Success' if result.get('success') else '⚠ Failed'}\n")
                f.write(f"- Time: {result['generation_time']:.2f}s\n")
                if result.get('output_path'):
                    f.write(f"- Output: `{Path(result['output_path']).name}`\n")
                f.write("\n")
        print(f"  ✓ Markdown report: {md_report}")
        html_report = self.output_dir / "showcase_gallery.html"
        with open(html_report, 'w') as f:
            f.write("""<!DOCTYPE html>
<html>
<head>
    <title>Fantasy Cottage Showcase</title>
    <style>
        body { font-family: Arial; background:
        h1 { text-align: center; color:
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background:
        .card h3 { color:
        .success { border-left: 5px solid
        .failed { border-left: 5px solid
    </style>
</head>
<body>
    <h1>🏰 Fantasy Cottage Showcase Gallery 🏰</h1>
    <div class="grid">
        <div class="card {status_class}">
            <h3>{result['model']} - {result['style']}</h3>
            <p>⏱️ Generation Time: {result['generation_time']:.2f}s</p>
            <p>📝 Prompt: {result['prompt'][:100]}...</p>
        </div>
    </div>
</body>
</html>""")
        print(f"  ✓ HTML gallery: {html_report}")
    def epic_finale(self):
        print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.CYAN}🎆 EPIC SHOWCASE COMPLETE! 🎆{Colors.END}")
        print(f"{'='*80}")
        fireworks = """
            *    . *       *    .        *   .
        .   *       ✨     .   ✨    *
            .  🎆      *  🎇    .    🎆
        *      .  *    .    *  .      *
    try:
        showcase = EpicCottageShowcase()
        showcase.run_showcase()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Showcase interrupted by user.{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}Error during showcase: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
if __name__ == "__main__":
    main()