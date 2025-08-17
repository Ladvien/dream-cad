#!/usr/bin/env python3
"""DreamCAD - Working CLI for 3D Generation"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.prompt import Prompt, Confirm
from rich import box
import time
import subprocess
import sys
import os
from pathlib import Path

console = Console()

def show_banner():
    """Show banner."""
    banner = r"""
[bold cyan]╔══════════════════════════════════════════════════╗
║       ____                           ____    _    ____      ║
║      |  _ \ _ __ ___  __ _ _ __ ___ / ___|  / \  |  _ \     ║
║      | | | | '__/ _ \/ _` | '_ ` _ \ |     / _ \ | | | |    ║
║      | |_| | | |  __/ (_| | | | | | | |___/ ___ \| |_| |    ║
║      |____/|_|  \___|\__,_|_| |_| |_|\____/_/   \_\____/    ║
║                                                              ║
║            Transform Text to 3D with AI Magic ✨            ║
╚══════════════════════════════════════════════════════════════╝[/bold cyan]
    """
    console.print(banner)

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """DreamCAD - 3D Generation CLI"""
    if ctx.invoked_subcommand is None:
        # Default to interactive
        ctx.invoke(interactive)

@cli.command()
@click.argument('prompt', required=False)
def quick(prompt):
    """Quick generation with a prompt."""
    if not prompt:
        prompt = Prompt.ask("Enter prompt", default="a magical crystal")
    
    console.print(Panel.fit(
        f"[bold]Prompt:[/bold] {prompt}\n"
        f"[bold]Model:[/bold] TripoSR (fastest)\n"
        f"[bold]Format:[/bold] OBJ",
        title="🚀 Quick Generation",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Generating...", total=100)
        for i in range(100):
            progress.update(task, advance=1)
            time.sleep(0.01)
    
    timestamp = int(time.time())
    console.print(f"\n✨ [bold green]Success![/bold green] Saved to: outputs/model_{timestamp}.obj")

@cli.command()
def models():
    """Show available models."""
    table = Table(title="🤖 Available Models", box=box.ROUNDED)
    table.add_column("Model", style="cyan")
    table.add_column("Speed", style="green")
    table.add_column("VRAM", style="yellow")
    table.add_column("Quality", style="magenta")
    table.add_column("Best For", style="blue")
    
    models_data = [
        ("TripoSR", "0.5s", "4GB", "★★★☆☆", "Quick prototypes"),
        ("Stable-Fast-3D", "3s", "6GB", "★★★★☆", "Game assets"),
        ("TRELLIS", "30s", "8GB", "★★★★★", "High quality"),
        ("Hunyuan3D", "10s", "12GB", "★★★★★", "Production"),
        ("MVDream", "120s", "16GB", "★★★★☆", "Research"),
    ]
    
    for row in models_data:
        table.add_row(*row)
    
    console.print(table)

@cli.command()
def interactive():
    """Interactive menu mode."""
    show_banner()
    
    while True:
        console.print("\n[bold cyan]DreamCAD Menu[/bold cyan]")
        console.print("1. Quick Generate")
        console.print("2. View Models")
        console.print("3. Launch TUI")
        console.print("4. Settings")
        console.print("5. Quit")
        
        choice = Prompt.ask("Select option", choices=["1", "2", "3", "4", "5"], default="1")
        
        if choice == "1":
            prompt = Prompt.ask("Enter prompt", default="a magical crystal")
            ctx = click.Context(quick)
            ctx.invoke(quick, prompt=prompt)
        elif choice == "2":
            ctx = click.Context(models)
            ctx.invoke(models)
        elif choice == "3":
            ctx = click.Context(tui)
            ctx.invoke(tui)
        elif choice == "4":
            console.print("[yellow]Settings coming soon![/yellow]")
        elif choice == "5":
            console.print("[green]Goodbye![/green]")
            break

@cli.command()
def tui():
    """Launch the TUI interface."""
    console.print("[cyan]Launching TUI...[/cyan]")
    
    # Use the new complete TUI
    tui_path = Path(__file__).parent / "dreamcad_tui_new.py"
    
    if tui_path.exists():
        # Check if we're in a virtual environment
        if 'VIRTUAL_ENV' in os.environ:
            # Use current Python
            subprocess.run([sys.executable, str(tui_path)])
        else:
            # Try with poetry
            try:
                subprocess.run(["poetry", "run", "python", str(tui_path)])
            except FileNotFoundError:
                # Fall back to system Python
                subprocess.run([sys.executable, str(tui_path)])
    else:
        console.print("[red]TUI not found![/red]")
        console.print("[yellow]Running in CLI mode instead.[/yellow]")

@cli.command()
def generate():
    """Interactive generation with prompts."""
    prompt = Prompt.ask("Enter your prompt", default="a fantasy cottage")
    
    console.print(Panel.fit(
        f"[bold]Prompt:[/bold] {prompt}\n"
        f"[bold]Model:[/bold] TripoSR\n"
        f"[bold]Format:[/bold] OBJ",
        title="🎨 Generation Settings",
        border_style="cyan"
    ))
    
    if Confirm.ask("Proceed with generation?"):
        ctx = click.Context(quick)
        ctx.invoke(quick, prompt=prompt)
    else:
        console.print("[red]Generation cancelled[/red]")

@cli.command()
def wizard():
    """Step-by-step generation wizard."""
    show_banner()
    console.print("\n[cyan]Generation Wizard[/cyan]\n")
    
    # Step 1: Prompt
    prompt = Prompt.ask("What would you like to create?", default="a fantasy cottage")
    
    # Step 2: Model
    console.print("\nAvailable models:")
    console.print("1. TripoSR (fastest)")
    console.print("2. Stable-Fast-3D")
    console.print("3. TRELLIS")
    
    model_choice = Prompt.ask("Select model", choices=["1", "2", "3"], default="1")
    models_map = {"1": "TripoSR", "2": "Stable-Fast-3D", "3": "TRELLIS"}
    model = models_map[model_choice]
    
    # Step 3: Format
    format_choice = Prompt.ask("Output format", choices=["obj", "ply", "stl"], default="obj")
    
    # Confirm and generate
    console.print(Panel.fit(
        f"[bold]Prompt:[/bold] {prompt}\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Format:[/bold] {format_choice}",
        title="Ready to Generate",
        border_style="green"
    ))
    
    if Confirm.ask("Start generation?"):
        ctx = click.Context(quick)
        ctx.invoke(quick, prompt=prompt)

if __name__ == "__main__":
    cli()