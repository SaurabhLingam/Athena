"""
Athena CLI — entry point for `athena run`, `athena info`, `athena doctor`.
Lives at backend/cli.py
"""

import time
import webbrowser
import threading
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.align import Align
from rich.rule import Rule
from rich import box
from rich.padding import Padding

app = typer.Typer(
    name="athena",
    help="Athena — Automated EDA & ML Readiness Platform",
    add_completion=False,
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()

BANNER = """
     █████╗ ████████╗██╗  ██╗███████╗███╗   ██╗ █████╗ 
 ██╔══██╗╚══██╔══╝██║  ██║██╔════╝████╗  ██║██╔══██╗
 ███████║   ██║   ███████║█████╗  ██╔██╗ ██║███████║
 ██╔══██║   ██║   ██╔══██║██╔══╝  ██║╚██╗██║██╔══██║
 ██║  ██║   ██║   ██║  ██║███████╗██║ ╚████║██║  ██║
 ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═══╝╚═╝  ╚═╝
"""

def _banner():
    console.print()
    console.print(Align.center(Text(BANNER.strip(), style="bold cyan")))
    console.print(Align.center(Text("Automated EDA  ·  ML Readiness  ·  Data Intelligence", style="dim cyan")))
    console.print()


# ─────────────────────────────────────────────
#  athena run
# ─────────────────────────────────────────────

@app.command()
def run(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    no_browser: bool = typer.Option(False, "--no-browser", help="Don't auto-open browser"),
    reload: bool = typer.Option(False, "--reload", help="Hot-reload for dev"),
):
    """
    [bold cyan]Launch Athena[/bold cyan] locally in your browser.

    Starts the backend server and opens the UI automatically.
    """
    _banner()

    info = Table.grid(padding=(0, 2))
    info.add_column(style="dim")
    info.add_column(style="cyan bold")
    info.add_row("address",  f"http://{host}:{port}")
    info.add_row("mode",     "[green]dev — hot reload[/green]" if reload else "[blue]production[/blue]")
    info.add_row("browser",  "[green]opens automatically[/green]" if not no_browser else "[yellow]manual[/yellow]")

    console.print(Panel(info, title="[bold cyan]⚡ Starting Athena[/bold cyan]", border_style="cyan", padding=(1, 2)))
    console.print()

    steps = [
        ("Loading core modules",    0.25),
        ("Initialising EDA engine", 0.25),
        ("Mounting API routes",     0.20),
        ("Serving frontend assets", 0.20),
        ("Ready",                   0.10),
    ]

    with Progress(
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30, style="cyan", complete_style="bold cyan"),
        TextColumn("[cyan]{task.percentage:>3.0f}%[/cyan]"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Booting…[/cyan]", total=len(steps))
        for label, delay in steps:
            progress.update(task, description=f"[cyan]{label}…[/cyan]", advance=1)
            time.sleep(delay)

    console.print(
        Panel(
            Align.center(
                Text.assemble(
                    ("✦ Athena is running\n\n", "bold cyan"),
                    ("http://", "dim"),
                    (f"{host}:{port}", "bold white underline"),
                    ("\n\nPress ", "dim"),
                    ("Ctrl+C", "bold white"),
                    (" to stop", "dim"),
                )
            ),
            border_style="cyan",
            padding=(1, 4),
        )
    )
    console.print()

    if not no_browser:
        def _open():
            time.sleep(3)
            webbrowser.open(f"http://{host}:{port}")
        threading.Thread(target=_open, daemon=True).start()

    try:
        import uvicorn
        _mount_static()
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=reload,
            log_level="warning",
        )
    except ImportError as e:
        console.print(f"\n[bold red]✗ Missing dependency:[/bold red] {e}")
        console.print("[dim]Run [bold]pip install athena-eda[/bold] to fix.[/dim]\n")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        _goodbye()


def _mount_static():
    """
    Attach backend/static/ to the FastAPI app so the bundled React UI
    is served at localhost. On Railway, static/ doesn't exist so this
    is silently skipped — Railway is unaffected.
    """
    static_dir = Path(__file__).parent / "static"
    if not static_dir.exists():
        console.print(
            Panel(
                Text.assemble(
                    ("[yellow]No frontend bundle found[/yellow] at [bold]backend/static/[/bold]\n\n", ""),
                    ("Run this first:\n", "dim"),
                    ("  bash scripts/build_frontend.sh", "bold white"),
                ),
                border_style="yellow",
                padding=(1, 2),
            )
        )
        return

    try:
        from backend.main import app as fastapi_app
        from fastapi.staticfiles import StaticFiles
        from fastapi.responses import FileResponse

        assets = static_dir / "assets"
        if assets.exists():
            # Only mount if not already mounted (avoids duplicate route error on --reload)
            existing = [r.name for r in getattr(fastapi_app, "routes", []) if hasattr(r, "name")]
            if "assets" not in existing:
                fastapi_app.mount("/assets", StaticFiles(directory=assets), name="assets")

        index = static_dir / "index.html"

        # Add SPA catch-all only once
        spa_routes = [r for r in fastapi_app.routes if getattr(r, "path", "") == "/{full_path:path}"]
        if not spa_routes:
            @fastapi_app.get("/{full_path:path}", include_in_schema=False)
            async def _spa(full_path: str):
                return FileResponse(index)

    except Exception as e:
        console.print(f"[dim yellow]Warning: could not mount static files — {e}[/dim yellow]")


def _goodbye():
    console.print()
    console.print(
        Panel(
            Align.center(
                Text.assemble(
                    ("Thanks for using Athena  ", "dim"),
                    ("✦", "cyan"),
                    ("\nSee you next time.", "dim"),
                )
            ),
            border_style="dim cyan",
            padding=(1, 4),
        )
    )
    console.print()


# ─────────────────────────────────────────────
#  athena info
# ─────────────────────────────────────────────

@app.command()
def info():
    """Show [bold cyan]version info[/bold cyan] and installed packages."""
    _banner()
    from importlib.metadata import version, PackageNotFoundError

    def _v(pkg):
        try: return version(pkg)
        except PackageNotFoundError: return "[dim]not installed[/dim]"

    core = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    core.add_column(style="dim", width=22)
    core.add_column(style="white")
    for pkg in ["athena-eda", "fastapi", "pandas", "scikit-learn", "lightgbm", "nltk", "slowapi"]:
        core.add_row(pkg, _v(pkg))

    optional = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    optional.add_column(style="dim", width=22)
    optional.add_column()
    for pkg, label in [
        ("statsmodels", "Time series EDA"),
        ("pmdarima",    "Auto-ARIMA"),
        ("mlflow",      "MLflow tracking"),
        ("xgboost",     "XGBoost"),
    ]:
        try:
            v = version(pkg)
            optional.add_row(label, f"[green]✓  {v}[/green]")
        except PackageNotFoundError:
            optional.add_row(label, "[yellow]✗  not installed[/yellow]")

    console.print(Panel(
        Columns([
            Panel(core,     title="[cyan]core[/cyan]",     border_style="dim", padding=(0, 1)),
            Panel(optional, title="[cyan]optional[/cyan]", border_style="dim", padding=(0, 1)),
        ]),
        title="[bold cyan]Athena — package info[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()
    console.print(Padding(
        Text.assemble(("Install extras:  ", "dim"), ("pip install 'athena-eda[all]'", "bold white")),
        (0, 2),
    ))
    console.print()


# ─────────────────────────────────────────────
#  athena version
# ─────────────────────────────────────────────

@app.command()
def version():
    """Print the installed Athena version."""
    try:
        from importlib.metadata import version as _v
        v = _v("athena-eda")
    except Exception:
        v = "0.1.0 (dev)"
    console.print(f"\n  [cyan]athena-eda[/cyan] [bold white]{v}[/bold white]\n")

@app.command()
def update():
    """Update Athena to the [bold cyan]latest version[/bold cyan]."""
    import subprocess
    _banner()
    console.print(Panel("[bold cyan]Checking for updates…[/bold cyan]", border_style="cyan", padding=(0, 2)))
    console.print()

    try:
        result = subprocess.run(
            ["pip", "install", "--upgrade", "athena-eda"],
            capture_output=True,
            text=True,
        )
        if "Successfully installed" in result.stdout:
            # Extract new version from output
            for word in result.stdout.split():
                if word.startswith("athena-eda-"):
                    new_version = word.replace("athena-eda-", "")
                    break
            else:
                new_version = "latest"
            console.print(
                Panel(
                    Align.center(Text.assemble(
                        ("✦ Athena updated to ", "dim"),
                        (new_version, "bold cyan"),
                    )),
                    border_style="cyan",
                    padding=(1, 4),
                )
            )
        elif "already up-to-date" in result.stdout.lower() or "already satisfied" in result.stdout.lower():
            console.print(
                Panel(
                    Align.center(Text("✦ Athena is already up to date.", style="bold green")),
                    border_style="green",
                    padding=(0, 2),
                )
            )
        else:
            console.print(result.stdout)
    except Exception as e:
        console.print(f"\n[bold red]✗ Update failed:[/bold red] {e}\n")
        console.print("[dim]Try manually: [bold]pip install --upgrade athena-eda[/bold][/dim]\n")
    console.print()


# ─────────────────────────────────────────────
#  athena doctor
# ─────────────────────────────────────────────

@app.command()
def doctor():
    """[bold cyan]Diagnose[/bold cyan] your Athena installation."""
    _banner()
    console.print(Panel("[bold cyan]Running diagnostics…[/bold cyan]", border_style="cyan", padding=(0, 2)))
    console.print()

    checks = [
        ("Python ≥ 3.9",            _chk_python),
        ("fastapi",                 lambda: _chk_import("fastapi")),
        ("uvicorn",                 lambda: _chk_import("uvicorn")),
        ("pandas",                  lambda: _chk_import("pandas")),
        ("scikit-learn",            lambda: _chk_import("sklearn")),
        ("lightgbm",                lambda: _chk_import("lightgbm")),
        ("nltk",                    lambda: _chk_import("nltk")),
        ("slowapi",                 lambda: _chk_import("slowapi")),
        ("python-dotenv",           lambda: _chk_import("dotenv")),
        ("backend/static/ exists",  _chk_static),
        ("Port 8000 free",          _chk_port),
    ]

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    table.add_column(width=28)
    table.add_column(width=14)
    table.add_column()

    all_ok = True
    for label, fn in checks:
        ok, note = fn()
        status = "[bold green]  ✓  pass[/bold green]" if ok else "[bold red]  ✗  fail[/bold red]"
        table.add_row(label, status, f"[dim]{note}[/dim]" if note else "")
        if not ok:
            all_ok = False

    console.print(table)
    console.print()

    if all_ok:
        console.print(Panel(
            Align.center(Text("✦ All checks passed — Athena is ready to run!", style="bold green")),
            border_style="green", padding=(0, 2),
        ))
    else:
        console.print(Panel(
            Align.center(Text("Some checks failed — see notes above.", style="yellow")),
            border_style="yellow", padding=(0, 2),
        ))
    console.print()


def _chk_python():
    ok = sys.version_info >= (3, 9)
    return ok, f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

def _chk_import(name):
    try: __import__(name); return True, ""
    except ImportError: return False, f"pip install {name}"

def _chk_static():
    p = Path(__file__).parent / "static" / "index.html"
    if p.exists(): return True, str(p.parent)
    return False, "run  bash scripts/build_frontend.sh"

def _chk_port(host="127.0.0.1", port=8000):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        busy = s.connect_ex((host, port)) == 0
    return (not busy), f"port {port} {'in use — try --port 8001' if busy else 'available'}"


if __name__ == "__main__":
    app()