"""
Command Line Interface for Secure FL

This module provides command-line interfaces for running the Secure FL framework,
including server, client, experiment, and setup commands.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import click
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich import print as rprint
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from secure_fl.client import create_client, start_client

# Import centralized models
from secure_fl.models import CIFAR10Model, MNISTModel, SimpleModel
from secure_fl.server import SecureFlowerServer, create_server_strategy
from secure_fl.setup import SecureFLSetup

# Setup rich console and logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--quiet", "-q", is_flag=True, help="Suppress output")
def main(verbose: bool, quiet: bool):
    """
    🔐 Secure FL: Dual-Verifiable Federated Learning with Zero-Knowledge Proofs

    A complete framework for federated learning with ZKP verification using
    client-side zk-STARKs and server-side zk-SNARKs.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Display banner
    if not quiet:
        rprint(
            """
[bold blue]🔐 Secure FL Framework[/bold blue]
[dim]Dual-Verifiable Federated Learning with Zero-Knowledge Proofs[/dim]
        """
        )


@main.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Configuration file path"
)
@click.option("--host", default="localhost", help="Server host address")
@click.option("--port", default=8080, type=int, help="Server port")
@click.option("--rounds", "-r", default=10, type=int, help="Number of training rounds")
@click.option(
    "--min-clients", default=2, type=int, help="Minimum number of clients required"
)
@click.option(
    "--enable-zkp/--disable-zkp",
    default=True,
    help="Enable/disable zero-knowledge proofs",
)
@click.option(
    "--proof-rigor",
    type=click.Choice(["low", "medium", "high"]),
    default="high",
    help="ZKP rigor level",
)
@click.option(
    "--momentum", default=0.9, type=float, help="FedJSCM momentum coefficient"
)
@click.option(
    "--blockchain/--no-blockchain", default=False, help="Enable blockchain verification"
)
@click.option(
    "--model",
    type=click.Choice(["mnist", "cifar10", "custom"]),
    default="mnist",
    help="Model type to use",
)
def server(
    config: Optional[str],
    host: str,
    port: int,
    rounds: int,
    min_clients: int,
    enable_zkp: bool,
    proof_rigor: str,
    momentum: float,
    blockchain: bool,
    model: str,
):
    """Start the Secure FL server"""

    try:
        import torch.nn as nn
        import torch.nn.functional as F

        from .server import SecureFlowerServer, create_server_strategy
        from .utils import get_default_config

        # Load configuration
        if config:
            import yaml

            with open(config, "r") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = get_default_config()

        # Override with CLI arguments
        cfg["server"].update(
            {
                "host": host,
                "port": port,
                "num_rounds": rounds,
                "min_fit_clients": min_clients,
                "min_evaluate_clients": min_clients,
            }
        )

        cfg["aggregation"]["momentum"] = momentum
        cfg["zkp"].update(
            {
                "enable_zkp": enable_zkp,
                "proof_rigor": proof_rigor,
                "blockchain_verification": blockchain,
            }
        )

        # Define model based on choice using centralized models
        if model == "mnist":
            model_fn = lambda: MNISTModel()
        elif model == "cifar10":
            model_fn = lambda: CIFAR10Model()
        elif model == "simple":
            model_fn = lambda: SimpleModel(input_dim=784, output_dim=10)
        else:
            raise click.ClickException(
                f"Unknown model '{model}'. Available: mnist, cifar10, simple"
            )

        # Create server strategy
        strategy = create_server_strategy(
            model_fn=model_fn, **cfg["aggregation"], **cfg["zkp"]
        )

        # Create and start server
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting Secure FL Server...", total=None)

            server = SecureFlowerServer(strategy=strategy, **cfg["server"])

            progress.update(task, description="Server ready! Waiting for clients...")

            # Display server info
            table = Table(title="Server Configuration")
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Host", f"{host}:{port}")
            table.add_row("Rounds", str(rounds))
            table.add_row("Min Clients", str(min_clients))
            table.add_row("ZKP Enabled", "✓" if enable_zkp else "✗")
            table.add_row("Proof Rigor", proof_rigor)
            table.add_row("Momentum", str(momentum))
            table.add_row("Blockchain", "✓" if blockchain else "✗")

            console.print(table)

            server.start()

    except ImportError as e:
        raise click.ClickException(f"Missing dependencies: {e}")
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        raise click.ClickException(f"Server error: {e}")


@main.command()
@click.option(
    "--server-address",
    "-s",
    default="localhost:8080",
    help="Server address (host:port)",
)
@click.option("--client-id", "-i", required=True, help="Unique client identifier")
@click.option(
    "--dataset",
    type=click.Choice(["mnist", "cifar10", "synthetic"]),
    default="mnist",
    help="Dataset to use for training",
)
@click.option("--data-path", type=click.Path(), help="Path to custom dataset")
@click.option("--partition", type=int, help="Data partition index")
@click.option(
    "--enable-zkp/--disable-zkp",
    default=True,
    help="Enable/disable zero-knowledge proof generation",
)
@click.option(
    "--epochs", "-e", default=5, type=int, help="Local training epochs per round"
)
@click.option("--batch-size", "-b", default=32, type=int, help="Training batch size")
@click.option(
    "--learning-rate", "-lr", default=0.01, type=float, help="Local learning rate"
)
def client(
    server_address: str,
    client_id: str,
    dataset: str,
    data_path: Optional[str],
    partition: Optional[int],
    enable_zkp: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """Start a Secure FL client"""

    try:
        from .client import create_client, start_client
        from .utils import load_dataset

        # Load dataset
        if data_path:
            # Load custom dataset
            train_data, val_data = load_dataset(data_path, partition)
        else:
            # Load built-in dataset
            train_data, val_data = load_dataset(dataset, partition or 0)

        # Define model (should match server model)
        if dataset in ["mnist", "synthetic"]:
            if dataset != "custom":
                if dataset == "mnist":
                    model_fn = lambda: MNISTModel()
                elif dataset == "cifar10":
                    model_fn = lambda: CIFAR10Model()
                else:
                    model_fn = lambda: SimpleModel()

        # Create client
        client = create_client(
            client_id=client_id,
            model_fn=model_fn,
            train_data=train_data,
            val_data=val_data,
            enable_zkp=enable_zkp,
            local_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )

        # Display client info
        table = Table(title=f"Client {client_id} Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Server", server_address)
        table.add_row("Dataset", dataset)
        table.add_row("ZKP Enabled", "✓" if enable_zkp else "✗")
        table.add_row("Epochs", str(epochs))
        table.add_row("Batch Size", str(batch_size))
        table.add_row("Learning Rate", str(learning_rate))

        console.print(table)

        # Start client
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Connecting to server...", total=None)
            start_client(client, server_address)

    except ImportError as e:
        raise click.ClickException(f"Missing dependencies: {e}")
    except Exception as e:
        logger.error(f"Client failed to start: {e}")
        raise click.ClickException(f"Client error: {e}")


@main.command()
@click.option(
    "--config", "-c", type=click.Path(exists=True), help="Experiment configuration file"
)
@click.option(
    "--num-clients", "-n", default=3, type=int, help="Number of federated clients"
)
@click.option("--rounds", "-r", default=10, type=int, help="Training rounds")
@click.option(
    "--dataset",
    type=click.Choice(["mnist", "cifar10", "medmnist", "synthetic"]),
    default="synthetic",
    help="Dataset for experiment",
)
@click.option(
    "--enable-zkp/--disable-zkp", default=True, help="Enable/disable ZKP verification"
)
@click.option(
    "--proof-rigor",
    type=click.Choice(["low", "medium", "high"]),
    default="medium",
    help="ZKP rigor level",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./results",
    help="Output directory for results",
)
@click.option(
    "--visualize/--no-visualize", default=True, help="Generate visualization plots"
)
@click.option(
    "--save-models/--no-save-models", default=False, help="Save trained models"
)
def experiment(
    config: Optional[str],
    num_clients: int,
    rounds: int,
    dataset: str,
    enable_zkp: bool,
    proof_rigor: str,
    output_dir: str,
    visualize: bool,
    save_models: bool,
):
    """Run a complete federated learning experiment"""

    try:
        # Import experiment module
        import sys

        from .experiments.train_secure_fl import main as experiment_main

        # Prepare arguments for experiment script
        args = [
            "--num-clients",
            str(num_clients),
            "--rounds",
            str(rounds),
            "--dataset",
            dataset,
            "--proof-rigor",
            proof_rigor,
            "--output-dir",
            output_dir,
        ]

        if config:
            args.extend(["--config", config])

        if not enable_zkp:
            args.append("--disable-zkp")

        if visualize:
            args.append("--visualize")

        if save_models:
            args.append("--save-models")

        # Display experiment info
        table = Table(title="Experiment Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Clients", str(num_clients))
        table.add_row("Rounds", str(rounds))
        table.add_row("Dataset", dataset)
        table.add_row("ZKP Enabled", "✓" if enable_zkp else "✗")
        table.add_row("Proof Rigor", proof_rigor)
        table.add_row("Output Dir", output_dir)

        console.print(table)

        # Run experiment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Running experiment...", total=None)

            # Backup original sys.argv and replace
            original_argv = sys.argv
            sys.argv = ["experiment"] + args

            try:
                experiment_main()
                progress.update(
                    task, description="✓ Experiment completed successfully!"
                )
            finally:
                sys.argv = original_argv

        rprint(f"\n[green]✓ Experiment results saved to: {output_dir}[/green]")

    except ImportError as e:
        raise click.ClickException(f"Missing dependencies: {e}")
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise click.ClickException(f"Experiment error: {e}")


@main.command()
@click.option(
    "--action",
    type=click.Choice(["install", "zkp", "check", "clean", "full"]),
    default="check",
    help="Setup action to perform",
)
@click.option("--force", is_flag=True, help="Force reinstallation")
@click.option("--skip-zkp", is_flag=True, help="Skip ZKP tools installation")
def setup(action: str, force: bool, skip_zkp: bool):
    """Setup and configure Secure FL environment"""

    try:
        from .setup import SecureFLSetup

        setup_manager = SecureFLSetup()

        if action == "check":
            rprint("[bold]🔍 Checking System Requirements[/bold]")
            setup_manager.check_system_requirements()

        elif action == "install":
            rprint("[bold]📦 Installing Python Dependencies[/bold]")
            success = setup_manager.install_python_deps()
            if success:
                rprint("[green]✓ Python dependencies installed successfully[/green]")
            else:
                rprint("[red]✗ Failed to install some dependencies[/red]")

        elif action == "zkp":
            if skip_zkp:
                rprint("[yellow]⚠ Skipping ZKP tools installation[/yellow]")
                return

            rprint("[bold]🔐 Setting up ZKP Tools[/bold]")
            success = setup_manager.setup_zkp_tools()
            if success:
                rprint("[green]✓ ZKP tools setup completed[/green]")
            else:
                rprint("[yellow]⚠ ZKP tools setup completed with warnings[/yellow]")

        elif action == "clean":
            rprint("[bold]🧹 Cleaning temporary files[/bold]")
            setup_manager.clean()
            rprint("[green]✓ Cleanup completed[/green]")

        elif action == "full":
            rprint("[bold]🚀 Full Setup[/bold]")

            # Install Python deps
            rprint("1. Installing Python dependencies...")
            setup_manager.install_python_deps()

            # Setup ZKP tools (unless skipped)
            if not skip_zkp:
                rprint("2. Setting up ZKP tools...")
                setup_manager.setup_zkp_tools()
            else:
                rprint("2. Skipping ZKP tools...")

            # Create config
            rprint("3. Creating configuration...")
            setup_manager.create_config()

            # Run tests
            rprint("4. Running tests...")
            test_success = setup_manager.run_tests()

            if test_success:
                rprint("[green]✓ Full setup completed successfully![/green]")
                rprint("\n[bold]Next steps:[/bold]")
                rprint(
                    "1. Run a demo: [cyan]secure-fl experiment --dataset synthetic --rounds 3[/cyan]"
                )
                rprint("2. Start a server: [cyan]secure-fl server --rounds 5[/cyan]")
                rprint("3. Connect clients: [cyan]secure-fl client -i client_1[/cyan]")
            else:
                rprint("[yellow]⚠ Setup completed with test failures[/yellow]")

    except Exception as e:
        logger.error(f"Setup failed: {e}")
        raise click.ClickException(f"Setup error: {e}")


@main.command()
def demo():
    """Run a quick demonstration of Secure FL"""

    try:
        from .experiments.demo import run_demo

        rprint("[bold]🎬 Running Secure FL Demo[/bold]")
        rprint(
            "[dim]This will run a small federated learning experiment with 3 clients[/dim]\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Preparing demo...", total=None)

            success = run_demo()

            if success:
                progress.update(task, description="✓ Demo completed successfully!")
                rprint(
                    "\n[green]✓ Demo finished! Check ./demo_results/ for outputs[/green]"
                )
            else:
                progress.update(task, description="✗ Demo failed")
                rprint("\n[red]✗ Demo encountered errors[/red]")

    except ImportError as e:
        raise click.ClickException(f"Demo dependencies missing: {e}")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise click.ClickException(f"Demo error: {e}")


@main.command()
def info():
    """Display system information and component status"""

    from . import print_system_info

    print_system_info()


def server_command():
    """Entry point for secure-fl-server command"""
    main(["server"] + sys.argv[1:])


def client_command():
    """Entry point for secure-fl-client command"""
    main(["client"] + sys.argv[1:])


def experiment_command():
    """Entry point for secure-fl-experiment command"""
    main(["experiment"] + sys.argv[1:])


def setup_command():
    """Entry point for secure-fl-setup command"""
    main(["setup"] + sys.argv[1:])


if __name__ == "__main__":
    main()
