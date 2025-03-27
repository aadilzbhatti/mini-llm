import datetime
from invoke import task, run
import subprocess
import sys

@task
def bootstrap_env(ctx):
    """Sets up the virtual environment and installs dependencies using Poetry."""
    required_version = (3, 13)
    if sys.version_info < required_version:
        print(f"Python {required_version[0]}.{required_version[1]} or higher is required.")
        print("Attempting to install the required Python version...")

        # Add deadsnakes PPA and install Python 3.13
        run("sudo add-apt-repository -y ppa:deadsnakes/ppa", echo=True)
        run("sudo apt update", echo=True)
        run("sudo apt install -y python3.13", echo=True)

        # Update alternatives to use Python 3.13
        run("sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1", echo=True)
        print("Python 3.13 installed. Please restart your terminal and try again.")
        return

    # Ensure Poetry is installed without the --user flag
    run("pip install poetry", echo=True)

    # Use Poetry to create the virtual environment and install dependencies
    run("poetry env use python", echo=True)
    run("poetry install", echo=True)

    # Initialize and update git submodules
    run("git submodule init", echo=True)
    run("git submodule update", echo=True)

@task
def pull_models(ctx):
    """Pulls models from Hugging Face using git-lfs."""
    with ctx.cd("models/wiki-llm/"):
        run("yes | sudo apt-get install git-lfs", echo=True)
        run("git-lfs install", echo=True)
        run("git-lfs pull", echo=True)

@task
def install_requirements(ctx):
    """Installs requirements from requirements.txt and PyTorch nightly."""
    with open("requirements.txt", "r") as f:
        packages = [line.strip().split("#")[0].strip() for line in f if line.strip() and not line.startswith("#")]

    for package in packages:
        try:
            subprocess.check_call(["pip", "install", "--no-dependencies", "--ignore-installed", package])
        except subprocess.CalledProcessError:
            print(f"Failed to install {package}, continuing...")

    run("pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128", echo=True)

@task
def setup_hf(ctx):
    """Configures Hugging Face CLI login."""
    run("pip install huggingface_hub", echo=True)
    run("git config --global credential.helper store", echo=True)
    run("huggingface-cli login", echo=True)
    run("git submodule init", echo=True)
    run("git submodule update --remote", echo=True)
    run("sudo apt-get install -y git-lfs", echo=True)
    run("git lfs install", echo=True)

@task
def run_tensorboard(ctx):
    """
    Runs tensorboard in the background. NOTE: Only run this on your local
    machine; this script assumes you have a SSH target named "gpu-dev" and will
    set up a background SSH process based on that.
    """
    run("python scripts/run_tensorboard.py", echo=True)

def process_infinity(value_as_string):
    """Converts 'inf' to float('inf'), handles other numbers."""
    if isinstance(value_as_string, str):
        if value_as_string.lower() == 'inf':
            return float('inf')
        else:
            try:
                return float(value_as_string)
            except ValueError:
                return None  # Or raise an exception
    else:
        return value_as_string

@task
def train_engine(ctx, batch_size=64, max_iters=5000, max_epochs=20, eval_iters=100, eval_interval=100, grad_norm_clip_value=1.0, background=False):
    """Trains the engine with specified parameters, optionally in the background with timestamped logging."""
    grad_norm_clip_value = process_infinity(grad_norm_clip_value)

    command = [
        "python",
        "engine.py",
        "--mode", "train",
        "--save_checkpoints",
        "--verbose",
        "--enable_tqdm",
        f"--batch_size={batch_size}",
        f"--max_iters={max_iters}",
        f"--max_epochs={max_epochs}",
        f"--eval_iters={eval_iters}",
        f"--eval_interval={eval_interval}",
        f"--grad_norm_clip_value={grad_norm_clip_value}"
    ]

    command_str = " ".join(command)

    if background:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"train_engine_{timestamp}.log"
        pid_file = f"train_engine_{timestamp}.pid"
        background_command = f"nohup {command_str} > {log_file} 2>&1 & echo $! > {pid_file}"
        run(background_command, echo=True)
        with open(pid_file, "r") as file:
            pid = file.read().strip()
        print(f"Training engine started in the background. Output logged to: {log_file}. PID: {pid}")
        run(f"rm {pid_file}", echo=True)
    else:
        run(command_str, echo=True)