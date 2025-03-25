from invoke import task, run
import subprocess

@task
def bootstrap_env(ctx):
    """Sets up the virtual environment and installs dependencies."""
    run("python -m venv .venv", echo=True)
    run("source .venv/bin/activate && ./scripts/install_requirements.sh", shell=True, echo=True)
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
    """Installs transformers and configures Hugging Face CLI login."""
    run("pip install transformers huggingface_hub", echo=True)
    run("git config --global credential.helper store", echo=True)
    run("huggingface-cli login", echo=True)
    run("git submodule init", echo=True)
    run("git submodule update --remote", echo=True)

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
def train_engine(ctx, batch_size=64, max_iters=5000, max_epochs=20, eval_iters=100, eval_interval=100, grad_norm_clip_value=1.0):
    """Trains the engine with specified parameters."""
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
    run(" ".join(command), echo=True)