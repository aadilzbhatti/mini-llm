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

# scripts/install_requirements.sh converted into invoke task:
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

# scripts/setup_hf.sh converted into invoke task:
@task
def setup_hf_script(ctx):
    """Installs transformers and configures Hugging Face CLI login."""
    run("pip install transformers huggingface_hub", echo=True)
    run("git config --global credential.helper store", echo=True)
    run("huggingface-cli login", echo=True)
    run("git submodule init", echo=True)
    run("git submodule update --remote", echo=True)
