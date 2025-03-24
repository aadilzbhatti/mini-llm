import subprocess
import signal
import os
import time

VM_HOSTNAME = "gpu-dev"
CHECKPOINT_PARENT_DIR = "mini-llm/models/wiki-llm/checkpoints"
LOCAL_PORT = 6006

tensorboard_process = None
ssh_process = None

def get_latest_checkpoint_dir_remote():
    """Finds the latest hashed checkpoint directory on the remote VM."""
    remote_command = f"ssh {VM_HOSTNAME} 'ls -dt {CHECKPOINT_PARENT_DIR}/* | head -n 1'"
    try:
        result = subprocess.run(remote_command, shell=True, capture_output=True, text=True, check=True)
        checkpoint_dir = result.stdout.strip()
        return checkpoint_dir
        # print(checkpoint_dir)
        # if os.path.isdir(checkpoint_dir):
        #     return checkpoint_dir
        # else:
        #     return None
    except subprocess.CalledProcessError as e:
        print(f"Error finding remote checkpoint directory: {e}")
        return None

def start_tensorboard():
    global tensorboard_process
    checkpoint_dir = get_latest_checkpoint_dir_remote()
    if not checkpoint_dir:
        print("No checkpoint directory found on remote VM.")
        return

    tensorboard_logdir = checkpoint_dir
    tensorboard_command = f"ssh {VM_HOSTNAME} 'tensorboard --logdir={tensorboard_logdir} --port=6006'"
    tensorboard_process = subprocess.Popen(tensorboard_command, shell=True)
    print(f"TensorBoard started on VM, using logdir: {tensorboard_logdir}")

def start_ssh_tunnel():
    global ssh_process
    ssh_command = f"ssh -f -N -L {LOCAL_PORT}:localhost:6006 {VM_HOSTNAME}"
    ssh_process = subprocess.Popen(ssh_command, shell=True)
    print(f"SSH tunnel established. Forwarding to localhost:{LOCAL_PORT}")
    time.sleep(2)

def stop_processes(signal, frame):
    global tensorboard_process, ssh_process
    print("\nStopping processes...")

    try:
        local_kill_ssh_command = f"pkill -f 'ssh -f -N -L {LOCAL_PORT}:localhost:6006 {VM_HOSTNAME}'"
        subprocess.run(local_kill_ssh_command, shell=True, check=False)
        print("Local SSH tunnel process terminated (if running).")
    except Exception as e:
        print(f"Error terminating local SSH process: {e}")

    try:
        remote_kill_command = f"ssh {VM_HOSTNAME} 'pkill tensorboard'"
        subprocess.run(remote_kill_command, shell=True, check=False)
        print("Remote TensorBoard process terminated (if running).")
    except Exception as e:
        print(f"Error terminating remote TensorBoard: {e}")

    if tensorboard_process:
        try:
            tensorboard_process.terminate()
            tensorboard_process.wait(timeout=5)
            print("Local TensorBoard process stopped.")
        except subprocess.TimeoutExpired:
            print("Local tensorboard process did not terminate gracefully. Killing it.")
            tensorboard_process.kill()
            tensorboard_process.wait()
            print("Local tensorboard process killed.")
        except Exception as e:
            print(f"Error terminating Local Tensorboard process: {e}")

    exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, stop_processes)

    try:
        start_tensorboard()
        start_ssh_tunnel()

        print("Processes running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)

    except Exception as e:
        print(f"An error occurred: {e}")
        stop_processes(None, None)