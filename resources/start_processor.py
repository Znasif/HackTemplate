import json
import subprocess
import os

# The script assumes it's being run from the /app directory
LOGS_DIR = "/app/logs"
CONFIG_FILE = "/app/processor_config.json"

def main():
    """Reads config and starts all enabled processors in the background."""
    print("--- Starting background processors ---")
    
    # Ensure logs directory exists
    os.makedirs(LOGS_DIR, exist_ok=True)

    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)

    for proc_id, proc_config in config.items():
        if not proc_config.get('enabled', True):
            print(f"Skipping disabled processor: {proc_config['name']}")
            continue

        env_name = proc_config['conda_env']
        proc_name = proc_config['name']
        host = proc_config['host']
        port = proc_config['port']
        log_file = os.path.join(LOGS_DIR, f"{proc_name}.log")

        # Command to be run in the background
        cmd = f"""
        eval "$(/opt/conda/bin/conda shell.bash hook)" && \
        conda activate {env_name} && \
        uvicorn processors.{proc_name}:app --host {host} --port {port}
        """

        # Using Popen to run the command in the background (non-blocking)
        # and redirect stdout/stderr to a log file.
        print(f"Starting {proc_name} in '{env_name}' env on {host}:{port}. Logging to {log_file}")
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                ['/bin/bash', '-c', cmd],
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Start in a new session to detach from this script
            )
        print(f"-> Started {proc_name} with PID: {process.pid}")

if __name__ == "__main__":
    main()