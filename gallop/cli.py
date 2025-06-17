# gallop/cli.py
import subprocess
import sys
from importlib.resources import files

def main():
    """Launches the Streamlit app from within the package."""
    try:
        # This is the key change:
        # It gets a handle to the installed 'gallop' package...
        # ...and finds the 'gallop_streamlit.py' file inside it.
        script_path = files('gallop').joinpath('gallop_streamlit.py')

    except (ModuleNotFoundError, FileNotFoundError):
        print("Error: Could not find the gallop_streamlit.py script inside the gallop package.", file=sys.stderr)
        print("Please ensure 'gallop' is installed correctly and the file was included.", file=sys.stderr)
        sys.exit(1)

    command = [sys.executable, "-m", "streamlit", "run", str(script_path)]
    print(f"Starting Gallop UI... (running: {' '.join(command)})")

    try:
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print("Error: 'streamlit' command not found.", file=sys.stderr)
        print("Please make sure Streamlit is installed in your environment (`pip install streamlit`).", file=sys.stderr)
        sys.exit(1)