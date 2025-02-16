import os
import nbformat
import re
import argparse
from nbconvert.preprocessors import ExecutePreprocessor

parser = argparse.ArgumentParser()
parser.add_argument("--pattern", type=str, default=".*")
args = parser.parse_args()

# Define the folder containing notebooks
folder_path = "../notebooks"
priority_notebook = "Macro_Assets.ipynb"  # Change this to the notebook that should run first

# Get the list of all notebooks, ensuring the priority notebook runs first
notebooks = [f for f in os.listdir(folder_path) if f.endswith(".ipynb") and re.match(args.pattern, f)]
if priority_notebook in notebooks:
    notebooks.remove(priority_notebook)  # Remove the priority notebook from the list
    notebooks.insert(0, priority_notebook)  # Insert it at the beginning

# Execute each notebook
for notebook in notebooks:
    notebook_path = os.path.join(folder_path, notebook)
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    
    try:
        ep.preprocess(nb, {"metadata": {"path": folder_path}})
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        print(f"Successfully executed: {notebook}")
    except Exception as e:
        print(f"Error executing {notebook}: {e}")
