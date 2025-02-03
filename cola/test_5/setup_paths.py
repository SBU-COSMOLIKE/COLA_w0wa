"""
    Script that adjusts all hardcoded paths to your current system.
"""
import os

print("Setting up paths...")

hardcoded_path = "/gpfs/projects/MirandaGroup/victoria/cola_projects/test_5"
new_path = os.getcwd()

lua_folders = [f"lua_files_{i}" for i in ["a", "b", "no_pairfix", "no_pairfix_projected"]]
for folder in lua_folders:
    for file in os.listdir(folder):
        with open(f"{folder}/{file}", "r") as f: contents = f.read()
        new_contents = contents.replace(hardcoded_path, new_path)
        with open(f"{folder}/{file}", "w") as f: contents = f.write(new_contents)

transfer_folders = ["transfers", "transfers_projected"]
for folder in transfer_folders:
    ls = os.listdir(folder)
    for subfolder in ls:
        with open(f"{folder}/{subfolder}/transferinfo.dat", "r") as f: contents = f.read()
        new_contents = contents.replace("/gpfs/projects/MirandaGroup/victoria/cola_projects/test/w0waCDM/transfer_functions", f"{new_path}/{folder}")
        with open(f"{folder}/{subfolder}/transferinfo.dat", "w") as f: contents = f.write(new_contents)


print("Finished!")