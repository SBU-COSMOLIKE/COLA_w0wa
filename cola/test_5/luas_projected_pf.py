import os
for phase in ["a", "b"]:
    os.chdir(f"./lua_files_projected_{phase}")
    luas = os.listdir()
    for lua in luas:
        with open(lua, "r") as f: contents = f.read()
        contents = contents.replace("ic_fix_amplitude = false", "ic_fix_amplitude = true")
        if phase == "b":
            contents = contents.replace("ic_reverse_phases = false", "ic_reverse_phases = true")
        contents = contents.replace("output/no_pairfix_projected", f"output/projected_{phase}")
        with open(lua, "w") as f: f.write(contents)
    os.chdir("..")

