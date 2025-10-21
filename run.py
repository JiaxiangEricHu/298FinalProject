import os
import subprocess

import vitis

cwd = os.getcwd() + "/"

# Initialize session
client = vitis.create_client()
client.set_workspace(path="./w")

# Delete the component if it already exists
if os.path.exists("./w/add"):
    client.delete_component(name="add")

# Create component. Create new config file in the component folder of the workspace
comp = client.create_hls_component(
    name="add",
    cfg_file=[cwd + "hls_config.cfg"],
    template="empty_hls_component",
)

# Run flow steps
comp = client.get_component(name="add")
comp.run(operation="C_SIMULATION")
comp.run(operation="SYNTHESIS")
# comp.run(operation="CO_SIMULATION")


# subprocess.run(["notepad.exe", "./w/add/add/reports/hls_compile.rpt"])
with open("./w/add/add/reports/hls_compile.rpt", "r") as f:
    c = f.read()
    print(c)
