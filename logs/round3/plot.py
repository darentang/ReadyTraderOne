import os
import subprocess

def send_command(command, timer=None):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    print(output)
    p_status = p.wait()


plot_python = "~/projects/readytraderone/liveplot.py"
cwd = os.getcwd()

for file in os.listdir():
    if "match" in file and "csv" in file:
        file_dir = cwd + "/" + file
        out_dir = cwd + "/img/" + file[:-4] + ".png"
        command = f"python3 {plot_python} {file_dir} {out_dir}"
        send_command(command)