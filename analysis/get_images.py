import subprocess

def send_command(command, timer=None):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (output, err) = p.communicate()
    print(output)
    p_status = p.wait()

instances = {
    193: "D",
    297: "D",
    387: "D",
    770: "D",
    1659: "D",
    2027: "D",
    2117: "D",
    253: "U",
    338: "U",
    602: "U",
    1535: "U",
    1820: "U",
    196: "U",
}

window = 30
md_filename = "information.md"
content = "# High concavity analysis \n"
surgery_location = "~/projects/readytraderone/surgery.py"
match_data = "~/projects/readytraderone/logs/round1/match31_events.csv"


for time, concavity in instances.items():
    print(concavity, time)
    start_time = int(time - window/2)
    out_filename = f"{int(time)}_{concavity}.png"
    print(f"Generating at time {start_time} saving to {out_filename}")
    command = f"python3 {surgery_location} {match_data} {out_filename} {start_time} {window}"
    # send_command(command)
    content += f"\n## Concave {concavity} at {int(time)}\n"
    content += f"\n![{int(time)}]({out_filename})\n"


with open(md_filename, "w") as f:
    f.write(content)
