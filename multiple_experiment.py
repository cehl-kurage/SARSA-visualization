import subprocess

num_experiments = 10

for _ in range(num_experiments):
    subprocess.run("python experiment.py")
