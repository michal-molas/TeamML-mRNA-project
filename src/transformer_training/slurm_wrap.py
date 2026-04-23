#!/usr/bin/env python3
import argparse
import subprocess
import sys
import os
import time

def is_job_running(job_id) -> bool:
    print(f"Waiting for job {job_id} to run", end='.', flush=True)

    while True:
        proc = subprocess.run(
            ['squeue'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        text = proc.stdout

        for line in text.splitlines()[1:]:
            words = line.split()
            current_id = words[0].strip()

            if len(current_id) >= len(job_id) and current_id[:len(job_id)] == job_id:
                if words[-1][0] != '(':
                    print()
                    print(line)
                    print(f"Job {job_id} running on {words[-1]}")
                    return current_id 

        print('.', end='', flush=True)
        time.sleep(1)

def run_sbatch(cmd):
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    job_id = proc.stdout.split()[-1]
    print(f"Job id: {job_id}")

    return job_id 

def tail_exec(job_id):
    file_name = f'err/error.{job_id}'
    timeout = 120
    start = time.time()

    while not os.path.exists(file_name):
        if time.time() - start > timeout:
            raise TimeoutError("File not created!")
        time.sleep(1)

    cmd = ['tail', '-f', file_name]
    os.execvp(cmd[0], cmd)

def run_command(cmd: list[str]) -> int:
    debug = False

    if cmd[0] in ['--debug', '-d']:
        debug = True
        cmd = cmd[1:]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    print("STDOUT:")
    print(proc.stdout)

    print("STDERR:")
    print(proc.stderr)

    return proc.returncode

def main():
    cmd = sys.argv[1:]
    if not cmd:
        print("usage: wrap.py [--debug | -d] <cmd> [args...]")
        sys.exit(1)

    job_id = run_sbatch(cmd)

    job_id = is_job_running(job_id)

    tail_exec(job_id)

if __name__ == "__main__":
    main()
