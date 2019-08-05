from . import os, argparse, arguments
import subprocess

def handle_restore_dir(args: argparse.Namespace):
    if args.restore_dir is not None:
        command = ["mkdir", "-p", args.model_dir]
        print("Running bash command: \"{}\"".format(" ".join(command)))
        subprocess.run(command)
        command = ["cp", "-r", "--reflink=auto", os.path.join(args.restore_dir,"."), args.model_dir]
        print("Running bash command: \"{}\"".format(" ".join(command)))
        subprocess.run(command)
        input("Bash command finished. Please check if there is no error and press any key to continue")

