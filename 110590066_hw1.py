import os
import shutil


# clean up results dir
def clean_results_dir():
    cwd = os.getcwd()
    results_dir = os.path.join(cwd, "results")
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)


if __name__ == "__main__":
    clean_results_dir()
