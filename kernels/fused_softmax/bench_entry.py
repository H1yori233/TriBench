import sys


def main():
    from gpubench.cli import main as gpubench_main
    sys.argv = ["gpubench", "run", "--kernel", "fused_softmax"] + sys.argv[1:]
    gpubench_main()


if __name__ == "__main__":
    main()
