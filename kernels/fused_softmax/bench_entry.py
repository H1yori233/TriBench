import sys


def main():
    from tribench.cli import main as tribench_main
    sys.argv = ["tribench", "run", "--kernel", "fused_softmax"] + sys.argv[1:]
    tribench_main()


if __name__ == "__main__":
    main()
