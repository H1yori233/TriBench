import sys


def main():
    from tribench.cli import main as tribench_main
    import sys
    sys.argv = ["tribench", "run", "--kernel", "rope"] + sys.argv[1:]
    tribench_main()


if __name__ == "__main__":
    main()
