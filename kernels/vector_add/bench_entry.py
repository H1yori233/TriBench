import sys


def main():
    from gpubench.cli import main as gpubench_main
    import sys
    sys.argv = ["gpubench", "run", "--kernel", "vector_add"] + sys.argv[1:]
    gpubench_main()


if __name__ == "__main__":
    main()
