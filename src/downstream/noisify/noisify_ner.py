"""
Add noise to NER dataset instances.
Performed locally.
"""
import argparse

from noise_funcs import rand_case, repeat

REPETITION_CHANCE = 0.2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file')
    parser.add_argument('--out-file')
    parser.add_argument('--action', choices=['randcase', 'repeat'])
    args = parser.parse_args()

    noise_f = rand_case if args.action == 'randcase' else lambda x: repeat(x, REPETITION_CHANCE)

    with open(args.out_file, 'w') as outf:
        with open(args.in_file) as inf:
            for line in inf:
                if '\t' not in line:
                    outf.write(line)
                else:
                    w, t = line.split('\t')
                    outf.write(f'{noise_f(w)}\t{t}')


if __name__ == '__main__':
    main()
