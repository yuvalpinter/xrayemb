"""
Analyze results of an emoji prediction output based on sequence length (in subword tokens)
Performed locally.
"""
import argparse

import pandas as pd

file_names = ['none', 'second_base', 'scaffolding', 'stoch', 'nosuf']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dir', help=f'a directory name, including emoji output files named {file_names}.tsv.')
    args = parser.parse_args()

    head = ['raw', 'pieces', 'gold', 'pred']
    df = pd.read_csv(f'{args.in_dir}/none.tsv', sep='\t', quoting=3, names=head)
    df['plen'] = df['pieces'].str.split().apply(len)  # .value_counts()
    df['none'] = df.pred
    minlen = df["plen"].min()
    maxlen = df["plen"].max()
    print(f'Min len: {minlen}, max len: {maxlen}')

    corr_dfs = {}
    for fname in file_names[1:]:
        tmp_df = pd.read_csv(f'{args.in_dir}/{fname}.tsv', sep='\t', quoting=3, names=head)
        df[fname] = tmp_df.pred
    for fname in file_names:
        corr_dfs[fname] = df[df['gold'] == df[fname]]

    names_header = "\t".join(file_names)
    print(f'l\ttotal\t{names_header}')
    for l in list(range(minlen, min(maxlen, 61))) + [maxlen+1]:
        tot_l = len(df[df.plen <= l])
        corrs_l = [len(corr_dfs[k][corr_dfs[k].plen <= l]) for k in file_names]
        acc_strs = "\t".join([f'{c / tot_l:.4f}' for c in corrs_l])
        print(f'{l}\t{tot_l}\t{acc_strs}')


if __name__ == '__main__':
    main()
