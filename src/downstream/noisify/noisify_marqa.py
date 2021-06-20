"""
Add noise to MSMARCOQA dataset instances.
Performed locally.
"""
import argparse
import codecs
import gzip
import json

from noise_funcs import rand_case, repeat

REPETITION_CHANCE = 0.2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-file')
    parser.add_argument('--out-file')
    parser.add_argument('--action', choices=['randcase', 'repeat'])
    args = parser.parse_args()

    noise_f = rand_case if args.action == 'randcase' else lambda x: repeat(x, REPETITION_CHANCE)

    with open(args.out_file, 'w', encoding='utf-8') as outf:
        new_data = {}
        with gzip.open(args.in_file, 'rb') as in_f:
            reader = codecs.getreader("utf-8")
            data = json.load(reader(in_f))

            new_data['query'] = {k: noise_f(v) for k, v in data['query'].items()}
            new_data['wellFormedAnswers'] = {k: v if v == '[]' else [noise_f(a) for a in v]
                                             for k, v in data['wellFormedAnswers'].items()}
            new_data['answers'] = {k: [noise_f(a) for a in v]
                                   for k, v in data['answers'].items()}
            new_data['query_type'] = data['query_type']
            psgs = {}
            for k, p in data['passages'].items():
                psgs[k] = [{'is_selected': p_i['is_selected'],
                            'url': p_i['url'],
                            'passage_text': noise_f(p_i['passage_text'])}
                           for p_i in p]
            new_data['passages'] = psgs

        json.dump(new_data, outf)


if __name__ == '__main__':
    main()
