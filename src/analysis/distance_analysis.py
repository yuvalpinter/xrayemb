"""
Print out euclidean distances between vectors produced by embedding table and by Tok
"""
import argparse
import logging
import os

import tensorflow as tf
import torch
from tqdm import tqdm

from src.downstream.dataset import get_dataset
from src.downstream.run_downstream import tensify_batch, UNK_LABEL
from src.tdt.infer import load_tdt_for_inference, call_bmod, AllMultiToksFilter, NoEasySuffsFilter, \
    StochasticFilter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['emoji', 'conll', 'nemer', 'ner',
                                              'marcoqa', 'marcosamp'])
    parser.add_argument('--dataloc', choices=['cto', 'ypinter'], default='ypinter')
    parser.add_argument('--base-model-dir')
    parser.add_argument('--tdt-model-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--vectorizer-type', choices=[None, 'lstm', 'conv', 'transformer'], default=None)
    parser.add_argument('--infer-policy', choices=['all-multi', 'no-easy-suffs', 'all'], default='no-easy-suffs',
                        help="Policy for selecting tokens to infer using TDT vectorizer")
    parser.add_argument('--pool-policy', choices=['max', 'avg', 'first'], default='max',
                        help="Policy for pooling multi-tokens")
    parser.add_argument('--hashtml', action='store_true')
    parser.add_argument('--lowercase', action='store_true')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--data-sample', type=float, default=1.0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model-type', choices=['bert', 'cbert', 'roberta', 'gpt2'], default='roberta')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    if args.model_type == 'roberta':
        tdt, tmp_dir, _ = load_tdt_for_inference(args)
        btok = tdt.btok
        bmod = tdt.bmodel
        tdt.tdtgen = None  # save us generation
    else:
        raise ValueError(f'Model type {args.model_type} not supported')

    ds = get_dataset(args)
    dstr = ds["train"]
    dsdv = ds["dev"]
    lab_list = list(set(dstr.labels + dsdv.labels)) + [UNK_LABEL]
    labs = {l: i for i, l in enumerate(lab_list)}
    dv_set = tensify_batch(dsdv, labs, btok, args, None, get_all_toks=True)

    if args.infer_policy == 'all-multi':
        infer_filter = AllMultiToksFilter(btok)
    elif args.infer_policy == 'no-easy-suffs':
        infer_filter = NoEasySuffsFilter(btok)
    elif args.infer_policy == 'all':
        infer_filter = StochasticFilter(btok, 1.0)
    else:
        raise ValueError('No infer policy specified')

    with torch.no_grad():
        dst_report = ''
        for i, (batch, targets, sents) in enumerate(tqdm(dv_set, mininterval=120, desc='Dev run')):
            b_m_out, b_inps = call_bmod(bmod, batch)
            tdt_m_out, _, _, tdt_embs, tdt_inps = tdt(batch, mask=False, get_inputs=True, get_embs=True)
            tdt_reps = tdt_m_out[2]
            base_reps = b_m_out[1]
            _, seqlen, dim = tdt_reps[0].shape
            dt = tdt_reps[0].dtype

            infer_replacement_map = infer_filter(batch).to(args.device)

            dst_report += f'=============\nSentence {i}: {sents[0]}\n'

            zero_diff = (tdt_embs-tdt_reps[0]).norm()
            last_diff = (tdt_embs-tdt_reps[-1]).norm()

            dst_report += f'base inputs shape: {base_reps[0].shape}, ' \
                          f'embeddings diff with layer 0: {zero_diff:.6f}, ' \
                          f'embedding diff with last layer: {last_diff:.6f}\n'

            for lyr, (tdt_layer, base_layer) in enumerate(zip(tdt_reps, base_reps)):
                tdt_l_reps = tdt_layer[0]    # shape: seq_len * emb_size
                base_l_reps = base_layer[0]  # shape: seq_len * emb_size
                t_i = 0
                b_i = 0
                diffs = []
                vec_ts = []

                while t_i < seqlen:
                    if tdt_inps[0, t_i] == btok.pad_token_id:
                        break
                    v_ti = tdt_l_reps[t_i]
                    rep_i = infer_replacement_map[0, b_i]
                    if rep_i > -1:
                        vec_ts.append(f'{t_i}')
                        rep_map = rep_i
                        b_vecs = []
                        while rep_i == rep_map:
                            v_bi = base_l_reps[b_i]
                            b_vecs.append(v_bi)
                            b_i += 1
                            rep_i = infer_replacement_map[0, b_i]
                        v_bi = torch.stack(b_vecs).max(dim=0).values
                    else:
                        v_bi = base_l_reps[b_i]
                        b_i += 1
                    diff_i = (v_ti - v_bi).norm()
                    diffs.append(f'{diff_i.item():.3f}')
                    t_i += 1

                if lyr == 0:
                    dst_report += 'Vectorized tokens: ' + ' '.join(vec_ts) + '\n'

                dst_report += ' '.join(diffs) + '\n'

        out_path = os.path.join(tmp_dir, 'dists.txt')
        with open(out_path, "w", encoding="utf-8") as writer:
            writer.write(dst_report)
        tf.io.gfile.copy(out_path, os.path.join(args.output_dir, 'dists.txt'))
        logger.info(f'Wrote report to {args.output_dir}dists.txt.')


if __name__ == "__main__":
    main()
