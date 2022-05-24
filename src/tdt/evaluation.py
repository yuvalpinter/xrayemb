import logging
import os
from typing import List, Dict

import torch
from torch.nn import Module
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from src.tdt.data import load_dataset
from src.tdt.tokdetok import TdtWrapper

logger = logging.getLogger(__name__)


def evaluate(args, model_wrapper: TdtWrapper, pretok,
             distloss: Module = torch.nn.MSELoss(),
             genloss: Module = torch.nn.CrossEntropyLoss(),
             prefix="") -> Dict:
    """
    Evaluate an LM on a held-out dev/test set.
    :param args: arguments passed by user to training loop
    :param model_wrapper: TDT main object
    :param prefix: reporting option
    :param distloss: distance loss function for vectors ("Tok")
    :param genloss: generation loss function for sequences ("Detok")
    :return: dict containing reporting evaluation measures
    """

    eval_output_dir = args.output_dir

    btok = model_wrapper.btok
    eval_dataset = load_dataset(args.eval_data_file, pretok, btok,
                                args.line_by_line, args.block_size, portion=1.0, shuffle=False)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    def collate(examples: List[torch.Tensor]):
        if btok.pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=btok.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model_wrapper = torch.nn.DataParallel(model_wrapper)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_lm_loss = 0.0
    eval_vc_loss = 0.0
    eval_gn_loss = 0.0
    eval_oa_loss = 0.0
    nb_eval_steps = 0
    model_wrapper.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating", mininterval=60):
        with torch.no_grad():
            lm_out, vec_lrn_batch, generated_batch = model_wrapper(batch, mask=True, generate_all=args.generate_all)
            lm_loss = lm_out[0].mean().item() if lm_out else 0.0
            vec_lss = distloss(*(vec_lrn_batch[:2])).mean().item() if vec_lrn_batch else 0.0
            gen_lss = genloss(*generated_batch).mean().item() if generated_batch else 0.0

            eval_lm_loss += lm_loss
            eval_vc_loss += vec_lss
            eval_gn_loss += gen_lss
            eval_oa_loss += lm_loss + (args.alpha_vec * vec_lss) + (args.alpha_gen * gen_lss)
        nb_eval_steps += 1

    eval_lm_loss = eval_lm_loss / nb_eval_steps
    eval_vc_loss = eval_vc_loss / nb_eval_steps
    eval_gn_loss = eval_gn_loss / nb_eval_steps
    eval_oa_loss = eval_oa_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_lm_loss)).item()

    result = {"perplexity": perplexity,
              "vectorizer_loss": eval_vc_loss,
              "generator_loss": eval_gn_loss,
              "overall_loss": eval_oa_loss}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result
