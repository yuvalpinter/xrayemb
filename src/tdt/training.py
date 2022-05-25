"""
A training loop for TDT
"""
import logging
import os
import random
import time
from typing import Tuple, List

import torch
from torch.nn import Module, DataParallel
from torch.nn.utils.rnn import pad_sequence
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import RandomSampler, DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange, tqdm
from transformers import get_linear_schedule_with_warmup

from src.tdt.consts import SPECIAL_CHAR_LIST
from src.tdt.cycles import TdCycleTrainer, DtCycleTrainer
from src.tdt.evaluation import evaluate
from src.tdt.gen import gen_rand, cycle_check, gen_from_masks
from src.tdt.tokdetok import TdtWrapper
from src.tdt.utils import PreTokenizer, rotate_checkpoints, set_seed

logger = logging.getLogger(__name__)


def train(args, train_dataset, tdt_wrapper: TdtWrapper,
          td_cycle: TdCycleTrainer, dt_cycle: DtCycleTrainer,
          distloss: Module = torch.nn.MSELoss(),
          genloss: Module = torch.nn.CrossEntropyLoss(),
          char_vocab=None) \
        -> Tuple[int, float]:
    """
    Train the model
    :param args: user-supplies parameters
    :param train_dataset: pre-ingested training dataset
    :param tdt_wrapper: main model object
    :param td_cycle: trainer for T->D cycles
    :param dt_cycle: trainer for D->T cycles
    :param distloss: distance loss function for vectorizer
    :param genloss: generation loss function for detokenizer
    :param char_vocab: all characters available for TDT modules
    :return: number of steps performed, mean training loss
    """
    if args.local_rank in [-1, 0]:
        os.makedirs(args.logdir)
        tb_writer = SummaryWriter(log_dir=args.logdir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    pretok = PreTokenizer(hashtml=args.hashtml)
    
    btok = tdt_wrapper.btok

    clean_voc = char_vocab[:-len(SPECIAL_CHAR_LIST)]  # for eyeballing generation sequences

    def collate(examples: List[torch.Tensor]):
        if btok.pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=btok.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = (len(train_dataloader) // args.gradient_accumulation_steps) * args.num_train_epochs
    t_cyc_total = t_total * args.cycle_batch_iters // args.cycle_freq
    if args.warmup_steps > 1.0:
        raise AttributeError('task-warmup parameter is proportion of total steps')
    t_warm = int(args.warmup_steps * t_total)
    t_cyc_warm = int(args.warmup_steps * t_cyc_total)

    all_named_params = tdt_wrapper.named_parameters()

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in all_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in all_named_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=t_warm, num_training_steps=t_total
    )

    # Initialize cycle training
    if td_cycle is not None or dt_cycle is not None:
        cycle_optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        cycle_scheduler = get_linear_schedule_with_warmup(
            cycle_optimizer, num_warmup_steps=t_cyc_warm, num_training_steps=t_cyc_total
        )
    else:
        cycle_optimizer = None
        cycle_scheduler = None

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        tdt_wrapper, optimizer = amp.initialize(tdt_wrapper, optimizer, opt_level=args.fp16_opt_level)

    # Multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        tdt_wrapper = DataParallel(tdt_wrapper)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        # noinspection PyArgumentList
        tdt_wrapper = DistributedDataParallel(
            tdt_wrapper, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    # noinspection PyUnresolvedReferences
    global_batch_size = args.train_batch_size * args.gradient_accumulation_steps \
                        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        global_batch_size
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_lm_loss, tr_ds_loss, tr_gn_loss, tr_oa_loss = 0.0, 0.0, 0.0, 0.0
    logging_lm_loss, logging_ds_loss, logging_gn_loss, logging_oa_loss = 0.0, 0.0, 0.0, 0.0

    tdt_wrapper.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility

    train_time = 0.0

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader,
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0],
                              mininterval=60)  # only update every minute
        t0 = time.time()

        for step, batch in enumerate(epoch_iterator):
            tdt_wrapper.train()

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            def forward_backward_pass():
                lm_out, vec_lrn_batch, generated_batch = tdt_wrapper(batch, mask=True, generate_all=args.generate_all)
                if lm_out is None:
                    return torch.tensor([0.0]).to(args.device), torch.tensor([0.0]).to(args.device)

                lm_lss = lm_out[0].to(args.device)  # model outputs are always tuple in transformers (see doc)
                if vec_lrn_batch is not None:
                    if len(vec_lrn_batch) > 2:
                        vec_lrn_batch = vec_lrn_batch[:2]

                    vec_lss = distloss(*vec_lrn_batch).to(args.device)
                else:
                    vec_lss = torch.tensor(0.0, requires_grad=True).to(args.device)
                if generated_batch is not None:
                    gen_lss = genloss(*generated_batch).to(args.device)
                else:
                    gen_lss = torch.tensor(0.0, requires_grad=True).to(args.device)

                if args.n_gpu > 1:
                    # mean() to average on multi-gpu parallel training
                    lm_lss = lm_lss.mean()
                    vec_lss = vec_lss.mean()
                    gen_lss = gen_lss.mean()
                if args.gradient_accumulation_steps > 1:
                    lm_lss = lm_lss / args.gradient_accumulation_steps
                    vec_lss = vec_lss / args.gradient_accumulation_steps
                    gen_lss = gen_lss / args.gradient_accumulation_steps

                overall_lss = lm_lss + \
                              args.alpha_vec * vec_lss + \
                              args.alpha_gen * gen_lss

                if args.fp16:
                    with amp.scale_loss(overall_lss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    overall_lss.backward()

                return lm_lss.item(), vec_lss.item(), gen_lss.item(), overall_lss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                lm_loss, ds_loss, gn_loss, oa_loss = forward_backward_pass()
            else:
                # When doing gradient accumulation, syncing at non-optimization steps should be avoided
                # to decrease inter-gpu communication time.
                # https://github.com/pytorch/pytorch/pull/19577
                with tdt_wrapper.no_sync():
                    lm_loss, ds_loss, gn_loss, oa_loss = forward_backward_pass()

            tr_lm_loss += lm_loss
            tr_ds_loss += ds_loss
            tr_gn_loss += gn_loss
            tr_oa_loss += oa_loss
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(tdt_wrapper.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                tdt_wrapper.zero_grad()
                global_step += 1

                train_time += time.time() - t0
                throughput = (global_batch_size * global_step) / train_time
                if step % args.logging_steps == 0:
                    logger.info(f"Throughput {throughput:.3f}")
                t0 = time.time()

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # report generator state
                    if args.report:
                        logger.info(f'generator samples after {global_step} steps:')
                        with torch.no_grad():
                            if args.local_rank > -1:
                                actual_tdt = next(tdt_wrapper.children())
                                assert isinstance(actual_tdt, TdtWrapper), f"Found wrapper of type {type(actual_tdt)}"
                            else:
                                actual_tdt = tdt_wrapper

                            rand_gen = '\t'.join(gen_rand(actual_tdt, char_vocab, args))
                            mask_gen = '\t'.join(gen_from_masks(actual_tdt, char_vocab))
                            randseq = ''.join([random.choice(clean_voc) for _ in range(15)])
                            cyclic, manual = cycle_check(actual_tdt, char_vocab, seq=randseq)

                            logger.info('from random:\t' + rand_gen)
                            logger.info('from masks :\t' + mask_gen)
                            logger.info(f'from {randseq}:\tcyclic: {cyclic}\tmanual: {manual}')

                            tb_writer.add_text('random', rand_gen, global_step)
                            tb_writer.add_text('masks', mask_gen, global_step)
                            tb_writer.add_text('random sequence', randseq, global_step)
                            tb_writer.add_text('cyclic', cyclic, global_step)
                            tb_writer.add_text('manual', manual, global_step)

                    # Log metrics
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, tdt_wrapper, pretok)
                        for key, value in results.items():
                            # noinspection PyUnboundLocalVariable
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    lr_val = scheduler.get_last_lr()[0]
                    av_lm_logged_loss = (tr_lm_loss - logging_lm_loss) / args.logging_steps
                    av_ds_logged_loss = (tr_ds_loss - logging_ds_loss) / args.logging_steps
                    av_gn_logged_loss = (tr_gn_loss - logging_gn_loss) / args.logging_steps
                    av_oa_logged_loss = (tr_oa_loss - logging_oa_loss) / args.logging_steps
                    tb_writer.add_scalar(f"lr", lr_val, global_step)
                    tb_writer.add_scalar(f"lm_loss", av_lm_logged_loss, global_step)
                    tb_writer.add_scalar(f"vectorizer_loss", av_ds_logged_loss, global_step)
                    tb_writer.add_scalar(f"generator_loss", av_gn_logged_loss, global_step)
                    tb_writer.add_scalar(f"overall_loss", av_oa_logged_loss, global_step)
                    logger.info(f'lm loss = {av_lm_logged_loss:.3f}, '
                                f'ds loss (x1000) = {av_ds_logged_loss * 1000:.3f}, '
                                f'gn loss = {av_gn_logged_loss:.3f}, '
                                f'lr = {lr_val:.3e}')
                    logging_lm_loss = tr_lm_loss
                    logging_ds_loss = tr_ds_loss
                    logging_gn_loss = tr_gn_loss
                    logging_oa_loss = tr_oa_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, f"{checkpoint_prefix}-{global_step}")
                    os.makedirs(output_dir, exist_ok=True)

                    # NOT tdt_wrapper.save(output_dir), which fails for distributed
                    torch.save(tdt_wrapper.state_dict(), os.path.join(output_dir, "model.checkpoint"))

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.train_cycle_dep and (step + 1) % args.cycle_freq == 0:
                tdt_wrapper.train()
                if args.local_rank == -1:
                    tdloss = td_cycle(args, tdt_wrapper, genloss, cycle_optimizer, cycle_scheduler)
                    dtloss = dt_cycle(args, tdt_wrapper, distloss, cycle_optimizer, cycle_scheduler)
                    logger.info(f'Completed cycle loops with {tdloss:.3f} T-D loss and {dtloss:.3f} D-T loss.')
                else:
                    with tdt_wrapper.no_sync():
                        tdloss = td_cycle(args, tdt_wrapper, genloss, cycle_optimizer, cycle_scheduler)
                        dtloss = dt_cycle(args, tdt_wrapper, distloss, cycle_optimizer, cycle_scheduler)
                        logger.info(f'Completed no_sync cycle loops with {tdloss:.3f} T-D loss and {dtloss:.3f} D-T loss.')

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_lm_loss / global_step
