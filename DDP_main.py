import os
import sys
import random
import numpy as np
import argparse
import torch
from dataloader import DiffusionLoader
from transformers import AutoTokenizer, AutoConfig, BartForConditionalGeneration
import diffusion_token_freq
from torch.optim import AdamW
from torch.nn.utils.rnn import pad_sequence
import fastNLP
from tqdm import tqdm
from sample import Categorical, WholeWordMasking
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import datetime

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name_or_path", default='/NAS/luoyc/wuyux/data/MolGen-large', type=str, required=False)
    parser.add_argument("--task_name", default='lm1b', type=str, required=False)
    parser.add_argument("--lr", default=5e-4, type=float, required=False)
    parser.add_argument("--epochs", default=3, type=int, required=False)
    parser.add_argument("--batch_size", default=64, type=int, required=False)
    parser.add_argument("--token_freq_lambda", default=0.0, type=float, required=False)
    parser.add_argument("--num_steps", default=2048, type=int, required=False)
    parser.add_argument("--eval_step_size", default=4, type=int, required=False)
    parser.add_argument("--dev_size", default=5e-4, type=float, required=False)
    parser.add_argument("--hybrid_lambda", default=1e-2, type=float, required=False)
    parser.add_argument("--eval_steps", default=500, type=int, required=False)
    parser.add_argument("--seed", default=42, type=int, required=False)
    # parser.add_argument("--device", default='cuda:0', type=str, required=False)
    parser.add_argument("--logging_steps", default=100, type=int, required=False)
    parser.add_argument('--predict_x0', default=True, type=bool, required=False)
    parser.add_argument("--load_step", default=-1, type=int, required=False)
    parser.add_argument("--sample_strategy", default='Categorical', type=str, required=False)
    parser.add_argument("--schedule", default='mutual', type=str, required=False)
    parser.add_argument("--from_scratch", default=False, type=bool, required=False)
    parser.add_argument("--timestep", default='none', type=str, required=False)
    # parser.add_argument("--local_rank", default=-1)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    local_rank = int(os.environ['LOCAL_RANK'])
    device = torch.device("cuda", local_rank)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=9600))

    set_seed(args)
    save_path = f'./model_name_MolGen-large-random_lr_{args.lr}_seed_{args.seed}_numsteps_{args.num_steps}_sample_{args.sample_strategy}_schedule_{args.schedule}_hybridlambda_{args.hybrid_lambda}_wordfreqlambda_{args.word_freq_lambda}_fromscratch_{args.from_scratch}_timestep_{args.timestep}_ckpts'
    tokenizer = AutoTokenizer.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large')
    token_freq = torch.load(f'./token_freq/MolGen-large_zinc250k.pt')
    assert token_freq.size(0) == len(tokenizer)

    def token_freq_preprocess_fn(wf):
        wf = wf + 1
        wf = wf.log()
        wf = wf / wf.max()
        return wf

    def process_fn_in_collate(wf):
        return wf - wf.mean()

    token_freq = token_freq_preprocess_fn(token_freq)
    token_freq[tokenizer.pad_token_id] = 0.  # stable training

    if args.sample_strategy == 'Categorical':
        sample_cls = Categorical()
    elif args.sample_strategy == 'wwm':
        sample_cls = WholeWordMasking(tokenizer)
    else:
        raise ValueError

    diffusion_schedule = diffusion_token_freq.create_discrete_diffusion_schedule(args.schedule, num_steps=args.num_steps)
    diffusion_instance = diffusion_token_freq.MaskDiffusion(
        dim=len(tokenizer),
        schedule=diffusion_schedule,
        tokenizer=tokenizer,
        sample_cls=sample_cls,
        # token_freq=token_freq,
        token_freq_lambda=args.token_freq_lambda,
        device=device
    )

    if args.load_step > 0:
        ckpt = torch.load('model_name_/MolGen-large_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_layerwise_ckpts/best(44999).th')
    cfg = AutoConfig.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large')
    cfg.overall_timestep = diffusion_instance.num_steps
    model = BartForConditionalGeneration.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large').to(device)
    # for name, param in model.named_parameters():
    #     print(f"参数名称: {name}, 参数大小: {param.size()}, 参数元素数: {param.numel()}")


    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda n: n / 10000. + 1e-3 if n < 10000 else 100. / math.sqrt(n))

    train_data, = DiffusionLoader(tokenizer=tokenizer).my_load(task_name='zinc250k', splits=['train'])
    train_data, dev_data = train_data.train_test_split(test_size=args.dev_size).values()

    logger = fastNLP.logger
    if dist.get_rank() == 0:
        print('# of train data: {}'.format(len(train_data)))
        print('Example:')
        print(train_data[0])
        print('\n# of dev data: {}'.format(len(dev_data)))
        print('Example:')
        print(dev_data[0])
        # print('\n# of test data: {}'.format(len(test_data)))
        # print('Example:')
        # print(test_data[0])

    def collate_fn(batch_input):
        input_ids = [torch.tensor(d['input_ids']) for d in batch_input]
        attention_mask = [torch.tensor(d['attention_mask']) for d in batch_input]
        token_freq_logits = [process_fn_in_collate(token_freq.gather(0, torch.tensor(d['input_ids']))) for d in batch_input]
        input_ids = pad_sequence(input_ids, batch_first=True)
        attention_mask = pad_sequence(attention_mask, batch_first=True)
        token_freq_logits = pad_sequence(token_freq_logits, batch_first=True)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_freq_logits': token_freq_logits
        }


    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=train_sampler)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=args.batch_size * 2, collate_fn=collate_fn, num_workers=4, pin_memory=True, sampler=dev_sampler)

    model.train()

    cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
    sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)
    att_ones = torch.ones((1, 1), device=device)
    att_zeros = torch.zeros((1, 1), device=device)

    def denoise_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        return model(input_ids=targets, attention_mask=attention_mask)['logits'][:, 1:-1, :]

    if dist.get_rank() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        best_dev_elbo = float('inf')

train_loss = .0
nan_count = 0
loss_list = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]
# accumulation_steps = 64
for epoch in range(args.epochs):
    if dist.get_rank() == 0:
        logger.info(f"开始第 {epoch + 1}/{args.epochs} 个 epoch")
        epoch_start_time = datetime.datetime.now()
    train_loader.sampler.set_epoch(epoch)
    dev_loader.sampler.set_epoch(epoch)
    for i, batch in enumerate(tqdm(train_loader), args.load_step + 1):
        metrics = diffusion_token_freq.compute_kl_reverse_process(
            batch['input_ids'].to(device),
            diffusion_instance.sample_t(),
            denoise_fn=denoise_fn,
            diffusion=diffusion_instance,
            target_mask=batch['attention_mask'].to(device),
            hybrid_lambda=args.hybrid_lambda,
            predict_x0=args.predict_x0,
            token_freq_logits=batch['token_freq_logits'].to(device)
        )

        loss = metrics['loss'] / args.batch_size
        # loss = loss / accumulation_steps
        dist.all_gather(loss_list, loss)
        if torch.stack(loss_list).isnan().any():
            nan_count += 1
            if dist.get_rank() == 0:
                logger.warning(f'在第 {i} 步遇到了 NaN，第 {nan_count} 次')
            continue
        train_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 5)
        # if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        warmup_scheduler.step()

        if dist.get_rank() == 0:
            if i % args.logging_steps == args.logging_steps - 1:
                avg_loss = train_loss / args.logging_steps
                logger.info(f'第 {epoch + 1}/{args.epochs} 个 epoch，第 {i} 步，平均损失: {avg_loss:.4f}')
                # fitlog.add_loss(avg_loss, name='train_loss', step=i)
                train_loss = .0

        if i % args.eval_steps == args.eval_steps - 1:
            nan_count_in_dev = 0
            model.eval()
            dev_metrics = {
                'elbo': .0,
                'elbo_in_bits_per_dim': .0,
                # 'likelihood': .0,
                # 'prior': .0,
            }

            if dist.get_rank() == 0:
                logger.info(f"在第 {i} 步开始验证集评估")

            with torch.no_grad():
                for dev_batch in dev_loader:
                    batch_dev_metrics = diffusion_token_freq.discrete_diffusion_elbo(
                        dev_batch['input_ids'].to(device),
                        denoise_fn=denoise_fn,
                        diffusion=diffusion_instance,
                        target_mask=dev_batch['attention_mask'].to(device),
                        normalize_without_padding=True,
                        eval_step_size=args.eval_step_size,
                        token_freq_logits=dev_batch['token_freq_logits'].to(device),
                        device=device
                    )

                    if dist.get_rank() == 0:
                        m = [torch.tensor(0., device=device) for _ in range(dist.get_world_size())]
                        for name in dev_metrics.keys():
                            dist.gather(batch_dev_metrics[name].squeeze(), m)
                            temp = sum(m)
                            if not torch.isnan(temp):
                                dev_metrics[name] += temp
                            else:
                                nan_count_in_dev += 1
                                logger.warning(f'在验证集第 {i} 步遇到了 NaN，第 {nan_count_in_dev} 次')
                    else:
                        for name in dev_metrics.keys():
                            dist.gather(batch_dev_metrics[name].squeeze())

                if dist.get_rank() == 0:
                    for name in dev_metrics.keys():
                        dev_metrics[name] /= len(dev_data)
                        # fitlog.add_metric(dev_metrics[name], name=name, step=i)
                        logger.info(f'验证集 {name}: {dev_metrics[name]:.4f}，在第 {i} 步')

                    if dev_metrics['elbo_in_bits_per_dim'] <= best_dev_elbo:
                        best_dev_elbo = dev_metrics['elbo_in_bits_per_dim']
                        # fitlog.add_best_metric(dev_metrics['elbo_in_bits_per_dim'], name='dev_elbo_in_bits_per_dim')
                        torch.save({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'warmup_scheduler': warmup_scheduler.state_dict(),
                        }, f'./{save_path}/best({i}).th')
                        logger.info(f'在第 {i} 步保存了新的最佳模型')
            model.train()
