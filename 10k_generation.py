import torch
import os
from transformers import AutoTokenizer, AutoConfig, BartForConditionalGeneration
from sample import Categorical, WholeWordMasking
import time
from tqdm import tqdm
import argparse
import diffusion_token_freq as diffusion
import json

parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=30, type=int, required=False)
parser.add_argument("--step_size", default=2, type=int, required=False)
parser.add_argument("--name", default='token_freq_D3PM', type=str, required=False)
parser.add_argument("--temperature", default=0.3, type=float, required=False)
args = parser.parse_args()

step_size = args.step_size
device = 'cuda:3'
model_name = '/NAS/luoyc/wuyux/data/MolGen-large'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 512
schedule = 'mutual'
topk = args.topk
iteration = 1
name = args.name
temperature = args.temperature
# seq_length = 37
# shape = torch.Size([1000, seq_length])

model_path_dict = {
    'token_freq_D3PM': ('/NAS/luoyc/wuyux/project/selfies/model_name_/NAS/luoyc/wuyux/data/MolGen-large_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts/best(999).th', 'none'),
    'D3PM': ('/NAS/luoyc/wuyux/project/selfies/model_name_MolGen-large-random_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.0_fromscratch_True_timestep_none_ckpts/best(499).th', 'none')
}
model_ckpt_path, timestep = model_path_dict[name]
if name.startswith('word_freq'):
    kind = 'word_freq'
else:
    kind = 'base'

tokenizer = AutoTokenizer.from_pretrained(model_name)


if sample_strategy == 'Categorical':
    sample_cls = Categorical()
elif sample_strategy == 'wwm':
    sample_cls = WholeWordMasking(tokenizer)
else:
    raise ValueError


if kind == 'word_freq':
    word_freq_lambda = 0.3
elif kind == 'base':
    word_freq_lambda = 0.0
else:
    raise ValueError

word_freq = torch.load(f'./token_freq/MolGen-large_zinc250k.pt', weights_only=True).to(device)
def word_freq_preprocess_fn(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()

    return wf

word_freq = word_freq_preprocess_fn(word_freq)
diffusion_schedule = diffusion.create_discrete_diffusion_schedule(schedule, num_steps=num_steps)
diffusion_instance = diffusion.MaskDiffusion(
    dim=len(tokenizer),
    schedule=diffusion_schedule,
    tokenizer=tokenizer,
    sample_cls=sample_cls,
    word_freq=word_freq,
    word_freq_lambda=word_freq_lambda,
    device=device
)



cfg = AutoConfig.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large')
cfg.overall_timestep = diffusion_instance.num_steps
model = BartForConditionalGeneration.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large').to(device)

ckpt = torch.load(model_ckpt_path, map_location=device)
state_dict = ckpt['model']

new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)

cls = torch.full((1, 1), fill_value=tokenizer.cls_token_id, device=device)
sep = torch.full((1, 1), fill_value=tokenizer.sep_token_id, device=device)


if timestep == 'none':
    def denoise_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        # attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        return model(
            input_ids=targets,
            # timestep=timestep - 1,
            # attention_mask=attention_mask
        )['logits'][:, 1:-1, :]
elif timestep == 'token':
    def denoise_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((
            cls.repeat(bsz, 1),
            torch.full((bsz, 1), fill_value=timestep.item() + 110, device=device),
            targets,
            sep.repeat(bsz, 1)
        ), dim=1)
        # attention_mask = torch.cat((torch.ones((bsz, 2), device=device), attention_mask, torch.zeros((bsz, 1), device=device)), dim=1)
        return model(
            input_ids=targets,
            timestep=timestep - 1,
            # attention_mask=attention_mask
        )['logits'][:, 2:-1, :]
elif timestep in ['layerwise', 'embedding']:
    def denoise_fn(targets, timestep, attention_mask):
        assert len(targets.size()) == 2  # bsz * seqlen
        bsz = targets.size(0)
        targets = torch.cat((cls.repeat(bsz, 1), targets, sep.repeat(bsz, 1)), dim=1)
        # attention_mask = torch.cat((att_ones.repeat(bsz, 1), attention_mask, att_zeros.repeat(bsz, 1)), dim=1)
        return model(
            input_ids=targets,
            timestep=timestep - 1,
            # attention_mask=attention_mask
        )['logits'][:, 1:-1, :]
else:
    raise NotImplementedError

all_molecules = []
model.eval()

with open('generation_distribution.json', 'r') as f:
    generation_dist = json.load(f)

for seq_length, num_samples in generation_dist.items():
    if num_samples == 0:
        continue
    output_file = f'./generation_molecules/{name}/temperature_{temperature}/length_{seq_length}.txt'
    shape = [num_samples, seq_length]
    with open(output_file, 'w') as fdata:
        with torch.no_grad():
            for i in tqdm(range(iteration)):
                state = diffusion.discrete_diffusion_predict_fn(
                    shape=shape,
                    denoise_fn=denoise_fn,
                    diffusion=diffusion_instance,
                    predict_x0=predict_x0,
                    sample_cls=sample_cls,
                    step_size=step_size,
                    topk=topk,
                    target_mask=torch.ones(shape, device=device),
                    show_process=False,
                    temperature=temperature
                    # word_freq=True
                    # context_fn=context_fn
                )['final_state']
                molecule = tokenizer.batch_decode(state)
                all_molecules.extend(molecule)
                # print(sentence)
                for s in molecule:
                    print(s, file=fdata, flush=True)

all_output_file = f'./generation_molecules/{name}/temperature_{temperature}/all_10k_samples.txt'
with open(all_output_file, 'w') as f:
    for mol in all_molecules:
        f.write(mol + '\n')

print(f"\nTotal generated samples: {len(all_molecules)}")
print(f"Results saved to {all_output_file}")
