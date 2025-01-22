import torch
import os
# from transformers import BertTokenizer, BertConfig
from transformers import AutoTokenizer, AutoConfig, BartForConditionalGeneration
from sample import Categorical, WholeWordMasking
import time
from tqdm import tqdm
# from compute_metric import bleu, self_bleu
# import nltk
import argparse
import diffusion_word_freq as diffusion

parser = argparse.ArgumentParser()
parser.add_argument("--topk", default=30, type=int, required=False)
parser.add_argument("--step_size", default=2, type=int, required=False)
parser.add_argument("--name", default='token_freq_D3PM', type=str, required=False)
args = parser.parse_args()

step_size = args.step_size
device = 'cuda:1'
model_name = '/NAS/luoyc/wuyux/data/MolGen-large'
predict_x0 = True
sample_strategy = 'Categorical'
num_steps = 512
schedule = 'mutual'
topk = args.topk
iteration = 1
name = args.name
temperature = 0.3
seq_length = 37
shape = torch.Size([1000, seq_length])

model_path_dict = {
    # 'D3PM': ('/remote-home/zfhe/projects/diffusion_torch/D3PM_new_timestep_ckpts/best(1799999).th', 'layerwise'),
    # 'dbnotimestep': ('/remote-home/zfhe/projects/diffusion_torch/diffusion_bert_base_no_timestep_ckpts/best(1749999).th', 'none'),
    # 'dbnewtimestep': ('/remote-home/zfhe/projects/diffusion_torch/diffusion_bert_base_new_timestep_ckpts/best(1849999).th', 'layerwise'),
    # 'dbtokentimestep': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_3e-06_seed_42_numsteps_512_sample_Categorical_hybridlambda_0.01_schedule_mutual_new_attmask_ckpts/best(1549999).th', 'token'),
    # 'word_freq5': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_3e-06_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.5_ckpts/best(1749999).th', 'embedding'),
    # 'word_freq3': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_3e-06_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_ckpts/best(1849999).th', 'none'),
    # 'word_freq3_newtimestep': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_3e-06_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_False_new_timestep_ckpts/best(1499999).th', 'layerwise'),
    # 'word_freq_D3PM': ('/remote-home/zfhe/projects/diffusion_torch/model_name_bert-base-uncased_lr_8e-06_seed_42_numsteps_512_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.5_frpmscratch_True_ckpts/best(1849999).th', 'layerwise')
    'token_freq_D3PM': ('/NAS/luoyc/wuyux/project/selfies/model_name_/NAS/luoyc/wuyux/data/MolGen-large_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.3_fromscratch_True_timestep_none_ckpts/best(999).th', 'none'),
    'D3PM': ('/NAS/luoyc/wuyux/project/selfies/model_name_MolGen-large-random_lr_5e-05_seed_42_numsteps_2048_sample_Categorical_schedule_mutual_hybridlambda_0.01_wordfreqlambda_0.0_fromscratch_True_timestep_none_ckpts/best(499).th', 'none')
}
model_ckpt_path, timestep = model_path_dict[name]
if name.startswith('word_freq'):
    kind = 'word_freq'
else:
    kind = 'base'

# if timestep in ['none', 'token']:
#     from models.modeling_bert import BertForMaskedLM
# elif timestep == 'embedding':
#     from models.modeling_bert_timestep import BertForMaskedLM
# elif timestep == 'layerwise':
#     from models.modeling_bert_new_timestep import BertForMaskedLM
# else:
#     raise NotImplementedError


# if model_name in ['fnlp/elasticbert-base', 'fnlp/elasticbert-large']:
#     model_cls = ElasticBertForPreTraining
#     cfg_cls = ElasticBertConfig
#     tok_cls = ElasticBertTokenizer
# elif model_name in ['bert-base-uncased', 'bert-large-uncased']:
#     model_cls = BertForMaskedLM
#     cfg_cls = BertConfig
#     tok_cls = BertTokenizer
# else:
#     raise NotImplementedError


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

# import diffusion_word_freq as diffusion
word_freq = torch.load(f'./token_freq/MolGen-large_zinc250k.pt', weights_only=True).to(device)
def word_freq_preprocess_fn(wf):
    wf = wf + 1
    wf = wf.log()
    wf = wf / wf.max()

    # range: 0 - 1
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

# if model_name in ['fnlp/elasticbert-base', 'fnlp/elasticbert-large']:
#     cfg.num_output_layers = cfg.num_hidden_layers
#     cfg.num_base_layers = 0
# model = model_cls(cfg).to(device)
model = BartForConditionalGeneration.from_pretrained('/NAS/luoyc/wuyux/data/MolGen-large').to(device)

ckpt = torch.load(model_ckpt_path, map_location=device)
state_dict = ckpt['model']

# 使用字典推导式移除 'module.' 前缀
new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}

# 加载新的状态字典到模型中
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
# att_ones = torch.ones((1, 1), device=device)
# att_zeros = torch.zeros((1, 1), device=device)


model.eval()

output_file = f'./generation_molecules/{name}_length_{seq_length}_temperature_{temperature}.txt'
curve_file = f'./generation_results/{name}_length_{seq_length}_temperature_{temperature}_curve.txt'

with open(output_file, 'w') as fdata:
    with open(curve_file, 'w') as fcurve:
        sentences = []
        wfs = []
        with torch.no_grad():
            for i in tqdm(range(iteration)):
                start = time.time()
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
                t = time.time() - start
                print(t, file=fcurve, end=' ')
                sentence = tokenizer.batch_decode(state)
                sentences.extend(sentence)
                # print(sentence)
                for s in sentence:
                    print(s, file=fdata, flush=True)
