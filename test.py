from test_evaluate import eval_

name = "D3PM"
token_length = 37
temperature = 0.3

# eval_file = f'./generation_molecules/{name}/temperature_{temperature}/length_{token_length}.txt'
eval_file = f'./generation_molecules/{name}/temperature_{temperature}/all_10k_samples.txt'
eval_(eval_file, 'selfies')