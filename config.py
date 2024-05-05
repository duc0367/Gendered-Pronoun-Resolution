import torch


class CFG:
    data_folder = 'data/gap-development.tsv'
    max_length = 512
    n_split = 3
    batch_size = 16
    n_epochs = 5
    learning_rate = 1e-3
    weight_decay = 1e-3
    gradient_accumulation_steps = 1
    enable_scaler = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_norm = 1000
    scheduler = True
    num_warmup_steps = 0
    print_freq = 32
    model_dir = 'model'
