import numpy as np
import torch
import os
import gc
from config import config as args
from models.SGF_model import SGF
from modules.MyTrainer import Trainer
from modules.dataloaders import MyDataLoader
from modules.loss import compute_loss
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def setup_memory_optimization():
    """Setup memory optimization for training"""
    # Enable expandable segments to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Clear GPU memory cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU memory optimization enabled")
    
    # Set memory-efficient backends
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    # Setup memory optimization first
    setup_memory_optimization()
    
    # fix random seed
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    tokenizer = Tokenizer(args)

    train_dataloader = MyDataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = MyDataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = MyDataLoader(args, tokenizer, split='test', shuffle=False)

    model = SGF(args, tokenizer)
    criterion = compute_loss
    metrics = compute_scores

    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                      test_dataloader)
    
    # Print memory info before training
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"GPU Total Memory: {total_memory:.2f} GB")
        print(f"Using batch size: {args.batch_size}")
    
    trainer.train()


if __name__ == '__main__':
    main()
