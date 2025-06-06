__author__ = 'Rio'
from yacs.config import CfgNode as CN
import os
config = CN()
config.base_path = r'C:\Users\user\Documents\GitHub\Ultrasound-Report-Generation'

config.models = os.path.join(config.base_path, 'Result', 'Models', 'breast_model_best.pth')

config.dataset_name = 'Mammary'
config.distiller_num = 18 # liver:18; Mamary:18; Thyroid:5
config.jieba_dir = os.path.join(config.base_path, 'Data', 'key_technical_words.txt')
config.image_dir = os.path.join(config.base_path, 'Data', 'Thyroid_report')
config.ann_path = os.path.join(config.base_path, 'Data', 'new_Thyroid2.json')
config.technical_word = os.path.join(config.base_path, 'Data', 'key_technical_words.txt')

config.dict_pth = ' '

config.max_seq_length_train = 100  # Reduced from 150 to save memory
config.max_seq_length = 100  # Reduced from 150 to save memory
config.threshold = 3
config.num_workers = 0
config.batch_size = 8  # Reduced from 32 to 8 to fit in 8GB GPU memory
config.evaluate_batch = 1

# Use ViT to extract visual features
config.visual_extractor = 'vit_base_patch16_224'
config.visual_extractor_pretrained = True

config.d_model = 512
config.d_ff = 512
config.d_vf = 768  # Changed from 2048 to 768 to match ViT base embedding dimension
config.num_heads = 8
config.num_layers = 2  # Reduced from 3 to 2 layers to save memory
config.dropout = 0.1
config.logit_layers = 1
config.bos_idx = 0
config.eos_idx = 0
config.pad_idx = 0
config.use_bn = 0
config.drop_prob_lm = 0.5

config.sample_n = 1
config.output_logsoftmax = 1
config.decoding_constraintt = 0

config.n_gpu = 1
config.epochs = 30

config.save_dir = os.path.join(config.base_path, 'Result', 'Models')
config.record_dir = os.path.join(config.base_path, 'Result', 'Records')

config.save_period = 1
config.monitor_mode = 'max'
config.monitor_metric = 'BLEU_4'
config.early_stop = 100
config.image_type = '2d'

config.optim = 'Adam'
config.lr_ve = 5e-5
config.lr_ed = 1e-3
config.weight_decay = 5e-5
config.amsgrad = True

config.lr_scheduler = 'StepLR'
config.step_size = 28
config.gamma = 0.1

config.seed = 9233
config.resume = None

config.embedding_vector = 300
config.nhidden = 512
config.nlayers = 1
config.bidirectional = True
config.rnn_type = 'LSTM'

# text_image_losses.py
config.cuda = True
config.train_smooth_gamma3 = 10.0
config.train_smooth_gamma2 = 5.0
config.train_smooth_gamma1 = 4.0
config.attn_pth = os.path.join(config.base_path, 'Result', 'Attn_pth')



