import math
import random
import numpy as np
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataset.shrec24_dataset import FinetuningDataset

import os
import os.path as opt
import sys
sys.path.append('./model')
from model.vit import ViT
from model.mae import MAE
from model.stgcn import STGCN
from utils.stgcn_utils import eval_stgcn


def seed_everything(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Training')
	parser.add_argument('--cfg_path', default='configs/shrec24_configs.yaml', help='Path to the train.yaml config')
	args = parser.parse_args()
	## configs
	cfg_path = args.cfg_path
	args = OmegaConf.load(cfg_path)

	device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
	seed_everything(args.seed)

	print('\nARGUMENTS....')
	for arg, val in args.__dict__['_content'].items():
		if isinstance(val, omegaconf.nodes.AnyNode):
			print('> {}: {}'.format(arg, val))
		else:
			print('> {}'.format(arg))
			for arg_, val_ in val.items():
				print('\t- {}: {}'.format(arg_, val_))


	print(15*'=', '\nEVALUATION', 15*'=')

	print('\nLoading Finetuning data....')
	data_args = args.data
	mae_args = args.mae
	stgcn_args = args.stgcn

	valid_set = FinetuningDataset(data_dir=data_args.test_data_dir,
								 T=stgcn_args.sequence_length,
								 normalize=data_args.normalize)

	print('# Test: {}'.format(len(valid_set)))
	valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False)

	print('\nLOADING MAE PRE-TRAINED WEIGHTS....', end='')	
	encoder = ViT(
				num_nodes=mae_args.num_joints,
				node_dim=mae_args.coords_dim,
				num_classes=mae_args.coords_dim,
				dim=mae_args.encoder_embed_dim,
				depth=mae_args.encoder_depth,
				heads=mae_args.num_heads,
				mlp_dim=mae_args.mlp_dim ,
				pool = 'cls',
				dropout = 0.,
				emb_dropout = 0.
			)

	mae = MAE(encoder=encoder,
			  decoder_dim=mae_args.decoder_dim,
			  decoder_depth=mae_args.decoder_depth,
			  masking_strategy=mae_args.masking_strategy,
			  masking_ratio=mae_args.masking_ratio
			)
	
	chkpt = opt.join(args.save_folder_path, args.exp_name,'weights', 'best_mae_model.pth')
	if not os.path.isfile(chkpt):
		print(f"File not found: ", chkpt)
		raise FileNotFoundError

	mae_chkpt = torch.load(chkpt)
	print('[optimal epoch={}]'.format(mae_chkpt['epoch']))
	mae.load_state_dict(mae_chkpt['state_dict'])
	mae = mae.to(device)

	print('\nLOADING STGCN PRE-TRAINED WEIGHTS....')
	stgcn = STGCN(channel=64,
				  num_class=stgcn_args.num_classes,
				  window_size=stgcn_args.sequence_length,
				  num_point=mae_args.num_joints,
				  num_person=1,
				  use_data_bn=False,
				  backbone_config=None,
				  graph_args={'config':'shrec24'},
				  mask_learning=False,
				  use_local_bn=False,
				  multiscale=False,
				  temporal_kernel_size=11,
				  dropout=0.5)

	stgcn_chkpt = opt.join(args.save_folder_path, args.exp_name,'weights', 'best_stgcn_model.pth')
	if not os.path.isfile(stgcn_chkpt):
		print(f"File not found: ", stgcn_chkpt)
		raise FileNotFoundError

	stgcn_chkpt = torch.load(stgcn_chkpt)
	stgcn.load_state_dict(stgcn_chkpt['state_dict'])
	stgcn = stgcn.to(device)

	print('\nINFERENCE [optimal MAE epoch={} & optimal STGCN epoch={}]\n'.format(mae_chkpt['epoch'], stgcn_chkpt['epoch']))
	eval_stgcn(stgcn, mae, valid_loader, device, args)