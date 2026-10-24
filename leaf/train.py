import os
os.environ['MUJOCO_GL'] = os.getenv("MUJOCO_GL", 'egl')
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg, save_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from leaf import LEAF
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer import OnlineTrainer
from common.logger import Logger, TBLogger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config', config_path='.')
def train(cfg: dict):
	"""
	Script for training a LEAF agent.

	Most relevant args:
		`task`: task name
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps
		`seed`: random seed

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=dog-run steps=1000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print('pid:', os.getpid(), flush=True)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
	logger_cls = TBLogger if cfg.use_tensorboard else Logger
	trainer = trainer_cls(
		cfg=cfg,
		env=make_env(cfg),
		agent=LEAF(cfg),
		buffer=Buffer(cfg),
		logger=logger_cls(cfg),
	)
	save_cfg(cfg, cfg.work_dir) # save parsed config, must do it after the Logger's init
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
