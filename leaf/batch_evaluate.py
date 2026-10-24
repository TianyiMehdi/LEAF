import argparse
import os
os.environ["MUJOCO_GL"] = os.getenv("MUJOCO_GL", "egl")
import pickle
import warnings
from pathlib import Path
from types import SimpleNamespace

warnings.filterwarnings("ignore")


DEFAULT_LOGS_ROOT = Path(__file__).resolve().parent / "logs" / "leaf"


class Config(SimpleNamespace):
	def get(self, key, default=None):
		return getattr(self, key, default)


def str2bool(value):
	if isinstance(value, bool):
		return value
	value = value.strip().lower()
	if value in {"true", "1", "yes", "y"}:
		return True
	if value in {"false", "0", "no", "n"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def safe_close_env(env):
	close_fn = getattr(env, "close", None)
	if close_fn is None:
		return
	try:
		close_fn()
	except AttributeError:
		pass


def parse_args():
	parser = argparse.ArgumentParser(
		description="Batch-evaluate all final.pt checkpoints under selected task directories."
	)
	parser.add_argument(
		"--algorithm",
		required=True,
		help="Algorithm name used in the output dict key. Example: flow_guidance",
	)
	parser.add_argument(
		"--task",
		nargs="+",
		required=True,
		help="One or more task names. Example: --task humanoid-run dog-run",
	)
	parser.add_argument(
		"--output",
		required=True,
		help="Path to the output pkl file.",
	)
	parser.add_argument(
		"--logs-root",
		default=str(DEFAULT_LOGS_ROOT),
		help=f"Root directory containing per-task log folders. Default: {DEFAULT_LOGS_ROOT}",
	)
	parser.add_argument(
		"--eval-episodes",
		type=int,
		default=5,
		help="Override eval_episodes for every checkpoint. Default: use each run's saved config.",
	)
	parser.add_argument(
		"--checkpoint-name",
		default="final.pt",
		help="Checkpoint filename to collect under each run's models directory.",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=None,
		help="Override evaluation seed. Default: use each run's saved config.",
	)
	parser.add_argument(
		"--use_mpc",
		type=str2bool,
		default=False,
		help="Whether to use MPC. Accepts true/false.",
	)
	parser.add_argument(
		"--flow_steps",
		type=int,
		default=1,
		help="Flow steps",
	)
	parser.add_argument(
		"--aggregate",
		choices=["all", "best_ckpt"],
		default="all",
		help="How to aggregate multiple checkpoints: 'all' pools all episode rewards, 'best_ckpt' selects the checkpoint with the highest per-checkpoint mean reward.",
	)
	parser.add_argument(
		"--use_guidance",
		type=str2bool,
		default=False,
		help="Whether to use guidance. Accepts true/false.",
	)
	return parser.parse_args()


def parse_tasks(task_args):
	tasks = []
	seen = set()
	for raw_item in task_args:
		for task_name in raw_item.split(","):
			task_name = task_name.strip()
			if not task_name:
				continue
			if task_name in seen:
				continue
			seen.add(task_name)
			tasks.append(task_name)
	if not tasks:
		raise ValueError("At least one task name must be provided.")
	return tasks


def load_run_cfg(cfg_path, checkpoint_path, args):
	import yaml

	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg_dict = yaml.unsafe_load(f)
	cfg = Config(**cfg_dict)
	cfg.checkpoint = str(checkpoint_path)
	cfg.save_video = False
	cfg.enable_wandb = False
	cfg.use_tensorboard = False
	cfg.mpc = args.use_mpc
	cfg.num_flow_steps = args.flow_steps
	cfg.use_score_guidance = args.use_guidance
	if args.eval_episodes is not None:
		cfg.eval_episodes = args.eval_episodes
	if args.seed is not None:
		cfg.seed = args.seed

	compat_defaults = {
		"multitask": False,
		"tasks": [cfg.task],
		"compile": False,
		"score_guidance_coef": 0.1,
		"flow_value_weight": True,
	}
	for key, value in compat_defaults.items():
		if not hasattr(cfg, key) or getattr(cfg, key) is None:
			setattr(cfg, key, value)
	return cfg


def find_checkpoints(logs_root, task_name, checkpoint_name):
	task_dir = Path(logs_root) / task_name
	if not task_dir.exists():
		raise FileNotFoundError(f"Task directory not found: {task_dir}")
	checkpoints = sorted(task_dir.glob(f"*/models/{checkpoint_name}"))
	if not checkpoints:
		raise FileNotFoundError(
			f'No checkpoints named "{checkpoint_name}" found under {task_dir}'
		)
	return checkpoints


def evaluate_checkpoint(agent, env, cfg):
	ep_rewards = []
	for _ in range(cfg.eval_episodes):
		obs = env.reset()
		done = False
		ep_reward = 0.0
		t = 0
		while not done:
			action, _ = agent.act(obs, t0=t == 0, eval_mode=True)
			obs, reward, done, _ = env.step(action)
			ep_reward += reward
			t += 1
		ep_rewards.append(ep_reward)
	return ep_rewards


def main():
	args = parse_args()

	import numpy as np
	import torch
	from termcolor import colored

	from common.seed import set_seed
	from envs import make_env
	from leaf import LEAF

	torch.backends.cudnn.benchmark = True
	assert torch.cuda.is_available(), "CUDA is required for evaluation."

	def build_agent(cfg):
		return LEAF(cfg)

	algorithm = args.algorithm.strip()
	if not algorithm:
		raise ValueError("Algorithm name must be non-empty.")
	tasks = parse_tasks(args.task)
	output_path = Path(args.output)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	if output_path.exists():
		with open(output_path, "rb") as f:
			results = pickle.load(f)
		if not isinstance(results, dict):
			raise TypeError(f"Existing output file must contain a dict, got {type(results).__name__}")
	else:
		results = {}

	for task_name in tasks:
		checkpoints = find_checkpoints(args.logs_root, task_name, args.checkpoint_name)
		all_rewards = []
		checkpoint_stats = []
		print(colored(f"Task: {task_name}", "blue", attrs=["bold"]))
		for checkpoint_path in checkpoints:
			run_dir = checkpoint_path.parents[1]
			cfg_path = run_dir / "config.yaml"
			if not cfg_path.exists():
				raise FileNotFoundError(f"Missing run config: {cfg_path}")
			cfg = load_run_cfg(
				cfg_path,
				checkpoint_path,
				args,
			)
			if getattr(cfg, "steps", None) == 2000000:
				print(colored(f"  Skipping {checkpoint_path} because steps=2000000", "red"))
				continue
			set_seed(cfg.seed)
			env = make_env(cfg)
			agent = build_agent(cfg)
			agent.load(str(checkpoint_path))
			with torch.no_grad():
				ep_rewards = evaluate_checkpoint(agent, env, cfg)
			checkpoint_mean = float(np.mean(ep_rewards))
			checkpoint_std = float(np.std(ep_rewards))
			checkpoint_stats.append({
				"path": str(checkpoint_path),
				"mean": checkpoint_mean,
				"std": checkpoint_std,
				"rewards": ep_rewards,
			})
			all_rewards.extend(ep_rewards)
			print(
				colored(
					f"  {checkpoint_path} -> reward (mean={checkpoint_mean:.4f}, std={checkpoint_std:.4f}) "
					f"({cfg.eval_episodes} episodes)",
					"yellow",
				)
			)
			safe_close_env(env)

		if not checkpoint_stats:
			print(colored(f"  No valid checkpoints left for {task_name} after filtering, skipping.", "red", attrs=["bold"]))
			continue

		if args.aggregate == "all":
			task_mean = float(np.mean(all_rewards))
			task_std = float(np.std(all_rewards))
			summary = f"all checkpoints / {len(all_rewards)} episodes"
		else:
			best_ckpt = max(checkpoint_stats, key=lambda item: item["mean"])
			task_mean = best_ckpt["mean"]
			task_std = best_ckpt["std"]
			summary = f"best checkpoint {best_ckpt['path']}"

		output_key = (task_name, algorithm)
		results[output_key] = (task_mean, task_std)
		print(
			colored(
				f"  Saved key {output_key} with reward (mean={task_mean:.4f}, std={task_std:.4f}) from {summary}",
				"green",
				attrs=["bold"],
			)
		)

	with open(output_path, "wb") as f:
		pickle.dump(results, f)
	print(colored("Final results:", "cyan", attrs=["bold"]))
	for key, value in results.items():
		mean, std = value
		print(colored(f"  {key}: (mean={mean:.4f}, std={std:.4f})", "cyan"))
	print(colored(f"Results written to {output_path}", "green", attrs=["bold"]))


if __name__ == "__main__":
	main()
