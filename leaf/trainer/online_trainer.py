from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainer(Trainer):
	"""Trainer class for single-task online training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._saved_checkpoints = set()
		self._last_checkpoint_step = None
		# Keep Phase II out of training interactions and replay reanalysis.
		self.agent.cfg.use_score_guidance = False
		self.agent.model.cfg.use_score_guidance = False

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate an agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.reset(), False, 0, 0
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action, _ = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)

	def to_td(self, obs, action=None, reward=None, plan_info=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if plan_info is None:
			plan_info = TensorDict({})
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan'))
		expert_value = plan_info.get("action_value", torch.tensor(float('nan'))).squeeze().unsqueeze(0)
		expert_action_dist = plan_info.get("action_dist", \
	  		torch.full(size=(1, 2*self.cfg.action_dim), fill_value=float('nan')))
		expert_flow_action = plan_info.get("flow_action", \
	  		torch.full(size=(1, self.cfg.topk_flow, self.cfg.action_dim), fill_value=float('nan')))
		expert_flow_value = plan_info.get("flow_value", \
			torch.full(size=(1, self.cfg.topk_flow), fill_value=float('nan')))
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0),
			reward=reward.unsqueeze(0),
			expert_value=expert_value,
			expert_action_dist=expert_action_dist,
			expert_flow_action=expert_flow_action,
			expert_flow_value=expert_flow_value,
		), batch_size=(1,))
		return td

	def _checkpoint_identifier(self, step):
		"""Return a sortable checkpoint name for a training step."""
		return f"step_{step // 1000:04d}k"

	def _save_checkpoint_if_needed(self):
		"""Save periodic checkpoints, including step 0 and the final configured step."""
		checkpoint_freq = getattr(self.cfg, "checkpoint_save_freq", 100000)
		if checkpoint_freq <= 0 or self._step in self._saved_checkpoints:
			return
		if self._step == 0 or self._step == self.cfg.steps or self._step % checkpoint_freq == 0:
			identifier = self._checkpoint_identifier(self._step)
			self.logger.save_agent(self.agent, identifier=identifier)
			self._saved_checkpoints.add(self._step)
			self._last_checkpoint_step = self._step


	def train(self):
		"""Train an agent."""
		train_metrics, done, eval_next = {}, True, False
		while self._step <= self.cfg.steps:
			self.agent.cfg.use_score_guidance = False
			self.agent.model.cfg.use_score_guidance = False
			self._save_checkpoint_if_needed()

			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True

			# Reset environment
			if done:
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					if self.agent.cfg.eval_pi: # evaluate flow policy with Phase II refinement
						mpc = self.agent.cfg.mpc
						num_flow_steps = self.agent.cfg.num_flow_steps
						use_score_guidance = self.agent.cfg.use_score_guidance
						self.agent.cfg.mpc = False
						self.agent.cfg.num_flow_steps = 1
						self.agent.cfg.use_score_guidance = True
						self.agent.model.cfg.use_score_guidance = True
						try:
							eval_metrics = self.eval()
							eval_metrics.update(self.common_metrics())
							self.logger.log(eval_metrics, "eval_pi")
						finally:
							self.agent.cfg.mpc = mpc
							self.agent.cfg.num_flow_steps = num_flow_steps
							self.agent.cfg.use_score_guidance = use_score_guidance
							self.agent.model.cfg.use_score_guidance = use_score_guidance
					eval_next = False

				if self._step > 0:
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(torch.cat(self._tds))

				obs = self.env.reset()
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step > self.cfg.seed_steps:
				torch.compiler.cudagraph_mark_step_begin()
				action, plan_info = self.agent.act(obs, t0=len(self._tds)==1)
			else:
				action = self.env.rand_act()
				plan_info = None
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, plan_info))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					pretrain = True
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
					pretrain = False
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer, pretrain)
				train_metrics.update(_train_metrics)

			# GMM policy entropy estimation every N steps (e.g. 20k), log to wandb
			entropy_interval = getattr(self.cfg, 'entropy_est_interval', 20000)
			if (
				entropy_interval > 0
				and self._step > 0
				and self._step % entropy_interval == 0
				and hasattr(self.agent, 'estimate_policy_entropy')
			):
				obs_batch, _, _, task_batch, _ = self.buffer.sample()
				policy_entropy_gmm = self.agent.estimate_policy_entropy(obs_batch[0], task=task_batch)
				self.logger.log({'step': self._step, 'pi_entropy': policy_entropy_gmm}, 'train')

			self._step += 1

		final_agent = None if self._last_checkpoint_step == self.cfg.steps else self.agent
		self.logger.finish(final_agent)
