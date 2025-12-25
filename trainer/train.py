
import os
import time
import torch
import random
import pickle
import numpy as np
import torch.nn as nn

from utils_nba.config import Config
from utils_nba.utils import print_log

from sklearn.metrics import accuracy_score

from torch.utils.data import DataLoader
from data_nba.dataloader_nba_tactic import NBADataset, seq_collate

from models.model_initializer import LEDInitializer as InitializationModel
from models.model_diffusion import TransformerDenoisingModel as DenoisingModel

from models.game_classifer_scene_banzhaf import TacticClassifierModel
from utils_nba.utils import rank_accuracy, MSE
from models.game import BanzhafInteraction, BanzhafModule

from torch.utils.tensorboard import SummaryWriter

from losses import *
import seaborn as sns
import matplotlib.pyplot as plt

import pdb
NUM_Tau = 5

class Trainer:
	def __init__(self, config):
		
		if torch.cuda.is_available(): torch.cuda.set_device(config.gpu)
		self.device = torch.device('cuda') if config.cuda else torch.device('cpu')
		self.cfg = Config(config.cfg, config.info)
		
		# data
		train_dset = NBADataset(
			obs_len=self.cfg.past_frames,
			pred_len=self.cfg.future_frames,
			training=True, aug=False)

		self.train_loader = DataLoader(
			train_dset,
			batch_size=self.cfg.train_batch_size,
			shuffle=True,
			num_workers=4,
			collate_fn=seq_collate,
			pin_memory=True)
		
		test_dset = NBADataset(
			obs_len=self.cfg.past_frames,
			pred_len=self.cfg.future_frames,
			training=False)

		self.test_loader = DataLoader(
			test_dset,
			batch_size=self.cfg.test_batch_size,
			shuffle=False,
			num_workers=4,
			collate_fn=seq_collate,
			pin_memory=True)
		
		self.traj_mean = torch.FloatTensor(self.cfg.traj_mean).cuda().unsqueeze(0).unsqueeze(0).unsqueeze(0)
		self.traj_scale = self.cfg.traj_scale
		self.temporal_reweight = torch.FloatTensor([21 - i for i in range(1, 21)]).cuda().unsqueeze(0).unsqueeze(0) / 10

		# diffusion
		self.w = self.cfg.classifier_diffusion.w
		self.n_steps = self.cfg.diffusion.steps
		self.betas = self.make_beta_schedule(
			schedule=self.cfg.diffusion.beta_schedule, n_timesteps=self.n_steps, 
			start=self.cfg.diffusion.beta_start, end=self.cfg.diffusion.beta_end).cuda()
		self.alphas = 1 - self.betas
		self.alphas_prod = torch.cumprod(self.alphas, 0)
		self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
		self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
		
		# model
		self.model = DenoisingModel(num_classes=16, ttlabel=True).cuda()
		model_cp = torch.load(self.cfg.pretrained_core_denoising_model, map_location='cpu')
		self.model.load_state_dict(model_cp['model_dict'])

		self.model_initializer = InitializationModel(t_h=self.cfg.past_frames, d_h=6, t_f=self.cfg.future_frames, d_f=2, k_pred=20, w_memory=self.w_memory, cfg=self.cfg).cuda()
		init_model_cp = torch.load(self.cfg.pretrained_initializer_model, map_location='cpu')
		self.model_initializer.load_state_dict(init_model_cp['model_initializer_dict'])

		self.cls_model = TacticClassifierModel().cuda()
		cls_model_cp = torch.load(self.cfg.pretrained_cls_model, map_location='cpu')
		self.cls_model.load_state_dict(cls_model_cp['model_dict'])
		
		self.params =  list(self.model_initializer.parameters()) + list(self.cls_model.parameters()) 
		self.opt = torch.optim.AdamW(self.params, lr=config.learning_rate)
		self.scheduler_model = torch.optim.lr_scheduler.StepLR(self.opt, step_size=self.cfg.decay_step, gamma=self.cfg.decay_gamma)

		self.criterion = MSE()
		self.gamma = self.cfg.gamma

		# log
		self.log = open(os.path.join(self.cfg.log_dir, 'log.txt'), 'a+')


	def make_beta_schedule(self, schedule: str = 'linear', 
			n_timesteps: int = 1000, 
			start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
		'''
		Make beta schedule.

		Parameters
		----
		schedule: str, in ['linear', 'quad', 'sigmoid'],
		n_timesteps: int, diffusion steps,
		start: float, beta start, `start<end`,
		end: float, beta end,

		Returns
		----
		betas: Tensor with the shape of (n_timesteps)

		'''
		if schedule == 'linear':
			betas = torch.linspace(start, end, n_timesteps) # beta_1, beta_T, steps
		elif schedule == "quad":
			betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
		elif schedule == "sigmoid":
			betas = torch.linspace(-6, 6, n_timesteps)
			betas = torch.sigmoid(betas) * (end - start) + start
		return betas


	def extract(self, input, t, x):
		shape = x.shape
		out = torch.gather(input, 0, t.to(input.device))
		reshape = [t.shape[0]] + [1] * (len(shape) - 1)
		return out.reshape(*reshape)

	def p_sample(self, x, mask, cur_y, t, target = None):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		eps_theta = self.model(cur_y, beta, x, mask, target)
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z
		return (sample)
	
	def p_sample_wcls_wuncond(self, x, pre_team1_label, 
			pre_team2_label, mask, cur_y, t):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()

		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		
		eps_theta_uncond = self.model(cur_y, beta, x, mask)
		eps_theta_cond = self.model.classifer_free_guidance(cur_y, beta, x, pre_team1_label, pre_team2_label, mask)
		
		eps_theta = (1 + self.w) * eps_theta_cond - self.w * eps_theta_uncond
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		
		z = torch.randn_like(cur_y).to(x.device)
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z
		return (sample)
	
	def p_sample_accelerate(self, x, mask, cur_y, t):
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		
		eps_theta = self.model.generate_accelerate(cur_y, beta, x, mask)
		
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		
		z = torch.randn_like(cur_y).to(x.device)
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z * 0.00001
		return (sample)

	def p_sample_accelerate_wcls(self, x, pre_team1_label, 
			pre_team2_label, mask, cur_y, t):
		# print(cur_y.shape)
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		# Factor to the model output
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		# Model output
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		# print('Diffusion denoising inf time:')
		start_time = time.time()
		# eps_theta_uncond = self.model.generate_accelerate(cur_y, beta, x, mask) # 110, 20, 2
		eps_theta_cond = self.model.generate_accelerate_wclsv1(cur_y, beta, x, pre_team1_label, pre_team2_label, mask)
		eps_theta = eps_theta_cond
		# eps_theta = (1 + self.w) * eps_theta_cond - self.w * eps_theta_uncond
		end_time = time.time()
		# print(end_time- start_time)
		
		# if target is not None:
		# 	eps_theta = eps_theta[:,:,:-1,:]
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		# Generate z
		z = torch.randn_like(cur_y).to(x.device)
		# Fixed sigma
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z * 0.00001
		return (sample)
	
	def p_sample_accelerate_wcls_wuncond(self, x, pre_team1_label, 
			pre_team2_label, mask, cur_y, t):
		# print(cur_y.shape)
		if t==0:
			z = torch.zeros_like(cur_y).to(x.device)
		else:
			z = torch.randn_like(cur_y).to(x.device)
		t = torch.tensor([t]).cuda()
		
		eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
		beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
		
		eps_theta_uncond = self.model.generate_accelerate(cur_y, beta, x, mask)
		eps_theta_cond = self.model.generate_accelerate_wclsv1(cur_y, beta, x, pre_team1_label, pre_team2_label, mask)
		eps_theta = (1 + self.w) * eps_theta_cond - self.w * eps_theta_uncond
		
		mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))
		
		z = torch.randn_like(cur_y).to(x.device)
		sigma_t = self.extract(self.betas, t, cur_y).sqrt()
		sample = mean + sigma_t * z * 0.00001
		return (sample)

	def p_sample_loop(self, x, mask, shape):
		self.model.eval()
		prediction_total = torch.Tensor().cuda()
		for _ in range(20):
			cur_y = torch.randn(shape).to(x.device)
			for i in reversed(range(self.n_steps)):
				cur_y = self.p_sample(x, mask, cur_y, i)
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total
	
	def p_sample_loop_mean(self, x, pre_team1_label, 
			pre_team2_label, mask, loc):
		prediction_total = torch.Tensor().cuda()
		for loc_i in range(1):
			cur_y = loc
			for i in reversed(range(NUM_Tau)):
				cur_y = self.p_sample_wcls_wuncond(x, pre_team1_label, 
			pre_team2_label, mask, cur_y, i) 
			prediction_total = torch.cat((prediction_total, cur_y.unsqueeze(1)), dim=1)
		return prediction_total

	def p_sample_loop_accelerate(self, x, mask, loc):
		'''
		Batch operation to accelerate the denoising process.

		x: [11, 10, 6]
		mask: [11, 11]
		cur_y: [11, 10, 20, 2]
		'''
		prediction_total = torch.Tensor().cuda()
		cur_y = loc[:, :10]
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate(x, mask, cur_y, i)
		cur_y_ = loc[:, 10:]
		for i in reversed(range(NUM_Tau)):
			cur_y_ = self.p_sample_accelerate(x, mask, cur_y_, i)
		prediction_total = torch.cat((cur_y_, cur_y), dim=1)
		return prediction_total
	

	def p_sample_loop_accelerate_wcls_wuncond(self, x, pre_team1_label, 
			pre_team2_label,  mask, loc):
		'''
		Batch operation to accelerate the denoising process.

		x: [11, 10, 6]
		mask: [11, 11]
		cur_y: [11, 10, 20, 2]
		'''
		prediction_total = torch.Tensor().cuda()
		cur_y = loc[:, :10]
	
		for i in reversed(range(NUM_Tau)):
			cur_y = self.p_sample_accelerate_wcls_wuncond(x, pre_team1_label, 
			pre_team2_label, mask, cur_y, i)
	
		cur_y_ = loc[:, 10:]
		for i in reversed(range(NUM_Tau)):
			cur_y_ = self.p_sample_accelerate_wcls_wuncond(x, pre_team1_label, 
			pre_team2_label, mask, cur_y_, i)
		#
		prediction_total = torch.cat((cur_y_, cur_y), dim=1)
		
		return prediction_total


	def data_preprocess_with_tactic(self, data):
		"""
			pre_motion_3D: torch.Size([32, 11, 10, 2]), [batch_size, num_agent, past_frame, dimension]
			fut_motion_3D: torch.Size([32, 11, 20, 2])
			fut_motion_mask: torch.Size([32, 11, 20])
			pre_motion_mask: torch.Size([32, 11, 10])
			traj_scale: 1
			pred_mask: None
			seq: nba
		"""
		batch_size, agent_num = data['pre_motion_3D'].shape[0], data['pre_motion_3D'].shape[1]
		past_len, fut_len = data['pre_motion_3D'].shape[2], data['fut_motion_3D'].shape[2]

		traj_mask = torch.zeros(batch_size*11, batch_size*11).cuda()
		for i in range(batch_size):
			traj_mask[i*11:(i+1)*11, i*11:(i+1)*11] = 1.

		initial_pos = data['pre_motion_3D'].cuda()[:, :, -1:]
		past_traj_abs = ((data['pre_motion_3D'].cuda() - self.traj_mean)/self.traj_scale).contiguous().view(-1, past_len, 2)
		past_traj_rel = ((data['pre_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, past_len, 2)
		past_traj_vel = torch.cat((past_traj_rel[:, 1:] - past_traj_rel[:, :-1], torch.zeros_like(past_traj_rel[:, -1:])), dim=1)
		past_traj = torch.cat((past_traj_abs, past_traj_rel, past_traj_vel), dim=-1) # #B*N, T_past, 2 * 3

		fut_traj = ((data['fut_motion_3D'].cuda() - initial_pos)/self.traj_scale).contiguous().view(-1, fut_len, 2)
		
		teamid, playerid = data['teamid'].cuda(), data['playerid'].cuda()
		tactic_labels = data['tactic_labels'].cuda()
		pre_team1_label, pre_team2_label, fut_team1_label, fut_team2_label = tactic_labels[:, 0], tactic_labels[:, 1], tactic_labels[:, 2], tactic_labels[:, 3]
		return batch_size, agent_num, traj_mask, past_traj, fut_traj, teamid, playerid, pre_team1_label, pre_team2_label, fut_team1_label, fut_team2_label

	def _train_single_epoch_wcls_wuncond(self, epoch):
		'''
		with tactic label e2e
		'''
		self.model.eval()
		self.model_initializer.train()
		self.cls_model.train()

		loss_total, loss_dt, loss_dc, count = 0, 0, 0, 0
		cls_game_loss_total, cls_loss_total, game_loss_total = 0, 0, 0
		
		accuracies = []
		for data in self.train_loader:
			(batch_size, agent_num, traj_mask, past_traj, fut_traj, teamid, playerid, pre_team1_label, 
			pre_team2_label, fut_team1_label, fut_team2_label) = self.data_preprocess_with_tactic(data)
			
			sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask) # guess_var, guess_mean, guess_scale
			sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
			loc = sample_prediction + mean_estimation[:, None]
			
			feat = self.model.feature_extract_wtt(past_traj, pre_team1_label, pre_team2_label, traj_mask)
			feat = feat.squeeze(1).reshape(batch_size, agent_num, -1)
			cls_prev, fteam_game_score, steam_game_score = self.cls_model.forward_ours(feat, self.model)

			classification_loss = FocalLoss(gamma=self.gamma)(
				cls_prev[0], fut_team1_label) + FocalLoss(gamma=self.gamma)(cls_prev[1], fut_team2_label) 
					
			game_loss = self.criterion(fteam_game_score[0], fteam_game_score[1]) + self.criterion(steam_game_score[0], steam_game_score[1])

			cls_game_loss = classification_loss + game_loss * self.cfg.game_factor
			cls_game_loss_total += cls_game_loss.item() * self.cfg.cls_rate
			cls_loss_total += classification_loss.item()
			game_loss_total += game_loss.item() * self.cfg.game_factor
			
			pred_labels1 = cls_prev[0].argmax(dim=1) 
			pred_labels2 = cls_prev[1].argmax(dim=1)
			accuracies.append(accuracy_score(fut_team1_label.cpu().numpy(), pred_labels1.cpu().numpy()))
			accuracies.append(accuracy_score(fut_team2_label.cpu().numpy(), pred_labels2.cpu().numpy()))
			
			generated_y = self.p_sample_loop_accelerate_wcls_wuncond(past_traj, pre_team1_label, 
			pre_team2_label, traj_mask, loc)

			
			loss_dist = (	(generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1) 
								* 
							self.temporal_reweight
						).mean(dim=-1).min(dim=1)[0].mean()
			loss_uncertainty = (torch.exp(-variance_estimation)
									*
								(generated_y - fut_traj.unsqueeze(dim=1)).norm(p=2, dim=-1).mean(dim=(1, 2)) 
									+ 
								variance_estimation
								).mean()
			
			loss =  loss_dist*50 + loss_uncertainty + cls_game_loss*self.cfg.cls_rate
			loss_total += loss.item()
			loss_dt += loss_dist.item()*50
			loss_dc += loss_uncertainty.item()

			self.opt.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(self.params, 1.)
			self.opt.step()
			count += 1
			if self.cfg.debug and count == 2:
				break
		
		avg_accuracy = sum(accuracies) / len(accuracies)

		return loss_total/count, loss_dt/count, loss_dc/count, cls_game_loss_total / count, cls_loss_total / count, game_loss_total / count, avg_accuracy

	def _test_single_epoch_wcls_wuncond(self):
		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0],
						'Rank1': [],
						'Rank2': [],
						'Rank3': [],
						'Rank5': [],
						}
		samples = 0
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)
		
		total_num = 0
		
		rank1_accs, rank2_accs, rank3_accs, rank5_accs,  = [], [], [], [],
		rank1_cnt, rank2_cnt, rank3_cnt, rank5_cnt = [], [], [], []
		
		all_preds, all_labels = [], []

		self.model.eval()
		self.model_initializer.eval()
		self.cls_model.eval()

		with torch.no_grad():
			for data in self.test_loader:
				(batch_size, agent_num, traj_mask, past_traj, fut_traj, teamid, playerid, pre_team1_label, 
			pre_team2_label, fut_team1_label, fut_team2_label) = self.data_preprocess_with_tactic(data)
				
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask) # guess_var, guess_mean, guess_scale
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
				
				### CLS
				feat = self.model.feature_extract_wtt(past_traj, pre_team1_label, pre_team2_label, traj_mask)
				feat = feat.squeeze(1).reshape(batch_size, agent_num, -1)
				cls_prev, _, _ = self.cls_model.forward_ours(feat, self.model)

				# metrics
				pred_labels1 = cls_prev[0].argmax(dim=1)
				pred_labels2 = cls_prev[1].argmax(dim=1)

				pred_scores1 = torch.softmax(cls_prev[0], dim=1)
				pred_scores2 = torch.softmax(cls_prev[1], dim=1)

				rank1, rank2, rank3, rank5 = rank_accuracy(pred_scores1, fut_team1_label, topk=(1, 2, 3, 5))
				rank1_cnt.append(batch_size * rank1)
				rank2_cnt.append(batch_size * rank2)
				rank3_cnt.append(batch_size * rank3)
				rank5_cnt.append(batch_size * rank5)

				rank1_accs.append(rank1)
				rank2_accs.append(rank2)
				rank3_accs.append(rank3)
				rank5_accs.append(rank5)
				
				rank1, rank2, rank3, rank5 = rank_accuracy(pred_scores2, fut_team2_label, topk=(1, 2, 3, 5))
				rank1_accs.append(rank1)
				rank2_accs.append(rank2)
				rank3_accs.append(rank3)
				rank5_accs.append(rank5)

				rank1_cnt.append(batch_size * rank1)
				rank2_cnt.append(batch_size * rank2)
				rank3_cnt.append(batch_size * rank3)
				rank5_cnt.append(batch_size * rank5)

				all_preds.append(pred_labels1.cpu().numpy())
				all_preds.append(pred_labels2.cpu().numpy())
				all_labels.append(fut_team1_label.cpu().numpy())
				all_labels.append(fut_team2_label.cpu().numpy())
			
				total_num += batch_size * 2

				
				pred_traj = self.p_sample_loop_accelerate_wcls_wuncond(past_traj, pre_team1_label, pre_team2_label,  traj_mask, loc)
				fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1)
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale ## B*N, K, T, 2
				# print(distances.shape)
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
					
				samples += distances.shape[0]
				

		batch_rank1 = torch.sum(torch.stack(rank1_cnt)) / total_num
		batch_rank2 = torch.sum(torch.stack(rank2_cnt)) / total_num
		batch_rank3 = torch.sum(torch.stack(rank3_cnt)) / total_num
		batch_rank5 = torch.sum(torch.stack(rank5_cnt)) / total_num

		performance['Rank1'].append(batch_rank1)
		performance['Rank2'].append(batch_rank2)
		performance['Rank3'].append(batch_rank3)
		performance['Rank5'].append(batch_rank5)

		return performance, samples


	def test(self):
		model_path = './results/models/best.p'
		model_dict = torch.load(model_path, map_location=torch.device('cpu'))['model_initializer_dict']
		self.model_initializer.load_state_dict(model_dict)

		performance = { 'FDE': [0, 0, 0, 0],
						'ADE': [0, 0, 0, 0],
						'Rank1': [],
						'Rank2': [],
						'Rank3': [],
						'Rank5': [],
						}
		samples = 0
		self.model.eval()
		self.model_initializer.eval()
		self.cls_model.eval()
		print_log(model_path, log=self.log)
		def prepare_seed(rand_seed):
			np.random.seed(rand_seed)
			random.seed(rand_seed)
			torch.manual_seed(rand_seed)
			torch.cuda.manual_seed_all(rand_seed)
		prepare_seed(0)

		print_log(' Config: \t {} \t'.format(self.cfg.__dict__), self.log)
		total_num = 0
		
		rank1_accs, rank2_accs, rank3_accs, rank5_accs = [], [], [], []
		rank1_cnt, rank2_cnt, rank3_cnt, rank5_cnt = [], [], [], []

		all_preds, all_labels = [], []

		with torch.no_grad():
			for data in self.test_loader:
				(batch_size, agent_num, traj_mask, past_traj, fut_traj, teamid, playerid, pre_team1_label, 
				pre_team2_label, fut_team1_label, fut_team2_label) = self.data_preprocess_with_tactic(data)
					
				sample_prediction, mean_estimation, variance_estimation = self.model_initializer(past_traj, traj_mask) # guess_var, guess_mean, guess_scale
				sample_prediction = torch.exp(variance_estimation/2)[..., None, None] * sample_prediction / sample_prediction.std(dim=1).mean(dim=(1, 2))[:, None, None, None]
				loc = sample_prediction + mean_estimation[:, None]
				
				### CLS
				feat = self.model.feature_extract_wtt(past_traj, pre_team1_label, pre_team2_label, traj_mask)
				feat = feat.squeeze(1).reshape(batch_size, agent_num, -1)
				cls_prev, _, _, _ = self.cls_model.forward_ours(feat, self.model)

				# metrics
				pred_labels1 = cls_prev[0].argmax(dim=1)
				pred_labels2 = cls_prev[1].argmax(dim=1)
				
				pred_scores1 = torch.softmax(cls_prev[0], dim=1)
				pred_scores2 = torch.softmax(cls_prev[1], dim=1)

				rank1, rank2, rank3, rank5 = rank_accuracy(pred_scores1, fut_team1_label, topk=(1, 2, 3, 5))
				rank1_cnt.append(batch_size * rank1)
				rank2_cnt.append(batch_size * rank2)
				rank3_cnt.append(batch_size * rank3)
				rank5_cnt.append(batch_size * rank5)

				rank1_accs.append(rank1)
				rank2_accs.append(rank2)
				rank3_accs.append(rank3)
				rank5_accs.append(rank5)
				
				rank1, rank2, rank3, rank5 = rank_accuracy(pred_scores2, fut_team2_label, topk=(1, 2, 3, 5))
				rank1_accs.append(rank1)
				rank2_accs.append(rank2)
				rank3_accs.append(rank3)
				rank5_accs.append(rank5)

				rank1_cnt.append(batch_size * rank1)
				rank2_cnt.append(batch_size * rank2)
				rank3_cnt.append(batch_size * rank3)
				rank5_cnt.append(batch_size * rank5)

				all_preds.append(pred_labels1.cpu().numpy())
				all_preds.append(pred_labels2.cpu().numpy())
				all_labels.append(fut_team1_label.cpu().numpy())
				all_labels.append(fut_team2_label.cpu().numpy())
				total_num += batch_size * 2

				pred_traj = self.p_sample_loop_accelerate_wcls_wuncond(past_traj, pre_team1_label, pre_team2_label, traj_mask, loc)
				fut_traj = fut_traj.unsqueeze(1).repeat(1, 20, 1, 1) # B*N, K, T, 2
				distances = torch.norm(fut_traj - pred_traj, dim=-1) * self.traj_scale
				for time_i in range(1, 5):
					ade = (distances[:, :, :5*time_i]).mean(dim=-1).min(dim=-1)[0].sum()
					fde = (distances[:, :, 5*time_i-1]).min(dim=-1)[0].sum()
					performance['ADE'][time_i-1] += ade.item()
					performance['FDE'][time_i-1] += fde.item()
				samples += distances.shape[0]

		batch_rank1 = torch.sum(torch.stack(rank1_cnt)) / total_num
		batch_rank2 = torch.sum(torch.stack(rank2_cnt)) / total_num
		batch_rank3 = torch.sum(torch.stack(rank3_cnt)) / total_num
		batch_rank5 = torch.sum(torch.stack(rank5_cnt)) / total_num

		performance['Rank1'].append(batch_rank1)
		performance['Rank2'].append(batch_rank2)
		performance['Rank3'].append(batch_rank3)
		performance['Rank5'].append(batch_rank5)

		
		for time_i in range(4):
			print_log('--ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format( time_i+1, performance['ADE'][time_i]/samples, \
				time_i+1, performance['FDE'][time_i]/samples), log=self.log)
		print_log('CLS: --Rank1: {:.4f}\t--Rank2: {:.4f}\t--Rank3: {:.4f}\t--Rank5: {:.4f}'.format(
					performance['Rank1'][0], performance['Rank2'][0], performance['Rank3'][0], performance['Rank5'][0]), self.log)

	
	def fit(self):
		""" Loading if needed """
		if self.cfg.epoch_continue > 0:
			checkpoint_path = os.path.join(self.cfg.model_save_dir,'model_00'+str(self.cfg.epoch_continue)+'.p')
			print(f'load model from: {checkpoint_path}')
			model_load = torch.load(checkpoint_path, map_location='cpu')
			self.model_initializer.load_state_dict(model_load['model_initializer_dict'])
			self.cls_model.load_state_dict(model_load['model_cls_dict'])
			if 'optimizer' in model_load:
				self.opt.load_state_dict(model_load['optimizer'])
				self.cls_opt.load_state_dict(model_load['cls_optimizer'])
			if 'scheduler' in model_load:
				self.scheduler_model.load_state_dict(model_load['scheduler'])
				self.cls_scheduler_model.load_state_dict(model_load['cls_scheduler_model'])
		
		print_log(' Config: \t {} \t'.format(self.cfg.__dict__), self.log)
		model_str = str(self.model)
		print_log(' Model: \t {} \t'.format(model_str), self.log)
		
		# Training loop
		for epoch in range(self.cfg.epoch_continue, self.cfg.num_epochs):
			loss_total, loss_distance, loss_uncertainty, loss_cls_game, loss_cls, loss_game, acc = self._train_single_epoch_wcls_wuncond(epoch)
			print_log('[{}] Epoch: {}\t\t TrajPred: Loss: {:.6f}\tLoss Dist.: {:.6f}\tLoss Uncertainty: {:.6f} CLS: CE_game Loss: {:.6f}\tCE Loss: {:.6f}\tgame Loss: {:.6f}\tACC: {:.6f}\t'.format(
				time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), 
				epoch, loss_total, loss_distance, loss_uncertainty, loss_cls_game, loss_cls, loss_game, acc), self.log)
			
			if (epoch + 1) % self.cfg.test_interval == 0:
				cp_path = self.cfg.model_path % (epoch + 1)
				model_cp = {'model_initializer_dict': self.model_initializer.state_dict(), 'model_cls_dict': self.cls_model.state_dict(),
					'optimizer': self.opt.state_dict(),'scheduler' : self.scheduler_model.state_dict()}
				torch.save(model_cp, cp_path)
				performance, samples = self._test_single_epoch_wcls_wuncond()
				for time_i in range(4):
					print_log('TrajPred: --ADE({}s): {:.4f}\t--FDE({}s): {:.4f}'.format(
						time_i+1, performance['ADE'][time_i]/samples,
						time_i+1, performance['FDE'][time_i]/samples), self.log)
				print_log('CLS: --F1: {:.4f}\t--Rank1: {:.4f}\t--Rank2: {:.4f}\t--Rank3: {:.4f}\t--Rank5: {:.4f}'.format(
						performance['F1'][0],  performance['Rank1'][0], performance['Rank2'][0], performance['Rank3'][0], performance['Rank5'][0]), self.log)
	
			self.scheduler_model.step()
