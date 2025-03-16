import json
import torch
from torch.utils.data import Dataset

class NBADataset(Dataset):
    def __init__(self, json_file, obs_len=5, pred_len=10):
        """
        load from JSON and init the dataset

        Args:
            json_file (str): JSON file path
            obs_len (int): observed sequence len
            pred_len (int): predict sequence len
        """
        super(NBADataset, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.trajs = self.data["trajs"]

        self.trajs[:,:,:,:2] = self.trajs[:,:,:,:2] / (94/28)
        
        self.batch_len = len(self.trajs)

    def __len__(self):
        return self.batch_len


        

    def __getitem__(self, index):
        traj_info = self.trajs[index]

        traj_data = torch.tensor(traj_info["traj_data"], dtype=torch.float32)
        tactic_labels = torch.tensor(traj_info["tactic_labels"], dtype=torch.int)
        team_id = torch.tensor(traj_info["team_id"], dtype=torch.int)
        player_id = torch.tensor(traj_info["player_id"], dtype=torch.int)

        past_traj = traj_data[:, :self.obs_len, :2]
        fut_traj = traj_data[:, self.obs_len:, :2]

        past_mask = torch.ones(11, self.obs_len)
        fut_mask = torch.ones(11, self.pred_len)

        out = [
            past_traj, fut_traj,
            past_mask, fut_mask,
            team_id, player_id,
            tactic_labels
        ]

        return out