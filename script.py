import sys 
import os 
import gc
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
import wandb
from omegaconf import OmegaConf
import matplotlib.pyplot as plt 
import mujoco
import gymnasium as gym 
import numpy as np
from core import DecisionTransformer
import matplotlib.pyplot as plt
from tqdm import tqdm
import cProfile, pstats


def train():
    run = wandb.init()
    wandb_config = wandb.config

    seed = wandb_config.seed
    batch_size = wandb_config.batch_size 
    env_name = str(wandb_config.env_name) 
    dataset_id = str(wandb_config.env_id)
    lr = float(wandb_config.lr)
    n_epochs = wandb_config.training_epochs
    eval_epochs = wandb_config.eval_epochs
    target_return = wandb_config.target_return
    sampled_ep = int(wandb_config.sampled_eps)
    
    env = gym.make(env_name)

    state_n = env.observation_space.shape[0]
    action_n = env.action_space.shape[0]

    params = {
        'block_size' : int(wandb_config.block_size),
        'n_embed':     int(wandb_config.n_embed), 
        'state_n':     state_n, 
        'action_n':    action_n
    } 

    dt = DecisionTransformer(params, batch_size, sampled_ep, lr, dataset_id)
    ev = 0
    for i in tqdm(range(0, n_epochs)):
        dt.learn()

        if i%eval_epochs == 0:
            re = dt.eval(env_name, 20, target_return, seed)
            run.log({"episode_reward": re}, step=ev)
            ev+=1

    dt = None
    gc.collect()


conf = OmegaConf.load('config.yml')
conf = OmegaConf.to_container(conf)
sweep_id = wandb.sweep(sweep=conf, project="dt_analysis")
wandb.agent(sweep_id, function=train)









