"""
 this is a script that doesnt contain all evaluations for Agent 2 it was for only one model checkpoint so ignore it. 
evaluation script for Agent 2: joint convention + goal belief agent 

Evaluate a new guided checkpoint or perform a post-hoc guide intervention
on an existing joint_belief_v1 checkpoint. No privileged goal inputs.

Existing baseline, post-hoc guided intervention:
python -m evaluation.evaluate_goal_guided --checkpoint <old_joint_checkpoint> --guide-strength 3.0

New guided checkpoint:
python -m evaluation.evaluate_goal_guided --checkpoint results/goal_guided/experiment2/guided_seed_20_steps100000.pt
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from agents.goal_guided_actor import GoalGuidedJointActor
from configs.experiments import make_env


def evaluate(checkpoint, episodes=1000, seed_start=10000, belief_mode='real', device='cpu',
             guide_strength=None, output=None):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    experiment = saved['experiment']
    architecture = saved.get('architecture')
    if architecture not in ('joint_belief_v1','joint_goal_guided_v1'):
        raise ValueError(f'Unsupported checkpoint architecture: {architecture}')
    scale = float(guide_strength) if guide_strength is not None else float(saved.get('guide_strength',0.))
    model = GoalGuidedJointActor(num_conventions=saved['num_conventions'], guide_strength=scale).to(device)
    model.load_state_dict(saved['actor'], strict=(architecture == 'joint_goal_guided_v1'))
    model.eval()
    env = make_env(experiment)
    records = []
    try:
        with torch.no_grad():
            for seed in range(seed_start, seed_start+episodes):
                obs, info = env.reset(seed=seed)
                convention, goal = int(obs['convention'][0]), int(info['learner_goal_id'])
                memory = model.initial_memory(1, device)
                done = truncated = False
                steps = 0; reward_sum = 0.; c_ok = g_ok = 0
                while not (done or truncated):
                    x = torch.as_tensor(np.asarray(obs['learner'], dtype=np.float32),device=device).unsqueeze(0)
                    action, _, _, memory = model.get_action(x,memory,deterministic=True,belief_mode=belief_mode)
                    c_ok = int(memory[0].argmax().item()==convention)
                    g_ok = int(memory[1].argmax().item()==goal)
                    obs,reward,done,truncated,_ = env.step(int(action.item()))
                    reward_sum += float(reward); steps += 1
                records.append({'seed':seed,'success':int(done),'reward':reward_sum,'length':steps,
                                'final_convention_id':c_ok,'final_goal_id':g_ok})
    finally:
        env.close()
    result = {
        'experiment': experiment,'checkpoint':str(checkpoint), 'architecture':architecture,
        'guide_strength':scale,'belief_mode':belief_mode,'episodes':episodes,
        'success_pct':100*np.mean([r['success'] for r in records]),
        'mean_reward':np.mean([r['reward'] for r in records]),
        'final_convention_id_pct':100*np.mean([r['final_convention_id'] for r in records]),
        'final_goal_id_pct':100*np.mean([r['final_goal_id'] for r in records]),
    }
    if output:
        path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--episodes',type=int,default=1000)
    p.add_argument('--seed-start',type=int,default=10000)
    p.add_argument('--belief-mode',choices=['real','uniform','uniform_goal','uniform_convention'],default='real')
    p.add_argument('--guide-strength',type=float,default=None)
    p.add_argument('--device',default='cpu')
    p.add_argument('--output',default=None)
    a=p.parse_args()
    if a.episodes<=0:p.error('episodes must be positive')
    for k,v in evaluate(**vars(a)).items():print(f'{k}: {v}')

if __name__=='__main__':main()
