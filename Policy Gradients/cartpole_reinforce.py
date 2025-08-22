#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import typing as tt
from torch.utils.tensorboard.writer import SummaryWriter
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# --- Hyperparameters ---
GAMMA = 0.99
LEARNING_RATE = 0.01
EPISODES_TO_TRAIN = 4 # The number of episodes to collect before a training step

class PGN(nn.Module):
    """
    Policy Gradient Network
    """
    def __init__(self, input_size: int, n_actions: int):
        super(PGN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def calc_qvals(rewards: tt.List[float]) -> tt.List[float]:
    """
    Calculates discounted rewards for an episode.
    Args:
        rewards: A list of rewards obtained during one episode.
    Returns:
        A list of discounted total rewards (Q-values).
    """
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    # The result is now in reverse order, so we reverse it back
    return list(reversed(res))


if __name__ == "__main__":
    # --- Initialization ---
    env = gym.make("CartPole-v1")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # --- Training Loop ---
    total_rewards = []
    step_idx = 0
    
    batch_episodes = 0
    batch_states, batch_actions, batch_qvals = [], [], []
    
    # Use itertools.count for an infinite episode loop
    for episode_idx in count():
        # --- Episode Initialization ---
        state, _ = env.reset()
        episode_rewards = []
        
        while True:
            # --- Action Selection ---
            # Convert state to a tensor and add a batch dimension
            state_t = torch.as_tensor(state, dtype=torch.float32)
            
            # Get action probabilities from the network
            logits_t = net(state_t)
            probs_t = F.softmax(logits_t, dim=0)
            
            # Sample an action from the probability distribution
            # .item() extracts the value from the tensor
            action = torch.multinomial(probs_t, num_samples=1).item()

            # --- Environment Step ---
            next_state, reward, done, truncated, _ = env.step(action)

            # --- Store Experience ---
            # We store the state, action, and reward for the current step
            batch_states.append(state)
            batch_actions.append(action)
            episode_rewards.append(reward)

            step_idx += 1
            state = next_state

            if done or truncated:
                # --- Episode End ---
                # Calculate discounted rewards for the completed episode
                batch_qvals.extend(calc_qvals(episode_rewards))
                
                # Update tracking variables
                batch_episodes += 1
                total_rewards.append(sum(episode_rewards))
                mean_rewards = float(np.mean(total_rewards[-100:]))
                
                # --- Logging ---
                print(f"{step_idx}: Episode {episode_idx}, Reward: {sum(episode_rewards):6.2f}, Mean-100: {mean_rewards:6.2f}")
                writer.add_scalar("reward", sum(episode_rewards), step_idx)
                writer.add_scalar("reward_100", mean_rewards, step_idx)
                writer.add_scalar("episodes", episode_idx, step_idx)

                if mean_rewards > 450:
                    print(f"Solved in {step_idx} steps and {episode_idx} episodes!")
                    break # Break from the inner while loop
                
                break # Break from the inner while loop
        
        if mean_rewards > 450:
            break # Break from the outer for loop

        # --- Training Step ---
        if batch_episodes < EPISODES_TO_TRAIN:
            continue

        # Convert collected batch data to tensors
        states_t = torch.as_tensor(np.asarray(batch_states), dtype=torch.float32)
        actions_t = torch.as_tensor(np.asarray(batch_actions), dtype=torch.int64)
        qvals_t = torch.as_tensor(np.asarray(batch_qvals), dtype=torch.float32)

        # --- Log the variance of the Q-values ---
        writer.add_scalar("qvals_variance", qvals_t.var().item(), step_idx)

        # --- Loss Calculation and Optimization ---
        optimizer.zero_grad()
        
        logits_t = net(states_t)
        log_prob_t = F.log_softmax(logits_t, dim=1)
        
        # Get the log probabilities for the actions that were actually taken
        log_prob_actions_t = log_prob_t[range(len(batch_states)), actions_t]
        
        # Calculate the policy loss: - (log_prob * Q-value)
        # We want to maximize (log_prob * Q-value), which is equivalent to
        # minimizing its negative.
        loss_t = -(qvals_t * log_prob_actions_t).sum()

        loss_t.backward()
        optimizer.step()

        # --- Clear Batch Data for Next Training Step ---
        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()
