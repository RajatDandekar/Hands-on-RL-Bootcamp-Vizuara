import gymnasium as gym
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo

# Create training environment
env = gym.make("LunarLander-v3")
writer = SummaryWriter(log_dir="runs/lunarlander_mc")

# Discretization bins for 8-dimensional state
NUM_BINS = (8, 8, 8, 8, 8, 8, 2, 2)
TOTAL_STATES = np.prod(NUM_BINS)

# Pre-defined observation bounds
obs_space_low = np.array([-1.5, -1.5, -1.5, -2.0, -math.pi, -5.0, 0.0, 0.0])
obs_space_high = np.array([1.5, 1.5, 1.5, 2.0, math.pi, 5.0, 1.0, 1.0])
obs_range = obs_space_high - obs_space_low

# Discretize continuous state
def discretize(obs):
    ratios = np.clip((obs - obs_space_low) / obs_range, 0, 0.999)
    bins = (ratios * NUM_BINS).astype(int)
    
    # Convert to single index
    multipliers = np.array([NUM_BINS[1] * NUM_BINS[2] * NUM_BINS[3] * NUM_BINS[4] * NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[2] * NUM_BINS[3] * NUM_BINS[4] * NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[3] * NUM_BINS[4] * NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[4] * NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[7],
                           1])
    return np.dot(bins, multipliers)

# Initialize Q-table and visit counts for Monte Carlo
Q = np.zeros((TOTAL_STATES, env.action_space.n))
returns_sum = np.zeros((TOTAL_STATES, env.action_space.n))
returns_count = np.zeros((TOTAL_STATES, env.action_space.n))

# Hyperparameters
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
NUM_EPISODES = 30000

# Reward tracking
all_rewards = []

# Epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    return np.argmax(Q[state])

print("Starting LunarLander Monte Carlo training...")
print(f"State space size: {TOTAL_STATES:,}")
print(f"Total episodes to train: {NUM_EPISODES:,}")
print("=" * 50)

# ---------------------------
# Monte Carlo Training
# ---------------------------
for episode in range(NUM_EPISODES):
    # Generate episode
    episode_states = []
    episode_actions = []
    episode_rewards = []
    
    state, _ = env.reset()
    state = discretize(state)
    done = False
    steps = 0

    # Collect episode data
    while not done and steps < 1000:
        action = choose_action(state, EPSILON)
        next_state, reward, done, truncated, _ = env.step(action)
        
        episode_states.append(state)
        episode_actions.append(action)
        episode_rewards.append(reward)
        
        state = discretize(next_state)
        steps += 1

    # Calculate returns for this episode
    episode_length = len(episode_rewards)
    total_reward = sum(episode_rewards)
    all_rewards.append(total_reward)
    
    # Calculate discounted returns (working backwards)
    G = 0
    for t in range(episode_length - 1, -1, -1):
        G = GAMMA * G + episode_rewards[t]
        
        state_t = episode_states[t]
        action_t = episode_actions[t]
        
        
        returns_sum[state_t, action_t] += G
        returns_count[state_t, action_t] += 1
            
        # Update Q-value with average return
        Q[state_t, action_t] = returns_sum[state_t, action_t] / returns_count[state_t, action_t]

    # Epsilon decay
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

    # Logging
    if episode % 50 == 0:
        writer.add_scalar("Train/EpisodeReward", total_reward, episode)
        if episode >= 100:
            avg_reward = np.mean(all_rewards[-100:])
            writer.add_scalar("Train/MovingAverageReward", avg_reward, episode)
            writer.add_scalar("Train/Epsilon", EPSILON, episode)

    # Simple progress tracking
    if (episode + 1) % 1000 == 0:
        recent_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
        states_visited = np.sum(returns_count > 0)
        print(f"Episode {episode + 1:5d}/{NUM_EPISODES} - Avg Reward: {recent_avg:.1f} - States Visited: {states_visited}")

env.close()

print("Training completed! Starting evaluation...")

# ---------------------------
# Testing Phase
# ---------------------------
test_env = gym.make("LunarLander-v3", render_mode="rgb_array")
test_env = RecordVideo(
    test_env,
    video_folder="lunarlander_videos_mc",
    episode_trigger=lambda ep_id: ep_id < 5,
    name_prefix="mc_lander"
)

total_test_rewards = []
successful_landings = 0

for test_episode in range(50):
    state, _ = test_env.reset()
    state = discretize(state)
    done = False
    test_reward = 0
    steps = 0

    while not done and steps < 1000:
        action = np.argmax(Q[state])  # Pure greedy policy
        next_state, reward, done, truncated, _ = test_env.step(action)
        test_reward += reward
        state = discretize(next_state)
        steps += 1

    total_test_rewards.append(test_reward)
    if test_reward >= 200:
        successful_landings += 1
    
    if test_episode % 10 == 0:
        writer.add_scalar("Test/EpisodeReward", test_reward, test_episode)

# Calculate final statistics
avg_test_reward = np.mean(total_test_rewards)
success_rate = successful_landings / len(total_test_rewards) * 100
final_training_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
states_visited = np.sum(returns_count > 0)

writer.add_scalar("Test/AverageReward", avg_test_reward, 0)
writer.add_scalar("Test/SuccessRate", success_rate, 0)
writer.add_scalar("Final/TrainingAverage", final_training_avg, 0)
writer.add_scalar("Final/StatesVisited", states_visited, 0)

print(f"\n=== MONTE CARLO RESULTS ===")
print(f"Training Episodes: {NUM_EPISODES:,}")
print(f"Final Training Average: {final_training_avg:.2f}")
print(f"States Visited: {states_visited:,} / {TOTAL_STATES:,}")
print(f"Average Test Reward: {avg_test_reward:.2f}")
print(f"Successful Landings: {successful_landings}/{len(total_test_rewards)} ({success_rate:.1f}%)")
print(f"Best Test Episode: {max(total_test_rewards):.2f}")

test_env.close()
writer.close()

print(f"\nTensorBoard logs: runs1/lunarlander_mc")
print(f"Videos saved to: lunarlander_videos_mc/")