import gymnasium as gym
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import RecordVideo

# Create training environment
env = gym.make("LunarLander-v3")
writer = SummaryWriter(log_dir="runs/lunarlander_td_optimized")

# Discretization bins for 8-dimensional state
NUM_BINS = (8, 8, 8, 8, 8, 8, 2, 2)
TOTAL_STATES = np.prod(NUM_BINS)

# Pre-defined observation bounds for faster computation
obs_space_low = np.array([-1.5, -1.5, -1.5, -2.0, -math.pi, -5.0, 0.0, 0.0])
obs_space_high = np.array([1.5, 1.5, 1.5, 2.0, math.pi, 5.0, 1.0, 1.0])
obs_range = obs_space_high - obs_space_low

# Optimized discretization with single index conversion
def discretize(obs):
    # Vectorized clipping and binning
    ratios = np.clip((obs - obs_space_low) / obs_range, 0, 0.999)
    bins = (ratios * NUM_BINS).astype(int)
    
    # Convert multi-dimensional index to single index for faster lookup
    multipliers = np.array([NUM_BINS[1] * NUM_BINS[2] * NUM_BINS[3] * NUM_BINS[4] * NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[2] * NUM_BINS[3] * NUM_BINS[4] * NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[3] * NUM_BINS[4] * NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[4] * NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[5] * NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[6] * NUM_BINS[7],
                           NUM_BINS[7],
                           1])
    return np.dot(bins, multipliers)

# Initialize Q-table as numpy array for faster access
Q = np.zeros((TOTAL_STATES, env.action_space.n))

# Hyperparameters
GAMMA = 0.99
ALPHA = 0.1
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.9995
NUM_EPISODES = 15000

# Pre-allocate arrays for efficient reward tracking
reward_window = np.zeros(100)
window_idx = 0
all_rewards = []

# Optimized epsilon-greedy policy
def choose_action(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(env.action_space.n)
    return np.argmax(Q[state])

print("Starting LunarLander Q-Learning training...")
print(f"State space size: {TOTAL_STATES:,}")
print(f"Total episodes to train: {NUM_EPISODES:,}")
print("=" * 50)

# ---------------------------
# Training with TD(0) Q-learning
# ---------------------------
for episode in range(NUM_EPISODES):
    state, _ = env.reset()
    state = discretize(state)
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 1000:  # Add step limit to prevent infinite episodes
        action = choose_action(state, EPSILON)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state_discrete = discretize(next_state)

        # Vectorized Q-learning update
        if not (done or truncated):
            td_target = reward + GAMMA * np.max(Q[next_state_discrete])
        else:
            td_target = reward
            
        Q[state, action] += ALPHA * (td_target - Q[state, action])

        total_reward += reward
        state = next_state_discrete
        steps += 1

    # Efficient epsilon decay
    EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)
    
    # Efficient reward tracking
    all_rewards.append(total_reward)
    reward_window[window_idx] = total_reward
    window_idx = (window_idx + 1) % 100

    # Reduced logging frequency for speed
    if episode % 50 == 0:  # Log every 50 episodes instead of 10
        writer.add_scalar("Train/EpisodeReward", total_reward, episode)
        if episode >= 100:
            avg_reward = np.mean(reward_window)
            writer.add_scalar("Train/MovingAverageReward", avg_reward, episode)
            writer.add_scalar("Train/Epsilon", EPSILON, episode)

    # Simple progress tracking
    if (episode + 1) % 1000 == 0:
        recent_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
        print(f"Episode {episode + 1:5d}/{NUM_EPISODES} - Avg Reward: {recent_avg:.1f}")

env.close()

print("Training completed! Starting evaluation...")

# ---------------------------
# Testing Phase + Video Recording
# ---------------------------
test_env = gym.make("LunarLander-v3", render_mode="rgb_array")
test_env = RecordVideo(
    test_env,
    video_folder="lunarlander_videos_td",
    episode_trigger=lambda ep_id: ep_id < 5,  # Reduced video count for speed
    name_prefix="td_lander_optimized"
)

total_test_rewards = []
successful_landings = 0

# Reduced test episodes for faster evaluation
for test_episode in range(50):  # Reduced from 100 to 50
    state, _ = test_env.reset()
    state = discretize(state)
    done = False
    test_reward = 0
    steps = 0

    while not done and steps < 1000:
        action = np.argmax(Q[state])  # Pure greedy policy for testing
        next_state, reward, done, truncated, _ = test_env.step(action)
        test_reward += reward
        state = discretize(next_state)
        steps += 1

    total_test_rewards.append(test_reward)
    if test_reward >= 200:  # Consider landing successful if reward >= 200
        successful_landings += 1
    
    # Log test results less frequently
    if test_episode % 10 == 0:
        writer.add_scalar("Test/EpisodeReward", test_reward, test_episode)

# Calculate and log final statistics
avg_test_reward = np.mean(total_test_rewards)
success_rate = successful_landings / len(total_test_rewards) * 100
final_training_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)

writer.add_scalar("Test/AverageReward", avg_test_reward, 0)
writer.add_scalar("Test/SuccessRate", success_rate, 0)
writer.add_scalar("Final/TrainingAverage", final_training_avg, 0)

print(f"\n=== RESULTS ===")
print(f"Training Episodes: {NUM_EPISODES:,}")
print(f"Final Training Average (last 100 episodes): {final_training_avg:.2f}")
print(f"Test Episodes: {len(total_test_rewards)}")
print(f"Average Test Reward: {avg_test_reward:.2f}")
print(f"Successful Landings: {successful_landings}/{len(total_test_rewards)} ({success_rate:.1f}%)")
print(f"Best Test Episode: {max(total_test_rewards):.2f}")

test_env.close()
writer.close()

print(f"\nTensorBoard logs saved to: runs/lunarlander_td_optimized")
print(f"Videos saved to: lunarlander_videos_td/")