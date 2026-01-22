# Step 1: Adding imports
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict

matplotlib.use("TkAgg")

def plot_policy_with_q_inspector(Q, env):
    desc = env.unwrapped.desc
    n = desc.shape[0]

    rng = np.random.default_rng(0)
    policy = np.array([
        greedy_action(Q[s, :], rng) for s in range(Q.shape[0])
    ]).reshape(n, n)

    fig, ax = plt.subplots(figsize=(6, 6))

    color_map = {
        b'S': '#A7C7E7',
        b'F': '#FFFFFF',
        b'H': '#FF6961',
        b'G': '#77DD77',
    }

    # nacrtaj mapu
    for i in range(n):
        for j in range(n):
            cell = desc[i, j]
            ax.add_patch(
                plt.Rectangle((j, i), 1, 1, facecolor=color_map[cell])
            )
            if cell in [b'S', b'H', b'G']:
                ax.text(j + 0.5, i + 0.5, cell.decode(),
                        ha='center', va='center', fontsize=12, weight='bold')

    # strelice politike
    action_to_vec = {
        0: (-0.3, 0),
        1: (0, 0.3),
        2: (0.3, 0),
        3: (0, -0.3),
    }

    for i in range(n):
        for j in range(n):
            if desc[i, j] in [b'H', b'G']:
                continue
            a = policy[i, j]
            dx, dy = action_to_vec[a]
            ax.arrow(j + 0.5, i + 0.5, dx, dy,
                     head_width=0.15, head_length=0.15,
                     fc='black', ec='black')

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid()
    ax.set_title("Click a cell to inspect Q(s, a)")

    # === INTERAKCIJA ===
    action_names = ["LEFT", "DOWN", "RIGHT", "UP"]

    def on_click(event):
        if event.inaxes != ax:
            return

        j = int(event.xdata)
        i = int(event.ydata)

        if not (0 <= i < n and 0 <= j < n):
            return

        s = i * n + j
        q_vals = Q[s]

        print("\n==========================")
        print(f"State {s}  (row={i}, col={j})")
        for a, q in enumerate(q_vals):
            mark = " ← greedy" if q == np.max(q_vals) else ""
            print(f"{action_names[a]:5s}: {q: .4f}{mark}")
        print("==========================")

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show(block=True)
def greedy_action(Q_row, rng):
    max_q = np.max(Q_row)
    actions = np.flatnonzero(Q_row == max_q)
    return rng.choice(actions)
def random_start_state(env, rng):
    desc = env.unwrapped.desc.flatten()  # mapa
    valid_states = [
        i for i, c in enumerate(desc)
        if c in [b'S', b'F']   # Start ili Frozen
    ]
    return rng.choice(valid_states)
def plot_policy_on_map(Q, env):
    desc = env.unwrapped.desc
    n = desc.shape[0]

    rng = np.random.default_rng(0)  # fiksno seme za lep prikaz
    policy = np.array([
        greedy_action(Q[s, :], rng) for s in range(Q.shape[0])
    ]).reshape(n, n)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Boje za polja
    color_map = {
        b'S': '#A7C7E7',  # start - plavo
        b'F': '#FFFFFF',  # frozen - belo
        b'H': '#FF6961',  # hole - crveno
        b'G': '#77DD77',  # goal - zeleno
    }

    # Crtanje polja
    for i in range(n):
        for j in range(n):
            cell = desc[i, j]
            rect = plt.Rectangle((j, i), 1, 1, facecolor=color_map[cell])
            ax.add_patch(rect)

            # oznaka S/H/G
            if cell in [b'S', b'H', b'G']:
                ax.text(j + 0.5, i + 0.5, cell.decode(),
                        ha='center', va='center', fontsize=12, weight='bold')

    # Mapiranje akcija → pomeraj
    action_to_vec = {
        0: (-0.3, 0),   # LEFT
        1: (0, 0.3),    # DOWN
        2: (0.3, 0),    # RIGHT
        3: (0, -0.3),   # UP
    }

    # Strelice
    for i in range(n):
        for j in range(n):
            if desc[i, j] in [b'H', b'G']:
                continue

            a = policy[i, j]
            dx, dy = action_to_vec[a]

            ax.arrow(j + 0.5, i + 0.5, dx, dy,
                     head_width=0.15, head_length=0.15,
                     fc='black', ec='black')

    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.grid()
    ax.set_title("Learned policy on FrozenLake map")

    plt.show()



def simulate_episode(Q,  env):
    rng = np.random.default_rng()

    #env.reset()
    #state = random_start_state(env, rng)
    #env.unwrapped.s = state
    state, _ = env.reset()
    start_state = state
    terminated = truncated = False
    steps = 0
    total_reward = 0

    while not terminated and not truncated:
        action = greedy_action(Q[state, :], rng)

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

    env.close()
    return total_reward, steps, start_state
##############################################################################################################
#
#
#
#
#
#
##############################################################################################################

def start(episodes, render=False, slippery=True, map=[]):
    # Step 2: Creating the environment
    env = gym.make(
        'FrozenLake-v1',
        desc=None,
        map_name='8x8',
        is_slippery=slippery,
        render_mode='human' if render else None,
        success_rate= 8.0 / 10.0,
        reward_schedule=(1, 0, 0) # Moze da se doda i kazna za korake i upade u rupe ali po originalnoj implementaciji je ovakav reward schedule
    )

    # Step 3: Q table init
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    N = np.zeros_like(Q)

    # Step 4: Hyperparameters
    gamma = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()
    returns = defaultdict(list)
    episode_rewards = []

    # Step 4: Ucenje
    for i in range(episodes):
        state , _ = env.reset()
        state = random_start_state(env, rng)
        env.unwrapped.s = state

        terminated = False  # True when fall in hole or reached goal
        truncated = False  # True when max actions gets surpassed ( msl da je 200 )
        total_reward = 0

        episode = []

        # Generisanje epizode
        while not terminated and not truncated:
            if rng.random() < epsilon:
                action = env.action_space.sample()  # actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = greedy_action(Q[state, :], rng)


            new_state, reward, terminated, truncated, _ = env.step(action)

            if reward == 1:
                print(f"Goal Reached in Episode {i}, Yay!")

            total_reward += reward
            episode.append((state, action, reward))
            state = new_state

        # Kraj epizode
        G = 0
        episode_rewards.append(total_reward)

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r
            returns[(s, a)].append(G)
            Q[s, a] = np.mean(returns[(s, a)])

        epsilon = max(epsilon - epsilon_decay_rate, 0.0001)


    # Step 5: Plotovanje ucenja i politike
    episode_rewards = np.array(episode_rewards)
    rolling = np.convolve(
        episode_rewards,
        np.ones(500) / 500,
        mode='valid'
    )

    plt.figure(figsize=(10, 5))
    plt.plot(rolling)
    plt.xlabel("Episode")
    plt.ylabel("Average reward")
    plt.title("Monte Carlo training progress")
    plt.grid()
    plt.show()

    # Step 6: Print policy

    plot_policy_with_q_inspector(Q, env)

    # Step 7: Simulate Learned Q values
    for i in range(5):
        reward, steps, start_state = simulate_episode(Q, env)
        print(f"Episode {i}: reward={reward}, steps={steps}, start_state={start_state}")

# Optimalno sa 100000 epizoda
start(50000, render=False, slippery=True)

