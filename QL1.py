import gym
import numpy as np
import matplotlib.pyplot as plt
# MountainCar-v0 is string reference value from gym lib
env = gym.make("MountainCar-v0")


LEARNING_RATE = 0.1
DISCOUNT = 0.95  # a measure of how how much we value future reward over current reward. And it's supposed to be between 0 and 1
EPISODES = 2000  # number of iteration
# print(env.observation_space.high)
# print(env.observation_space.low)
# print(env.action_space.n)

SHOW_EVERY = 500
DISCRETE_OS_SIZE = [20] * \
    len(env.observation_space.high)  # [20] * 2 == [20, 20]
discrete_os_win_size = (env.observation_space.high -
                        env.observation_space.low) / DISCRETE_OS_SIZE
# print(discrete_os_win_size)

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))
# print(q_table.shape)
# print(q_table)

ep_rewards = []  # contains each episodes reward
# ep is episode, min is the worst model we had, max is the best model
aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.astype.html
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    episode_reward = 0
    # Excuting every episode takes too longer. So we just check out the environment every 1000 steps.
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    print(discrete_state)
    # print(q_table[discrete_state])  # starting Q values
    # argmax returns the position of the largest value. max returns the largest value.
    # print(np.argmax(q_table[discrete_state]))
    """ this help to understand print(np.argmax(q_table[discrete_state]))
    a = np.arange(27).reshape((3, 3, 3))
    print(a)
    print(a(0, 0))   
    """

    done = False
    while not done:

        if np.random.random() > epsilon:  # create random float between 0 and 1
            action = np.argmax(q_table[discrete_state])
        else:
            # create random integer between 0 and 3
            action = np.random.randint(0, env.action_space.n)

        # action = 2  # There are 3 states in action. 2 is regarded as 'push the car right'
        # single underscore is used to ignore the value that is trivial
        new_state, reward, done, _ = env.step(action)
        # every time we step with an action, we get a new state. State has two values, position(x axis) and velocity
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)
        if render:
            env.render()  # preventing error
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward +
                                                                       DISCOUNT * max_future_q)  # fomula to calculate all Q values
            # update the discrete state after taking the step of the discrete state
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % 10:
        np.save("qtable.npy", q_table)

    if not episode % SHOW_EVERY:
        average_reward = (
            sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['ep'].append(episode)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))

        print(
            f"Episode: {episode} avg:{average_reward} min:{min(ep_rewards[-SHOW_EVERY:])} max:{max(ep_rewards[-SHOW_EVERY:])}")

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max")
plt.legend(loc=4)
plt.show()
# The goal is to make it the yellow flag.
# To make it happen, we need to create Q table that given any combination of states. Then we pick the maximum comb
# we initialize with random values and then slowly update those Q values to be optimal. This process is called explore
