# %%
"""
Simple script to simulate performance on a Probabilistic Instrumental Learning Task
as modelled by a Reinforcement Learning (RL) model with reciprocal value update 
(counterfactual update for unchosen option). Illustration that increased asymptotic choice 
accuracy can be attained through manipulations to reward sensitivity, value decay, and choice
stochasticity.

Accompanying editorial for:
Halahakoon DC, Kaltenboeck A, Martens M, Geddes JG, Harmer CJ, Cowen P, Browning M 
Biological Psychiatry (2023) 
Pramipexole Enhances Reward Learning by Preserving Value Estimates. 

Matthew Nour, Oxford, October 2023
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# win condition only
win_p = (0.7, 0.3)
n_trials = 100


# funcs
def choice_probs(q=None, beta=None):
    """pr(0) based on softmax"""
    return 1 / (1 + np.exp(-beta * (q[0] - q[1])))


def make_choice(pr_choice_0):
    """0 or 1, based on pr(0)"""
    chosen = np.random.choice([0, 1], p=[pr_choice_0, 1 - pr_choice_0])
    not_chosen = 0 if chosen == 1 else 1
    return chosen, not_chosen


def decay_to_uniform(q=None, decay=None):
    u = 1 / q.shape[0]
    new_q = []
    for q_i in q:
        new_q.append(q_i + decay * (u - q_i))
    return np.array(new_q)


# visualse choice rule
pr = []
x = np.linspace(0, 1, 100)
b = 10
for p in x:
    pr.append(choice_probs(q=[p, 1 - p], beta=b))
plt.plot(x, pr)
plt.xlabel("value(State A)")
plt.ylabel("pr(choice = State A)")
plt.show()


def plot_sweep(inv_temp_log=[10], rew_sens_log=[0.6], decay_log=[0.12], t=None, c=None):
    colors = mpl.colormaps[c](np.linspace(0, 1, 20))
    lr = 0.1
    # play task
    c_i = 0
    for decay in decay_log:
        for inv_temp in inv_temp_log:
            for rew_sens in rew_sens_log:
                prob_best_all_run = []
                for _ in range(200):
                    # random state outcomes
                    state_outcomes = []
                    catch = np.random.rand(n_trials) < win_p[0]
                    for v in catch:
                        if v:
                            state_outcomes.append([1, 0])
                        else:
                            state_outcomes.append([0, 1])

                    # play
                    value = [np.array([0.5, 0.5])]
                    choice = []
                    outcome = []
                    prob_best = []
                    for n in range(n_trials):
                        # state values carried forward
                        q = value[-1]
                        # value decay prior to choice
                        q = decay_to_uniform(q=q, decay=decay)
                        # make choice
                        pr0 = choice_probs(q=q, beta=inv_temp)
                        c, not_c = make_choice(pr0)
                        # get outcome
                        o = state_outcomes[n][c]
                        not_o = state_outcomes[n][not_c]
                        # update chosen state values according to received outcome
                        new_q = np.zeros((2,))
                        new_q[c] = q[c] + lr * (rew_sens * o - q[c])
                        # reciprocal update for unchosen state
                        new_q[not_c] = q[not_c] + lr * (rew_sens * not_o - q[not_c])
                        # update all
                        prob_best.append(pr0)
                        choice.append(c)
                        outcome.append(o)
                        value.append(new_q)
                    prob_best_all_run.append(prob_best)

                # Plot mean over runs
                plt.plot(np.stack(prob_best_all_run).mean(axis=0), c=colors[c_i + 9])
                c_i += 1
    plt.title(t)
    plt.ylim((0.5, 0.9))
    plt.ylabel("Accuracy")
    plt.xlabel("Trial number")
    plt.show()


plot_sweep(inv_temp_log=np.linspace(0.5, 20, 10), t="Choice stochasticity", c="Purples")
plot_sweep(decay_log=np.linspace(0.01, 0.7, 10), t="State value decay", c="Greens")
plot_sweep(rew_sens_log=np.linspace(0.1, 1, 10), t="Reward sensitivity", c="Blues")
