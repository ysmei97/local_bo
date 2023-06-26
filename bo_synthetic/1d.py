import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import animation

# Hyperparameter
iteration = 2
sample_point = 200
x_star_num = 3
T = 12
epsilon = 0.1
neighbor_threshold = 0.01
alpha = 10
length_scale = 0.1


def f(x):
    return 8 * np.cos(4 * x**0.7 - 0.4) - 20 * (x - 0.6)**2 + 25 * x + x**2 + 10 * np.cos(20 * (x**2.2 - 0.8))


def rbf(x, x_star):
    x_ = x - x_star
    return alpha * np.exp(-x_ * x_ / (2 * length_scale**2))


def rbf_prime(x, x_star):
    x_ = x - x_star
    return -alpha / length_scale**2 * np.exp(-x_ * x_ / (2 * length_scale**2)) * x_


def pi(mean, var):
    """Probability of improvement of joint distribution of f and f_prime"""
    p_f_f_prime = norm.sf((T - mean[0] + var[0][1] * mean[1] / var[1][1]) /
                          np.sqrt(var[0][0] - var[0][1]**2 / var[1][1]))
    p_f_prime = (norm.cdf((epsilon - mean[1]) / np.sqrt(var[1][1])) -
               norm.cdf((-epsilon - mean[1]) / np.sqrt(var[1][1]))) / (2 * epsilon)
    return p_f_f_prime * p_f_prime


# x_star = [np.random.uniform(0, 1) for i in range(x_star_num)]
x_star = [0.25, 0.5, 0.75]
x_sample = np.linspace(0, 1, sample_point)


def local_optima_pi(x_star, x_sample):
    acquisition_value = []
    mean_value = []
    var_value = []
    for j in range(sample_point):
        x = x_sample[j]
        if np.min(np.abs(x - np.array(x_star))) < neighbor_threshold:
            acquisition_value.append(0)
            continue
        kernel = alpha * rbf_kernel(np.array(x_star).reshape(-1, 1), gamma=1/(2*length_scale**2))
        kernel_pseudo_inverse = np.linalg.inv(kernel + np.diag([10**-6 for _ in range(len(x_star))]))
        mean_ = np.array([rbf(x, np.array(x_star)), rbf_prime(x, np.array(x_star))])
        mean = mean_ @ kernel_pseudo_inverse @ f(np.array(x_star))
        mean_value.append(mean[0])
        var = np.diag([alpha, alpha / length_scale**2]) - mean_ @ kernel_pseudo_inverse @ mean_.T
        var_value.append(var[0][0])
        acquisition_value.append(pi(mean, var))
    max_acquisition_value = np.max(acquisition_value)
    next_point = acquisition_value.index(max_acquisition_value) / sample_point
    x_star.append(next_point)
    return acquisition_value, mean_value, var_value


def init():
    """ Initialize the animation function"""
    pass


def animate(i):
    acquisition_value, mean_value, var_value = local_optima_pi(x_star, x_sample)
    mean_value += [0] * (sample_point - len(mean_value))
    var_value += [0] * (sample_point - len(var_value))
    acquisition_value += [0] * (sample_point - len(acquisition_value))

    ax1.clear()
    ax2.clear()
    ax1.plot(x_sample, f(x_sample), 'k', label='f(x)')
    # ax1.plot(0.905, f(0.905), 'go', ms=10)
    '''
    ax1.plot(x_sample, mean_value, 'b-', label='prediction')
    ax1.fill(np.concatenate([x_sample, x_sample[::-1]]),
             np.concatenate([np.array(mean_value) - 1.9600 * np.array(var_value),
                             (np.array(mean_value) + 1.9600 * np.array(var_value))[::-1]]),
             alpha=.3, fc='b', ec='None', label='CI')
    '''
    ax2.plot(x_sample, acquisition_value, 'y', label='PI')
    ax1.plot(np.array(x_star)[:-1], f(np.array(x_star)[:-1]), 'bo', ms=10, label='observations')
    ax1.plot(np.array(x_star)[-1:], f(np.array(x_star))[-1:], 'ro', ms=10, label='new observation')
    # print(np.array(x_star)[-1:])
    for k in range(len(x_star)):
        ax1.annotate(str(k + 1), xy=(np.array(x_star)[k], f(np.array(x_star)[k])), color='white',
                     fontsize="medium", fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax1.set_title("Iteration: %s" % str(i + x_star_num))
    ax2.set_title("Probability of improvement")
    ax1.set_ylim(-10, 25)
    ax1.legend()
    ax2.legend()


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=iteration, interval=1000, blit=False)
anim.save('bo.gif', fps=1, dpi=300)

