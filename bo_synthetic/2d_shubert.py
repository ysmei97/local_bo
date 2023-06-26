import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import norm
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm

'''------ Hyperparameter ------'''
dimension = 2
iteration = 40
sample_point = 400
x_star_num = 3
T = 0  # default 40
epsilon = 0.1  # default 0.1
neighbor_threshold = 0.2  # default 0.2
alpha = 10  # default 10
length_scale = 0.1  # default 0.1


def perm(xx, beta=10, d=dimension):
    """Perm Function 0, d, β. Bowl-Shaped.
    Dimension: d.
    Global Minimum: f(x*) = 0 at x* = (1, 1/2, ..., 1/d)
    The function is evaluated on the hypercube x_i ∈ [-d, d], for all i = 1, …, d."""
    outer = 0
    for i in range(1, 1 + d):
        inner = 0
        for j in range(1, 1 + d):
            xj = xx[:, j - 1]
            inner += (j + beta) * (xj ** i - (1 / j) ** i)
        outer += inner ** 2
    return outer


def shubert(xx):
    """Shubert Function. Many Local Minima.
    Dimension: 2.
    Global Minimum: f(x*) = -186.7309.
    The function has several local minima and many global minima. It is usually evaluated on the square xi ∈ [-10, 10],
    for all i = 1, 2, although this may be restricted to the square xi ∈ [-5.12, 5.12], for all i = 1, 2."""
    x1 = xx[:, 0]
    x2 = xx[:, 1]
    sum1 = 0
    sum2 = 0
    for i in range(1, 6):
        sum1 += i * np.cos((i + 1) * x1 + i)
        sum2 += i * np.cos((i + 1) * x2 + i)
    return sum1 * sum2


def griewank(xx, d=dimension):
    """Griewank Function. Many Local Minima.
    Dimensions: d.
    Global Minimum: f(x*) at x* = (0, ..., 0).
    The function has many widespread local minima, which are regularly distributed. It is usually evaluated
    on the hypercube xi ∈ [-600, 600], for all i = 1, …, d."""
    sum_ = 0
    prod = 1
    for i in range(1, 1 + d):
        xi = xx[:, i - 1]
        sum_ += xi ** 2 / 4000
        prod *= np.cos(xi / np.sqrt(i))
    return sum_ - prod + 1


def rbf(x, x_star):
    x_ = x - x_star
    return alpha * np.exp(-np.sum(x_ ** 2, axis=1) / (2 * length_scale ** 2))


def rbf_prime(x, x_star):
    x_ = x - x_star
    return -alpha / length_scale ** 2 * np.repeat(
        np.exp(-np.sum(x_ ** 2, axis=1) / (2 * length_scale ** 2)).reshape(-1, 1),
        dimension, axis=1) * x_


def pi(mean, var):
    """Probability of improvement of joint distribution of f and f_prime
    where |f'(x_i|x*_i)| <= ε_i"""
    p_f_f_prime = norm.sf((T - mean[0]) / np.sqrt(var[0][0]))
    p_f_prime = 1
    for i in range(dimension):
        p_f_prime_i = (norm.cdf((epsilon - mean[i + 1]) / np.sqrt(var[i + 1][i + 1])) -
                       norm.cdf((-epsilon - mean[i + 1]) / np.sqrt(var[i + 1][i + 1]))) / (2 * epsilon)
        p_f_prime *= p_f_prime_i
    return p_f_f_prime * p_f_prime


def ei(mean, var):
    """Expected improvement of joint distribution of f and f_prime
    where |f'(x_i|x*_i)| <= ε_i"""
    p_f_f_prime = 0
    for k in range(60):
        p_f_f_prime += norm.sf((T + (0.5 * k) - mean[0]) / np.sqrt(var[0][0]))
    p_f_prime = 1
    for i in range(dimension):
        p_f_prime_i = (norm.cdf((epsilon - mean[i + 1]) / np.sqrt(var[i + 1][i + 1])) -
                       norm.cdf((-epsilon - mean[i + 1]) / np.sqrt(var[i + 1][i + 1]))) / (2 * epsilon)
        p_f_prime *= p_f_prime_i
    return p_f_f_prime * p_f_prime


def local_optima_pi(x_star, x_sample):
    acquisition_value = []
    sample_point_value = []
    mean_value = []
    var_value = []
    for i in range(sample_point):
        for j in range(sample_point):
            xx = np.array([x_sample[0][i][j], x_sample[1][i][j]])
            if np.min(np.linalg.norm(xx - np.array(x_star), axis=1)) < neighbor_threshold:
                acquisition_value.append(0)
                sample_point_value.append(xx)
                continue
            kernel = alpha * rbf_kernel(np.array(x_star), gamma=1 / (2 * length_scale ** 2))
            kernel_pseudo_inverse = np.linalg.inv(kernel + np.diag([10 ** -6 for _ in range(len(x_star))]))
            mean_ = np.concatenate((rbf(xx, np.array(x_star)).reshape(-1, 1), rbf_prime(xx, np.array(x_star))), axis=1)
            mean = mean_.T @ kernel_pseudo_inverse @ shubert(np.array(x_star))
            mean_value.append(mean[0])
            var = np.diag([float(alpha)] +
                          [alpha / length_scale ** 2] * dimension) - mean_.T @ kernel_pseudo_inverse @ mean_
            var_value.append(var[0][0])
            acquisition_value.append(pi(mean, var))
            sample_point_value.append(xx)
    max_acquisition_value = np.max(acquisition_value)
    next_point = sample_point_value[acquisition_value.index(max_acquisition_value)]
    x_star.append(next_point)
    return acquisition_value, mean_value, var_value


def local_optima_ei(x_star, x_sample):
    acquisition_value = []
    sample_point_value = []
    mean_value = []
    var_value = []
    for i in range(sample_point):
        for j in range(sample_point):
            xx = np.array([x_sample[0][i][j], x_sample[1][i][j]])
            if np.min(np.linalg.norm(xx - np.array(x_star), axis=1)) < neighbor_threshold:
                acquisition_value.append(0)
                sample_point_value.append(xx)
                continue
            kernel = alpha * rbf_kernel(np.array(x_star), gamma=1 / (2 * length_scale ** 2))
            kernel_pseudo_inverse = np.linalg.inv(kernel + np.diag([10 ** -6 for _ in range(len(x_star))]))
            mean_ = np.concatenate((rbf(xx, np.array(x_star)).reshape(-1, 1), rbf_prime(xx, np.array(x_star))), axis=1)
            mean = mean_.T @ kernel_pseudo_inverse @ (shubert(np.array(x_star)) - np.array(50))
            mean_value.append(mean[0])
            var = np.diag([float(alpha)] +
                          [alpha / length_scale ** 2] * dimension) - mean_.T @ kernel_pseudo_inverse @ mean_
            var_value.append(var[0][0])
            acquisition_value.append(ei(mean, var))
            sample_point_value.append(xx)
    max_acquisition_value = np.max(acquisition_value)
    next_point = sample_point_value[acquisition_value.index(max_acquisition_value)]
    x_star.append(next_point)
    return acquisition_value, mean_value, var_value


def init():
    """ Initialize the animation function"""
    pass


def animate(i):
    # acquisition_value, mean_value, var_value = local_optima_pi(x_star, x_sample)
    acquisition_value, mean_value, var_value = local_optima_ei(x_star, x_sample)

    # mean_value += [0] * (sample_point - len(mean_value))
    # var_value += [0] * (sample_point - len(var_value))
    acquisition_value += [0] * (sample_point - len(acquisition_value))

    ax2.clear()

    f_x_star = shubert(np.array(x_star))
    ax2.plot_surface(x_sample[0], x_sample[1], np.array(acquisition_value).reshape(sample_point, -1),
                     color='y', alpha=0.6)
    ax1.scatter(np.array(x_star)[:-1, 0], np.array(x_star)[:-1, 1], f_x_star[:-1],
                s=30, color='blue', label='observations')
    ax1.scatter(np.array(x_star)[-1:, 0], np.array(x_star)[-1:, 1], f_x_star[-1:],
                s=30, color='red', label='new observation')
    for k in range(len(x_star)):
        if k < x_star_num:
            continue
        ax1.text(np.array(x_star)[k][0], np.array(x_star)[k][1], f_x_star[k], str(k - x_star_num + 1), color='black',
                 fontsize="medium", fontweight='bold')
    ax1.set_title("Shubert Function. Iteration: %s" % str(i + 1))
    # ax2.set_title("Probability of improvement")
    ax2.set_title("Expected improvement")
    ax1.set_zlim(-200, 300)

    print('Step: %s' % str(i + 1), 'new observation: ', np.array(x_star)[-1:], 'value: ', f_x_star[-1:],
          'max acquisition value: ', np.max(acquisition_value), 'min acquisition value: ', np.min(acquisition_value))


'''
x_star = []
for _ in range(x_star_num):
    x_star.append(np.array([np.random.uniform(-5, 5), np.random.uniform(-5, 5)]))
'''
x_star = [np.array([-0.5, -1]), np.array([-1, -0.5]), np.array([-1.5, -1.5])]
x1_sample = np.linspace(-2, 0, sample_point)
x2_sample = np.linspace(-2, 0, sample_point)
x_sample = np.meshgrid(x1_sample, x2_sample)
# print('Initials:', x_star, 'Values:', shubert(np.array(x_star)))

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(2, 1, 1, projection='3d')
ax2 = fig.add_subplot(2, 1, 2, projection='3d')
ax1.plot_surface(x_sample[0], x_sample[1],
                 np.array(shubert(np.dstack((x_sample[0], x_sample[1])).reshape(-1, 2))).reshape(sample_point, -1),
                 cmap=cm.coolwarm, alpha=0.5, label='f(x)')
# plt.show()
print('EI, T=0, remove mean_0')
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=iteration, interval=1000, blit=False)
anim.save('2d_shubert_ei_T0_nomean0.gif', writer='pillow', fps=1, dpi=1200)
