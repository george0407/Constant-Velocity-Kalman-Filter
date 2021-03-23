import numpy as np
from matplotlib import pyplot as plt

""" constant velocity kalman filter for position and velocity """
sigma_d = 1
sigma_v = 2
sigma_z = 4

x = [np.array([[0],
               [0]])]
P = [np.array([[0.16, 0],
               [0, 0.16]])]
F = np.array([[1, 1],
             [0, 1]])
Q = np.array([[sigma_d**2, 0],
              [0, sigma_v**2]])
H = np.array([[1, 0]])
d = [x[0][0]]
v = [x[0][1]]
zt = [0]
xt = x.copy()
dt = [0]
for i in range(1, 100):
    x_temp = x[i-1].copy()
    P_temp = P[i-1].copy()
    noise_d = np.random.normal(0, sigma_d)
    noise_v = np.random.normal(0, sigma_v)
    noise_x = np.array([[noise_d],
                        [noise_v]])
    x_prior = np.dot(F, x_temp) + noise_x
    P_prior = np.dot(np.dot(F, P_temp), F.T) + Q

    z = i + np.random.normal(0, sigma_z)
    y = z - np.dot(H, x_prior.copy())
    S = np.dot(np.dot(H, P_prior), H.T) + sigma_z**2
    K = np.dot(P_prior, H.T) / S

    x_posterior = x_prior + K * y
    KH = np.dot(K, H)
    coeff = np.eye(2) - KH
    P_posterior = np.dot(coeff, P_prior)
    x.append(x_posterior)
    P.append(P_posterior)
    d.append(x_posterior[0])
    v.append(x_posterior[1])
    zt.append(z)
    xt_previous = xt[i-1].copy()
    xt_temp = np.dot(F, xt_previous) + noise_x
    xt.append(xt_temp)
    # dt.append(xt_temp[0])
    dt.append(x_prior[0])
plt.figure(num=1, figsize=(5,5))
plt.subplot(121)
plt.scatter(range(100), d, label='fusion')
plt.scatter(range(100), zt, label='observation')
plt.scatter(range(100), dt, label='motion')
plt.plot(range(100), range(100), label='groundtruth')
plt.legend(), plt.grid()

plt.subplot(122)
plt.plot(range(100), v)
plt.show()
