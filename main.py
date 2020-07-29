from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import numpy as np
import random
import time

from icp import *

# num random points 
N = 100
X = np.random.rand(3,N) * 100.0

# comment below for using random points
X = np.loadtxt("data/bunny.txt")
X = X[::50,0:3].T
N = X.shape[1]

# random rotation and translation
t = np.random.rand(3,1) * 25.0

theta = np.random.rand() * 20
phi = np.random.rand() * 20
psi = np.random.rand() * 20

R = Rot.from_euler('zyx', [theta, phi, psi], degrees = True)

R = R.as_matrix()
# print("Input Rotation : \n{}".format(R))
# print("Input Translation : \n{}".format(t))

# select a subset percentage of points
subset_percent = 40
Ns = int(N * (subset_percent/100.0))
index = random.sample(list(np.arange(N)), Ns)
P = X[:, index]

# apply inverse transformation
P = ApplyInvTransformation(P, R, t)

# ICP algorithm
start = time.time()
Rr, tr, num_iter = IterativeClosestPoint(source_pts = P, target_pts = X, tau = 10e-6)
end = time.time()

print("Time taken for ICP : {}".format(end - start))
print("num_iterations: {}".format(num_iter))
# print("Rotation Estimated : \n{}".format(Rr))
# print("Translation Estimated : \n{}".format(tr))

# calculate error:
Re, te = CalcTransErrors(R, t, Rr, tr)
print("Rotational Error : {}".format(Re))
print("Translational Error : {}".format(te))

# transformed new points
Np = ApplyTransformation(P, Rr, tr)


# visual output
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[0,:], X[1,:], X[2,:], marker='o', alpha = 0.2, label="input target points")
ax.scatter(P[0,:], P[1,:], P[2,:], marker='^', label="input source points")
ax.scatter(Np[0,:], Np[1,:], Np[2,:], marker='x', label="transformed source points")
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
plt.show()
