import time
import datetime
import numpy as np
import random
import matplotlib.pyplot as plt

a = np.ones((3, 4))
b = a[:, 0:4]
print(b)
exit()
a = {"c": {"h": 3, "ee": 1, "gg": 5}, "b": 2,
     "d": {"h": 3, "ee": 1, "gg": 5}, "e": 4}
print(a)
a.pop("c", None)
a.pop("hhh", None)
print(a)
#
# a = [67.209, 64.227, 63.800, 61.715, 58.568]
# x = np.arange(1,6,1)
# plt.plot(x,a)
# plt.title("Average accuracy over 1000 experiments")
# plt.xlabel("# of test data per driver")
# plt.ylabel("Average accuracy")
# plt.show()
