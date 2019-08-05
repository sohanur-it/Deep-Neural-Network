
import numpy as np
import matplotlib.pyplot as plt
grayhounds = 500
labs = 500

gray_heights = 28 + 4 * np.random.randn(grayhounds)
# print(gray_heights[0:10])

lab_height = 24 + 4 * np.random.randn(labs)
# lab_height

plt.hist([gray_heights, lab_height], stacked=True, color=['r', 'b'])
plt.show()
