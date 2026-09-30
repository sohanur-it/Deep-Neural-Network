"""
Smallest possible example of `scipy.spatial.distance.euclidean` — the distance
function reused by the custom KNN classifier below.
"""



from scipy.spatial import distance


def euc(a, b):
    return distance.euclidean(a, b)


print(euc(10, 4	))
