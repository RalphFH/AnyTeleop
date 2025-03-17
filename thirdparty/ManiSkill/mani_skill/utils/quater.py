import numpy as np
import quaternion

def product(q1_list, q2_list):
    q1 = np.quaternion(*q1_list)
    q2 = np.quaternion(*q2_list)
    q = q1 * q2
    return [q.w, q.x, q.y, q.z]
