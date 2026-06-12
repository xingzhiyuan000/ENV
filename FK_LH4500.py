import numpy as np

# 假设 global 参数用一个类或字典存储
class GlobalParams:
    theta = np.zeros(8)
    alpha = np.zeros(8)
    a = np.zeros(8)
    d = np.zeros(8)

global_params = GlobalParams()

# def convert_joint(V1, V2, V3, V4, V5, V6, V7):
#     # 偏置补偿
#     O1 = 3288
#     O2 = np.pi / 2
#     O3 = -np.pi / 2
#     O4 = 2281
#     O5 = -np.pi / 2
#     O6 = np.pi / 2
#     O7 = 0
#
#     # 计算补偿后的theta
#     theta = np.zeros(7)
#     theta[0] = V1 + O1
#     theta[1] = V2 + O2
#     theta[2] = V3 + O3
#     theta[3] = V4 + O4
#     theta[4] = V5 + O5
#     theta[5] = V6 + O6
#     theta[6] = V7 + O7
#
#     return theta

# 添加补偿量
def add_offset(q):

    d1 = q[0] + 3288
    theta2 = q[1] + np.pi/2
    theta3 = q[2] - np.pi/2
    d4 = q[3] + 2281
    theta5 = q[4] - np.pi/2
    theta6 = q[5] + np.pi/2
    theta7 = q[6]

    return np.array([d1, theta2, theta3, d4, theta5, theta6, theta7])

def homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, cmd):
    # 参数组装
    theta = np.array([
        0,
        theta2,
        theta3,
        -90 * np.pi / 180,
        theta5,
        theta6,
        theta7,
        -90 * np.pi / 180
    ])

    alpha = np.array([
        0,
        -90 * np.pi / 180,
        -90 * np.pi / 180,
        90 * np.pi / 180,
        -90 * np.pi / 180,
        90 * np.pi / 180,
        90 * np.pi / 180,
        0
    ])

    a = np.array([0,0,40,559,0,280,70.114,0])
    d = np.array([d1,230.5,0,d4,-741,0,0,587.431])

    # cmd -> index
    cmd_map = {
        "01": 0, "12": 1, "23": 2, "34": 3,
        "45": 4, "56": 5, "67": 6, "78": 7
    }
    i = cmd_map[cmd]

    # DH变换
    nx = np.cos(theta[i])
    ny = np.sin(theta[i]) * np.cos(alpha[i])
    nz = np.sin(theta[i]) * np.sin(alpha[i])

    ox = -np.sin(theta[i])
    oy = np.cos(theta[i]) * np.cos(alpha[i])
    oz = np.cos(theta[i]) * np.sin(alpha[i])

    ax = 0
    ay = -np.sin(alpha[i])
    az = np.cos(alpha[i])

    px = a[i]
    py = -np.sin(alpha[i]) * d[i]
    pz = np.cos(alpha[i]) * d[i]

    T = np.array([
        [nx, ox, ax, px],
        [ny, oy, ay, py],
        [nz, oz, az, pz],
        [0,  0,  0,  1]
    ])

    return T

# 正向运动学
def fkine_LH4500(q, type_, use_convert=True):
    # ====== 1. 是否做关节转换 ======
    if use_convert:
        d1, theta2, theta3, d4, theta5, theta6, theta7 = add_offset(q)
    else:
        d1, theta2, theta3, d4, theta5, theta6, theta7 = q

    # 逐个关节矩阵
    T01 = homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, "01")
    T12 = homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, "12")
    T23 = homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, "23")
    T34 = homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, "34")
    T45 = homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, "45")
    T56 = homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, "56")
    T67 = homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, "67")
    T78 = homo_trans_mat(d1, theta2, theta3, d4, theta5, theta6, theta7, "78")

    # 正向链
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56
    T07 = T06 @ T67
    T08 = T07 @ T78

    # 反向链
    T18 = T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T78
    T28 = T23 @ T34 @ T45 @ T56 @ T67 @ T78
    T38 = T34 @ T45 @ T56 @ T67 @ T78
    T48 = T45 @ T56 @ T67 @ T78
    T58 = T56 @ T67 @ T78
    T68 = T67 @ T78

    T17 = T12 @ T23 @ T34 @ T45 @ T56 @ T67

    # 选择输出
    T_map = {
        "01": T01, "02": T02, "03": T03, "04": T04,
        "05": T05, "06": T06, "07": T07, "08": T08,
        "18": T18, "28": T28, "38": T38, "48": T48,
        "58": T58, "68": T68, "78": T78, "17": T17
    }

    return T_map[type_]