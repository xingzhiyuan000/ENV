import numpy as np
from mpParams import mpParams
from Collision_LH4500 import is_Collision
from Collision_Self_LH4500 import is_Self_Collision
from Evaluation import path_len
import FK_LH4500 as fk
from env_LH4500 import LH4500Env
import Tools as tools
import pandas as pd

# 通用参数
params = mpParams()

def test_rotation_matrix(T):
    R = T[:3, :3]
    should_be_I = R @ R.T
    error = np.linalg.norm(should_be_I - np.eye(3))
    print("Rotation orthogonality error:", error)


# ====== 测试2：链式一致性 ======
def test_chain_consistency():
    T02 = fk.fkine_LH4500(0.4, 0.1, 0.2, 0.6, 0.3, 0.2, 0.1, "02")
    T01 = fk.fkine_LH4500(0.4, 0.1, 0.2, 0.6, 0.3, 0.2, 0.1, "01")
    T12 = fk.homo_trans_mat(0.4, 0.1, 0.2, 0.6, 0.3, 0.2, 0.1, "12")

    error = np.linalg.norm(T02 - T01 @ T12)
    print("Chain consistency error:", error)

# ====== 测试3：【末端位姿】 ======
def test_end_pose():
    # q是7的数组 移动关节单位：mm  转动关节单位：rad
    q = np.array([1000, np.pi/2, 0, 0, 0, 0, 0])
    T08 = fk.fkine_LH4500(q, "08", True)
    T08_12 = T08[:3, :].T.reshape(1, -1)   # 展开1X12
    print("End effector pose at zero config:\n", T08)

# ====== 测试4：【磨机碰撞检测】 ======
def test_collision():
    # q是7维数组 移动关节单位：mm  转动关节单位：rad
    q = np.array([5600, np.pi/2, 0, 0, 0, 0, 0])
    state, idx, dist = is_Collision(q, params)

    print("是否碰撞:", state)
    print("最危险点:", idx)
    print("距离:", dist)


# ====== 测试4.1：【自碰撞检测】 ======
def test_self_collision():
    # q是7维数组 移动关节单位：mm  转动关节单位：rad
    # q = np.array([0, np.pi, 0, 3000, 0, 0, 0])
    q = np.array([8100, np.pi, np.deg2rad(15), 2000, 0, np.deg2rad(-103), 0])
    state, idx, dist = is_Self_Collision(q, params)

    print("是否碰撞:", state)
    print("最危险点:", idx)
    print("距离:", dist)

# ====== 测试5：【路径长度】 ======
def test_pathLen():
    seq_Path = np.array([
        [5600, 0, 0, 0, 0, 0, 0],
        [6600, 0, 0, 0, 0, 0, 0],
        [5600, 0, 0, 0, 0, 0, 0]
    ])
    length = path_len(seq_Path)
    print("路径长度:", length)

# ====== 测试6：【环境测试】 ======
def test_pathLen():
        Env = LH4500Env(params)
        Env.reset()

# ====== 测试7：【归一化测试】 ======
def test_normal():
    q = np.array([5600, np.pi, 0, 1000, 0, 0, 0])
    q_norm = tools.normalize_q(q, params)
    print(q_norm)

# ====== 测试7：【归一化测试】 ======
def test_normal():
    q = np.array([5600, np.pi, 0, 1000, 0, 0, 0])
    q_norm = tools.normalize_q(q, params)
    print(q_norm)

# ====== 测试8：【读取EXCEL】 ======
def test_excel():
    # ========= 读取轨迹 =========
    traj = pd.read_excel('traj.xlsx', header=None).values


# ====== 主测试 ======
if __name__ == "__main__":
    print("==== Test 1 ====")
    q = np.array([5600, np.pi/2, 0, 300, 0, 0, 0])
    T = fk.fkine_LH4500(q, "08", True)
    test_rotation_matrix(T)

    # print("\n==== Test 2 ====")
    # test_chain_consistency()

    print("\n==== Test 3 ====")
    test_end_pose()

    print("\n==== Test 4 ====")
    test_collision()

    print("\n==== Test 4.1 ====")
    test_self_collision()

    print("\n==== Test 5 ====")
    test_pathLen()

    print("\n==== Test 7 ====")
    test_normal()