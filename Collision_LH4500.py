import numpy as np

import ENV.FK_LH4500 as fk

# 距离场计算
def compute_sdf(x, y, z, params):
    # 读取参数
    Lc = params.Lc
    Ht = params.Ht
    Rc = params.Rc
    theta = params.theta
    rc = params.rc
    Lm = params.Lm

    # 转为numpy数组（支持标量或数组）
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # 径向距离
    r = np.sqrt(x**2 + y**2)

    # 三段SDF
    d1 = Rc - r
    d2 = (Rc - (np.abs(z) - Lc/2) * np.tan(theta) - r) * np.cos(theta)
    d3 = rc - r

    # 初始化输出
    d = np.full_like(r, np.inf, dtype=float)

    # 区域mask
    mask1 = np.abs(z) <= Lc / 2
    mask2 = (Lc / 2 < np.abs(z)) & (np.abs(z) < (Lc / 2 + Ht))
    mask3 = ((Lc / 2 + Ht) <= np.abs(z)) & (np.abs(z) <= Lm / 2)

    # 分段赋值
    d[mask1] = d1[mask1]
    d[mask2] = d2[mask2]
    d[mask3] = d3[mask3]

    return d

# 碰撞检测
def is_Collision(newQ, params):
    """
    碰撞检测函数
    返回:
        state: True/False 是否碰撞
        idx:   最危险点索引
        D_rep: 最小距离
    """
    # ====== 1. 正运动学 ======
    T04 = fk.fkine_LH4500(newQ, "04", True)
    T05 = fk.fkine_LH4500(newQ, "05", True)
    T07 = fk.fkine_LH4500(newQ, "07", True)

    # ====== 2. 关节4检测点 ======
    C4 = np.array([
        [510,  360.5,  350, 1],
        [510, -289.5,  350, 1],
        [-510,-289.5,  350, 1],
        [-510, 360.5,  350, 1],
        [510,  360.5, -6570-newQ[3]+3000, 1],
        [510, -289.5, -6570-newQ[3]+3000, 1],
        [-510,-289.5, -6570-newQ[3]+3000, 1],
        [-510, 360.5, -6570-newQ[3]+3000, 1],
    ]).T  # shape (4,8)

    # ====== 3. 关节5检测点 ======
    C5 = np.array([
        [467, 360, 451.5, 1],
        [467, 360, -172, 1],
        [467, -360, -172, 1],
        [467, -360, 451.5, 1],
        [-780, 360, 451.5, 1],
        [-780, 360, 103, 1],
        [-780, -360, 103, 1],
        [-780, -360, 451.5, 1],
    ]).T

    # ====== 4. 关节7检测点（带衬板） ======
    # C7 = np.array([
    #     [250.5, -510, 667.5, 1],
    #     [-273, -510, 667.5, 1],
    #     [-273, 510, 667.5, 1],
    #     [250.5, 510, 667.5, 1],
    #     [250.5, -510, 187, 1],
    #     [-273, -510, 187, 1],
    #     [-273, 510, 187, 1],
    #     [250.5, 510, 187, 1],
    # ]).T

    # ====== 4. 关节7检测点（带衬板）衬板宽度减400 ======
    C7 = np.array([
        [250.5, -510, 667.5, 1],
        [-273, -510, 667.5, 1],
        [-273, 510, 667.5, 1],
        [250.5, 510, 667.5, 1],
        [250.5, -510, 187, 1],
        [-273, -510, 187, 1],
        [-273, 510, 187, 1],
        [250.5, 510, 187, 1],
        [580, -720, 887, 1],
        [-620, -720, 887, 1],
        [-620, 720, 887, 1],
        [580, 720, 887, 1],
        [580, -720, 437, 1],
        [-620, -720, 437, 1],
        [-620, 720, 437, 1],
        [580, 720, 437, 1],
    ]).T

    # ====== 5. 坐标变换到磨机坐标系 ======
    T_c0 = params.T_c0

    Cc4 = T_c0 @ T04 @ C4
    Cc5 = T_c0 @ T05 @ C5
    Cc7 = T_c0 @ T07 @ C7

    # ====== 6. 拼接所有检测点 ======
    checkPoints = np.hstack([Cc4, Cc5, Cc7])  # shape (4, N)

    # ====== 7. SDF计算 ======
    d_vals = compute_sdf(
        checkPoints[0, :],
        checkPoints[1, :],
        checkPoints[2, :],
        params
    )

    dist = np.min(d_vals)
    idx = np.argmin(d_vals)

    # ====== 8. 碰撞判断 ======
    if dist < params.Thr_keypoints:
        # print(f"发生碰撞的距离: {dist:.2f}, 检测点索引: {idx}")
        state = True
    else:
        state = False

    return state, idx, dist