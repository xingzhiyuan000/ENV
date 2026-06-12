import numpy as np

# q 是需要转换的np数组
# 【弧度→角度】
def rad_to_deg(q_rad):
    q_deg=np.degrees(q_rad)
    return q_deg

# q 是需要转换的np数组
# 【角度→弧度】
def deg_to_rad(q_deg):
    q_rad=np.radians(q_deg)
    return q_rad

# 关节向量q角度和弧度转换 mode=‘deg2rad’ mode=‘rad2deg’
def deg_rad_trans(q_input, mode):
    cols_r = [1, 2, 4, 5, 6]  # 转动关节列（第2,3,5,6,7列）
    q_output = np.array(q_input, dtype=float).copy()
    if mode =='deg2rad':
        q_output[..., cols_r]=np.deg2rad(q_output[..., cols_r])
    elif mode == 'rad2deg':
        q_output[..., cols_r] = np.rad2deg(q_output[..., cols_r])
    else:
        raise ValueError("Type must be 'deg2rad' or 'rad2deg'")

    return q_output

# 归一化函数，范围-1到1
def normalize_q(q_rad, params):
    q_rad = np.array(q_rad, dtype=float)

    qmin = np.array(params.qmin, dtype=float)
    qmax = np.array(params.qmax, dtype=float)

    q_norm = 2 * (q_rad - qmin) / (qmax - qmin) - 1

    q_norm = np.clip(q_norm, -1, 1) # 防止越界（可选但推荐）

    return q_norm

# 逆归一化函数 输出q：mm/rad
def denormalize_q(q_norm, params):
    q_norm = np.array(q_norm, dtype=float)

    qmin = np.array(params.qmin, dtype=float)
    qmax = np.array(params.qmax, dtype=float)

    q_rad = (q_norm + 1) / 2 * (qmax - qmin) + qmin

    return q_rad

# 关节更新步长自适应权重，末端距离越远→w_s越大
def step_weight(d_c, d_max, w_min=0.1, w_max=10):
    # w_s = w_min * np.power(10 * w_max, d_c / d_max)
    w_s = w_min * np.power(w_max/w_min, d_c / d_max)
    return w_s