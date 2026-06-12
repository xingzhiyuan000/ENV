import numpy as np

from ENV.Collision_LH4500 import is_Collision
from ENV.Collision_Self_LH4500 import is_Self_Collision

## 【末端距离奖励】：指数递减，越近奖励越小，越远奖励越大
## 参考《一种基于改进SAC算法的六轴机械臂路径规划》

# def distance_to_target(p_target, p_current):
#     return np.linalg.norm(p_target - p_current)



# def rewardProgress(prev_distance, current_distance, params):
#     # progress = (prev_distance - current_distance) / params.progress_reward_scale
#     # progress = np.clip(progress, -1.0, 1.0)
#
#     progress = -abs(prev_distance - current_distance) / params.progress_reward_scale
#     progress = np.clip(progress, -10.0, 0.0)
#     proximity = np.exp(-current_distance / params.proximity_reward_scale)
#
#     # 距离靠近和不变-不惩罚  # 距离远离-惩罚
#     # delta = current_distance - prev_distance
#     # progress = -np.maximum(delta, 0) / params.progress_reward_scale
#     # # 越接近奖励越高 越接近 → 越接近 0 越远 → 越负
#     # proximity = np.exp(-current_distance / params.proximity_reward_scale) - 1.0
#
#     return progress + proximity

## 关节2进展奖励
def rewardProgress(prev_q2_distance, current_q2_distance, params):

    # progress = (prev_q2_distance - current_q2_distance)/params.progress_reward_scale
    progress = prev_q2_distance - current_q2_distance

    return progress


def rewardPathLen(prev_distance, current_distance, params):
    progress = -abs(prev_distance - current_distance)/ params.progress_reward_scale
    return progress

# # 带权重关节能耗
def rewardEnergyCost(prev_joint, current_joint, params):
    # energyCost = -np.linalg.norm(prev_joint - current_joint, ord=1)   # 2范数

    energyCost = -np.sum(params.jointCostWeight * np.abs(prev_joint - current_joint))
    return energyCost

# 关节距离
def rewardJointDis(q1_normal, q2_normal):
    diff = q1_normal - q2_normal
    # 权重（prismatic 大一点，revolute 小一点）
    w = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    r = -0.5 * np.sum((w * diff) ** 2)
    return r

# ## 【碰撞奖励机制】-仅磨机碰撞
# def rewardCollision_Mill(q_c, params):
#     state_mill, _, dist_mill = is_Collision(q_c, params)  # 与磨机碰撞
#     F_rep = np.array([0, 0, 0, -1, 0, 0, 0])
#
#     if state_mill:
#         return params.collision_penalty, True, dist_mill, F_rep  ## 注意注意
#     return 0.0, False, dist_mill, F_rep

## 【碰撞奖励机制】-仅磨机碰撞
# def rewardCollision_Mill(q_c, params):
#     state_mill, _, dist_mill = is_Collision(q_c, params)  # 与磨机碰撞
#     F_rep = np.array([0, 0, 0, -1, 0, 0, 0])
#
#     # w=0.1
#     # r=1/np.exp(-w*dist_mill)
#
#     if state_mill:
#         return params.collision_penalty, True, dist_mill, F_rep
#     else:
#         return 0.0, False, 0, F_rep
#
#     # return params.collision_penalty, False, dist_mill, F_rep

## 【碰撞奖励机制】-仅磨机碰撞-连续奖励
def rewardCollision_Mill(q_c, params):
    state_mill, _, dist_mill = is_Collision(q_c, params)  # 与磨机碰撞

    if state_mill:
        return params.collision_penalty+dist_mill, True
    else:
        return dist_mill, False


## 【碰撞奖励机制】-磨机和自碰撞
def rewardCollision(q_c, params):
    state_mill, _, dist_mill = is_Collision(q_c, params)  # 与磨机碰撞
    state_self, _, dist_self = is_Self_Collision(q_c, params)  # 自碰撞

    if state_self:
        F_rep = np.array([0, 0, 1, 0, 0, 0, 0])
    elif state_mill:
        F_rep = np.array([0, 0, 0, -1, 0, 0, 0])
    else:
        F_rep = np.array([0, 0, 0, 0, 0, 0, 0])

    if state_mill & state_self:
        F_rep = np.array([0, 0, 0.5, -0.5, 0, 0, 0])

    dist = np.minimum(dist_mill, dist_self)
    state = state_mill or state_self
    # state = state_mill

    if state:
        return params.collision_penalty, True, dist, F_rep  ## 注意注意
    else:
        return 0.0, False, 0, F_rep

## 【达到目标奖励机制】仅考虑位置
# def rewardSuccess(p_target, p_current, params):
#     dis = np.linalg.norm(p_target - p_current)
#     if dis <= params.Thr_Connect:
#         r = params.success_reward
#         done = True
#     else:
#         r=0
#         done = False
#     return r, done

## 【达到目标奖励机制】考虑位置和姿态
def rewardSuccess(p_target, p_current, O_target, O_current, params, type=3):
    # type: 1-满足位置 2-满足姿态误差 3-同时满足
    pos_err = np.linalg.norm(p_target - p_current)                      # 位置误差
    ori_err = np.sum((np.sum(O_target * O_current, axis=0) - 1) ** 2)   # 姿态误差
    if type==1:
        condition=pos_err <= params.Thr_Connect
    elif type==2:
        condition = ori_err <= params.Thr_Orientation
    else:
        condition = pos_err <= params.Thr_Connect and ori_err <= params.Thr_Orientation

    if condition:
        r = params.success_reward
        done = True
    else:
        r=0
        done = False

    return r, done

## 【模仿参考轨迹奖励机制】
def rewardImitate(q_c_norm, traj_ref_norm):

    # 计算每个轨迹点与当前状态之间的欧氏距离
    dists = np.linalg.norm(traj_ref_norm - q_c_norm, axis=1)
    min_index = np.argmin(dists) # 最短距离
    min_dist = dists[min_index]
    r = -min_dist  # 奖励（距离越小奖励越大）

    return r

## 停止不前惩罚
def rewardStill(q_next_norm, q_c_norm):

    delta_q = np.linalg.norm(q_next_norm - q_c_norm)
    if delta_q < 0.01:
        r_still = -1  # -0.1
        done = True
    else:
        r_still = 0
        done = False

    return r_still, done

def rewardAlignment(q_next_norm, q_c_norm, q_t_norm):
    move_dir = q_next_norm - q_c_norm
    target_dir = q_t_norm - q_c_norm

    cos_sim = np.dot(move_dir, target_dir) / (
        np.linalg.norm(move_dir) *
        np.linalg.norm(target_dir) + 1e-8
    )

    r_alignment = 1 * cos_sim

    return r_alignment

# def rewardStep():
#
#     return -10  # -10

## 懒惰惩罚
# def rewardStep(step, params):
#     # r_step=-100*(step/params.num_step)**2
#     r_step=-1  # -10
#     if step < params.num_step:
#         return r_step, False
#     else:
#         return r_step, True

## 懒惰惩罚
def rewardStep(step, params):

    r_step=-100
    if step < params.num_step-1:
        return -1, False
    else:
        return r_step, True

# # TCP到目标距离奖励
# def rewardDisPosEE(p_1, p_2):
# def rewardTCP2Target(p_c, p_t):
#     # p_c 当前末端位置 p_t 目标位置
#     dis = np.linalg.norm(p_c - p_t)
#     r = -dis
#     return r

# def rewardTCP2Target(p_c, p_n, p_t):
#     # p_c 当前位置
#     # p_n 下次位置
#     # p_t 目标位置
#     err_c = np.linalg.norm(p_c - p_t)
#     err_n = np.linalg.norm(p_n - p_t)
#     r = err_c-err_n
#     return r

## 位置误差 指数形式
def rewardTCP2Target(p_n, p_t): # p_n 下次位置 p_t 目标位置
    # p_n 下次位置 p_t 目标位置
    w=0.0001
    d_pos_err = np.linalg.norm(p_n - p_t)
    r_pos = np.exp(-w*d_pos_err)
    return r_pos, d_pos_err

# Dmin
# def rewardTCP2Target(p_n, p_t, d_min): # p_n 下次位置 p_t 目标位置
#     # p_n 下次位置 p_t 目标位置 d_min 一局最小距离
#     d_pos_err = np.linalg.norm(p_n - p_t)
#
#     if d_pos_err< d_min:
#         d_min_new=d_pos_err
#     else:
#         d_min_new = d_min
#     r_pos = d_min-d_pos_err
#     return r_pos, d_pos_err, d_min_new


## 姿态误差 实际值域：[0,8]  数学推导值域：[0,12]
# def rewardOrientation(Rd, Rn):
#     # Rd 目标姿态矩阵
#     # Rc 当前姿态矩阵
#     # Rn 下次姿态矩阵
#     r_orientation = -np.sum((np.sum(Rd * Rn, axis=0) - 1) ** 2)
#
#     return r_orientation

## 姿态误差 实际值域：[0,8]  数学推导值域：[0,12]
# def rewardOrientation(Rc, Rn, Rd):
#     # Rd 目标姿态矩阵
#     # Rc 当前姿态矩阵
#     # Rn 下次姿态矩阵
#     err_c = np.sum((np.sum(Rd * Rc, axis=0) - 1) ** 2)
#     err_n = np.sum((np.sum(Rd * Rn, axis=0) - 1) ** 2)
#     r = err_c-err_n
#     return r

## 姿态误差 指数形式
def rewardOrientation(Rn, Rd):

    w=1
    ori_err = np.sum((np.sum(Rd * Rn, axis=0) - 1) ** 2)
    r_ori = np.exp(-w * ori_err)

    return r_ori, ori_err

# def rewardOrientation(Rd, Rn):
#     r_orientation =  -0.5 * (3 - np.trace(Rd.T @ Rn))
#     return r_orientation

#R 与最小值比
# def rewardOrientation(Rn, Rd, Rmin):
#
#     ori_err = np.sum((np.sum(Rd * Rn, axis=0) - 1) ** 2)
#
#     if ori_err< Rmin:
#         Rmin_new=ori_err
#     else:
#         Rmin_new = Rmin
#     r_ori = Rmin-ori_err
#
#     return r_ori, ori_err, Rmin_new


