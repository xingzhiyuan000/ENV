import numpy as np
import ENV.FK_LH4500 as fk

def path_len(seq_Path):
    """
    计算末端执行器路径长度
    参数: seq_Path: (N, 7) 关节序列
    返回: PP_len: 路径总长度
    """
    PP_len = 0.0
    N = seq_Path.shape[0]

    for i in range(N - 1):
        q_c = seq_Path[i, :]
        q_n = seq_Path[i + 1, :]

        # 正运动学
        T_c = fk.fkine_LH4500(q_c, "08", True)
        T_n = fk.fkine_LH4500(q_n, "08", True)

        # 取末端位置（第4列前三个元素）
        p_c = T_c[:3, 3]
        p_n = T_n[:3, 3]

        PP_len += np.linalg.norm(p_c - p_n) # 累加距离

    # print(f"末端执行器路径长度: {PP_len:.2f}")

    return PP_len

# 带权重的平均关节能耗
def jont_energy_cost(seq_Path, params):

    totalEnergyCost = 0.0
    N = seq_Path.shape[0]
    for i in range(N - 1):
        q_c = seq_Path[i, :]
        q_n = seq_Path[i + 1, :]

        totalEnergyCost += np.sum(params.jointCostWeight * np.abs(q_c - q_n))
        avgEnergyCost=totalEnergyCost/(N-1)

    return avgEnergyCost