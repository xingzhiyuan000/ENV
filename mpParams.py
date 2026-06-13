import numpy as np
import ENV.Tools as tools

class mpParams:
    def __init__(self):

        # 测试
        # q_start_deg = np.array([5000, -40, 0, 0, 0, 0, 0])          # 起始构型 mm/deg
        # q_target_deg = np.array([5500, 0, 0, 0, 0, 0, 0])       # 目标构型 mm/deg

        # 实际任务
        # 起始构型 mm/deg
        q_start_deg = np.array([8100, -180, 15, 2000, 0, -103, 0])

        # 目标构型 mm/deg-10
        # q_target_deg = np.array([7232, -147.3, -0.3, 3000, -7.8, -2.24, -5.4])
        # 目标构型 mm/deg-36
        # q_target_deg = np.array([10935, -120, 30.7, 1137, 26.45, 2.9, 15.23])
        # 目标构型 mm/deg-60
        q_target_deg = np.array([9747, -15, -5.77, 3000, -6.84, -6.49, -151.68])
        # q_target_deg = np.array([9747, -15, -5.77, 3000, -6.84, -6.49, 28.32])    # q7优化

        # 目标构型 mm/deg-50
        # q_target_deg = np.array([9813, -31.8, 12.4, 3000, 7.84, -4.79, 165.29])
        # q_target_deg = np.array([9813, -31.8, 12.4, 3000, 7.84, -4.79, -14.71])

        # 中间构型
        # q_target_deg = np.array([8100, -90, 0, 200, 0, 0, 0])

        # 从中间构型到远端衬板
        # q_start_deg = np.array([8100, -90, 0, 200, 0, 0, 0])
        # q_target_deg = np.array([9747, -15, -5.77, 3000, -6.84, -6.49, -151.68])

        self.q_start = tools.deg_rad_trans(q_start_deg, 'deg2rad')
        self.q_target = tools.deg_rad_trans(q_target_deg, 'deg2rad')

        self.num_step = 500                # 每局步数
        self.collision_penalty = -100.0    # 碰撞基础惩罚  -100
        self.success_reward = 100.0        # 成功奖励  100
        self.Thr_Connect = 200             # 最后连接阈值-位置
        self.Thr_Orientation = 0.01        # 最后连接阈值-姿态
        self.max_action = 50               # 最大action范围 2

        self.max_action_3 = 10              # Action3的action范围 5
        self.max_action_arr = np.array([[-10, 0], [-10, 10], [-10, 10]])

        self.update_step_rad = np.array([20, 0.1, 0.01, 20, 0.1, 0.1, 0.1]) # 关节更新步长

        self.progress_reward_scale = 20.0
        self.proximity_reward_scale = 200.0

        # ======================
        # 碰撞检测参数
        # ======================
        self.Lc = 12700          # 圆柱长度 (mm)
        self.Ht = 1383           # 圆锥和小圆柱交界 (mm)
        self.Rc = 4115           # 圆柱半径 (mm)
        self.theta = np.deg2rad(65)  # 圆锥角 (rad)
        self.rc = 1000           # 端部圆柱半径 (mm)
        self.Lm = 2 * 9550       # 磨机总长

        self.Type = 1            # 1-安装衬板  0-拆卸衬板

        self.D_act = 4000        # 碰撞作用距离

        self.R_sel = 500        # 大臂碰撞检测半径 (mm)

        # ======================
        # 公共参数
        # ======================
        # self.qmax = np.array([11500, np.pi, 40 * np.pi / 180, 3000, 75 * np.pi / 180, 30 * np.pi / 180, np.pi])
        # self.qmin = np.array([0, -np.pi, -35 * np.pi / 180, 0, -75 * np.pi / 180, -105 * np.pi / 180, -np.pi])

        self.qmax = np.array([11500, np.pi, np.deg2rad(40), 3000, np.deg2rad(75), np.deg2rad(30), np.pi])
        self.qmin = np.array([0, -np.pi, -np.deg2rad(35), 0, -np.deg2rad(75), -np.deg2rad(105), -np.pi])

        # self.jointCostWeight = np.array([0.31, 0.13, 0.11, 0.22, 0.05, 0.1, 0.08])
        self.jointCostWeight = np.array([0.35, 0.15, 0.1, 0.25, 0.05, 0.05, 0.05])

        # True表示转动关节
        self.lt = np.array([False, True, True, False, True, True, True])
        self.Thr_keypoints = 5  # 碰撞检测阈值 5

        # ======================
        # 坐标变换矩阵
        # ======================
        # 从基坐标系→磨机坐标表达
        self.T_c0 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -399.5],
            [0, 0, 1, -11790.7],
            [0, 0, 0, 1]
        ])
        # 从磨机坐标→基坐标系表达
        self.T_0c = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 399.5],
            [0, 0, 1, 11790.7],
            [0, 0, 0, 1]
        ])
        # 从远端圆锥坐标→基坐标系表达
        self.T_0c3 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 399.5],
            [0, 0, 1, 20059.59],
            [0, 0, 0, 1]
        ])
        # 从近端圆锥坐标→基坐标系表达
        self.T_0c2 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 399.5],
            [0, 0, 1, 3521.88],
            [0, 0, 0, 1]
        ])
        # 从磨机坐标→远端圆锥坐标
        self.T_c3c = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -8268.86],
            [0, 0, 0, 1]
        ])
        # 从磨机坐标→近端圆锥坐标
        self.T_c2c = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 8268.86],
            [0, 0, 0, 1]
        ])

        # ======================
        # 关节极限（角度）
        # ======================
        self.q_lim_deg = np.array([
            [0, 11500],
            [-180, 180],
            [-35, 40],
            [0, 3000],
            [-75, 75],
            [-105, 30],
            [-180, 180]
        ])

        self.delta_q_step = np.array([
            10.0,
            np.deg2rad(0.1),
            np.deg2rad(0.1),
            10.0,
            np.deg2rad(0.1),
            np.deg2rad(0.1),
            np.deg2rad(0.1)
        ])