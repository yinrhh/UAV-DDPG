import math
import numpy as np


class UAVEnv(object):
    #################### 通用参数 ####################
    height = ground_length = ground_width = 100  # 场地长宽均为100m，UAV飞行高度也是
    T = 400  # 周期320s
    slot_num = 40  # 40个间隔
    slot_time = 10  # 时间帧时长
    tao_time = 2.5  # 时隙时长
    step_punish = -100  # 跳过step惩罚

    #################### 通信模型 ####################
    B = 1 * 10 ** 6  # 带宽1MHz
    p_noisy_los = 10 ** (-13)  # 噪声功率-100dBm
    p_uplink = 0.1  # 上行链路传输功率0.1W

    #################### 计算模型 ####################
    f_ue = 2e8  # UE的计算频率0.6GHz
    f_uav = 1.2e9  # UAV的计算频率1.2GHz
    s = 1000  # 单位bit处理所需cpu圈数1000
    alpha0 = 1e-5  # 距离为1m时的参考信道增益-30dB = 0.001， -50dB = 1e-5

    #################### UAV参数 ####################
    flight_speed = 50.  # 飞行速度50m/s
    e_battery_uav = 500000  # UAV电池电量: 500kJ.
    p_fly_uav = 0.1  # UAV飞行功率
    p_cal_uav = 0.4  # UAV计算功率
    loc_uav = [50, 50]  # UAV初始位置

    #################### UE参数 ####################
    M = 4  # UE数量
    p_off_loc = 0.4  # UE发送功率
    p_cal_loc = 0.4  # UE计算功率
    loc_ue_list = np.random.randint(0, 101, size=[M, 2])  # 随机生成终端设备位置
    task_list = np.random.randint(2097153, 2621440, M)  # 随机生成终端任务大小：2~2.5MB
    remain_task_list = task_list.copy()
    # todo：随机生成任务时延限制

    # 环境最重要的3个参数
    action_bound = [-1, 1]  # 对应tanh激活函数
    action_dim = 2 + M  # 前两位表示飞行角度和距离；后M位表示目前服务于UE的卸载率
    state_dim = 3 + M * 3  # 无人机电池, 位置, 剩余计算任务, 通信中断, 数据发送速率

    def __init__(self):
        # 状态初始化：无人机电池, 位置, 剩余计算任务, 通信中断, 数据发送速率
        self.start_state = np.append(self.e_battery_uav, self.loc_uav)
        self.start_state = np.append(self.start_state, np.ravel(self.loc_ue_list))
        self.start_state = np.append(self.start_state, self.task_list)
        # todo：通信中断、数据发送速率
        # 状态设置为初始状态
        self.state = self.start_state

    def reset(self):
        # 每个episode调用一次
        # 重置环境参数
        self.e_battery_uav = 500000  # uav电池电量: 500kJ
        self.loc_uav = [50, 50]
        self.loc_ue_list = np.random.randint(0, 101, size=[self.M, 2])  # 位置信息:x在0-100随机
        self.task_list = np.random.randint(2621440, 3145729, self.M)  # 随机计算任务1.5~2MB
        # todo：通信中断、数据发送速率
        # todo：考虑遮挡情况

        # 重置状态空间：无人机电池, 位置, 剩余计算任务, 通信中断, 数据发送速率
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        # todo：通信中断、数据发送速率
        # 重置状态并返回
        return self._get_obs()

    # 抽取出来的方法，便于复用
    def _get_obs(self):
        # 返回当前系统状态：无人机电池, 位置, 剩余计算任务, 通信中断, 数据发送速率
        self.state = np.append(self.e_battery_uav, self.loc_uav)
        self.state = np.append(self.state, np.ravel(self.loc_ue_list))
        self.state = np.append(self.state, self.task_list)
        # todo：通信中断、数据发送速率
        return self.state

    def step(self, action):
        # action：前两位表示飞行角度和距离；后M位表示目前服务于UE的卸载率
        # 各项标志位
        step_redo = False
        is_terminal = False
        offloading_ratio_change = False
        reset_dist = False

        action = (action + 1) / 2  # 将取值区间位-1~1的action -> 0~1的action。避免原来action_bound为[0,1]时训练actor网络tanh函数一直取边界0

        theta = action[0] * np.pi * 2  # 角度
        speed = action[1] * self.flight_speed  # 飞行速度

        ################# UE各项参数 ######################
        task_size = action[2:]  # UE卸载的任务量
        # todo：需要转换为真正处理的任务大小
        for i in range(self.M):
            task_size[i] = (self.remain_task_list[i] * task_size[i]) if (self.task_list[i] > 0) else 0

        ################# UAV各项参数 ######################
        dis_fly = speed * self.slot_time
        e_fly = self.p_fly_uav * self.slot_time  # 飞行能耗

        # UAV飞行后的位置
        dx_uav = dis_fly * math.cos(theta)
        dy_uav = dis_fly * math.sin(theta)
        uav_after_x = self.loc_uav[0] + dx_uav
        uav_after_y = self.loc_uav[1] + dy_uav

        e_uav_total = 0  # UAV计算耗能
        for i in range(self.M):
            task = task_size[i]
            t_uav = task / (self.f_uav / self.s)  # 在UAV边缘服务器上计算时延
            e_uav = self.p_cal_uav * t_uav  # 在UAV边缘服务器上计算耗能
            e_uav_total = e_uav_total + e_uav

        ################# UAV各项参数 ######################
        t_loc_off, t_loc_cal = self.com_time(np.array([uav_after_x, uav_after_y]), task_size, task_size)

        if self.is_finished():  # 计算任务全部完成
            is_terminal = True
            reward = 0
        elif self.time_limited(t_loc_off, t_loc_cal):
            # step_redo = True
            reward = self.step_punish
            self.record_step(0, uav_after_x, uav_after_y, np.zeros(self.M), np.zeros(self.M), t_loc_off, t_loc_cal)
        elif uav_after_x < 0 or uav_after_x > self.ground_width or uav_after_y < 0 or uav_after_y > self.ground_length:
            # 如果超出边界，则飞行距离dist置零。不需要重做此步,UAV选择变化之前的位置。不改变UAV位置。
            reset_dist = True
            energy = self.com_energy(t_loc_off, t_loc_cal)  # 计算energy
            reward = -energy
            self.e_battery_uav = self.e_battery_uav - e_uav_total  # UAV剩余电量
            self.record_step(energy, self.loc_uav[0], self.loc_uav[1], task_size, task_size, t_loc_off, t_loc_cal)
        elif self.e_battery_uav < e_fly or self.e_battery_uav - e_fly < e_uav_total:  # UAV电量不能支持计算,属于UAV电量约束条件
            # 任务全部在本地计算
            energy = self.com_energy(t_loc_off, t_loc_cal)  # 计算energy
            reward = -energy
            offloading_ratio_change = True
            self.record_step(energy, uav_after_x, uav_after_y, task_size, task_size, t_loc_off, t_loc_cal)
        else:  # 电量支持飞行,且计算任务合理,且计算任务能在剩余电量内计算
            energy = self.com_energy(t_loc_off, t_loc_cal)  # 计算energy
            reward = -energy
            self.e_battery_uav = self.e_battery_uav - e_fly - e_uav_total  # UAV剩余电量
            self.loc_uav[0] = uav_after_x  # UAV飞行后的位置x
            self.loc_uav[1] = uav_after_y  # UAV飞行后的位置y
            self.record_step(energy, uav_after_x, uav_after_y, task_size, task_size, t_loc_off, t_loc_cal)

        # 环境根据执行的动作输出奖励和状态。delay和reward互为相反数。
        return self._get_obs(), reward, is_terminal, step_redo, offloading_ratio_change, reset_dist

    def record_step(self, energy, x, y, uav_task_size, local_task_size, t_loc_off, t_loc_cal):
        for i in range(self.M):
            size = self.task_list[i] - uav_task_size[i] - local_task_size[i]
            self.task_list[i] = size if size > 0 else 0
        # 记录UE花费,输出至文件保存
        file_name = 'output.txt'
        with open(file_name, 'a') as file_obj:
            file_obj.write("\nremain_task: " + '{}'.format(self.task_list))
            file_obj.write("\nuav_task_size: " + '{}'.format(uav_task_size))
            file_obj.write("\nuav_task_size: " + '{}'.format(t_loc_off))
            file_obj.write("\nlocal_task_size:" + '{}'.format(local_task_size))
            file_obj.write("\nuav_task_size: " + '{}'.format(t_loc_cal))
            file_obj.write("\nenergy:" + '{:.2f}'.format(energy))
            file_obj.write("\nUAV hover loc:" + "[" + '{:.2f}'.format(x) + ', ' + '{:.2f}'.format(y) + ']')  # 输出保留两位结果

    def com_time(self, loc_uav, uav_task_size, local_task_size):
        loc_ue_list = self.loc_ue_list
        t_loc_off = np.zeros(self.M)
        t_loc_cal = np.zeros(self.M)
        for i in range(self.M):
            ################# 通信模型 ######################
            dx = loc_uav[0] - loc_ue_list[i][0]
            dy = loc_uav[1] - loc_ue_list[i][1]
            dh = self.height
            dist_uav_ue = np.sqrt(dx * dx + dy * dy + dh * dh)
            p_noise = self.p_noisy_los
            g_uav_ue = abs(self.alpha0 / dist_uav_ue ** 2)  # 信道增益
            trans_rate = self.B * math.log2(1 + self.p_uplink * g_uav_ue / p_noise)  # 上行链路传输速率bps

            ################# UE卸载任务 ######################
            uav_task = uav_task_size[i]  # UE卸载任务大小
            t_loc_off[i] = uav_task / trans_rate  # 卸载任务耗时

            ################# UE计算任务 ######################
            local_task = local_task_size[i]
            t_loc_cal[i] = local_task / (self.f_ue / self.s)  # 本地计算耗时

            if t_loc_off[i] < 0 or t_loc_cal[i] < 0:
                raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        return t_loc_off, t_loc_cal

    def com_energy(self, t_loc_off, t_loc_cal):
        e_total_local = 0
        for i in range(self.M):
            e_loc_off = self.p_off_loc * t_loc_off[i]  # UE发送数据能耗
            e_loc_cal = self.p_cal_loc * t_loc_cal[i]  # UE计算任务能耗
            e_total_local = e_total_local + e_loc_off + e_loc_cal
        return e_total_local

    def is_finished(self):
        for i in range(self.M):
            if self.task_list[i] > 0:
                return False
        return True

    def time_limited(self, t_loc_off, t_loc_cal):
        for i in range(self.M):
            if (t_loc_off[i] > self.tao_time) or (t_loc_cal[i] > self.tao_time):
                return True
        return False
