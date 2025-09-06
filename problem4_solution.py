import numpy as np
import random, math
import pandas as pd

# ================= 固定参数 =================
g = 9.801
v_m = 300
target_c = np.array([0, 200, 0])
target_r, target_h = 7, 10
R_smoke = 10
sink_v = 3
T_eff = 20.0
dt = 0.01
tau = np.arange(0, T_eff + dt, dt)

# 导弹M1的初始位置和方向
m1_0 = np.array([20000, 0, 2000])
dir_m1 = -m1_0 / np.linalg.norm(m1_0)
v_m1v = v_m * dir_m1

# 三架无人机的初始位置
fy1_0 = np.array([17800, 0, 1800])
fy2_0 = np.array([12000, 1400, 1400])
fy3_0 = np.array([6000, -3000, 700])

uav_positions = [fy1_0, fy2_0, fy3_0]
uav_names = ['FY1', 'FY2', 'FY3']

# ============== 工具函数 ==============
def point_to_segment_dist(P, A, B):
    """计算点P到线段AB的最短距离"""
    AB, AP, BP = B - A, P - A, P - B
    denom = np.dot(AB, AB)
    if denom < 1e-12: return np.linalg.norm(AP)
    t = np.dot(AP, AB) / denom
    if t < 0: return np.linalg.norm(AP)
    elif t > 1: return np.linalg.norm(BP)
    else: return np.linalg.norm(P - (A + t * AB))

# ============== 多无人机遮蔽时间计算 ==============
def fitness_multi_uav(x):
    """
    x的结构：[theta1, v1, t_d1, t_b1, theta2, v2, t_d2, t_b2, theta3, v3, t_d3, t_b3]
    每架无人机4个参数：飞行角度、速度、投放时间、起爆延迟
    """
    if len(x) != 12:
        return 0.0, [], [], []
    
    # 解析参数
    uav_params = []
    for i in range(3):
        theta = x[4*i]
        v_u = x[4*i + 1]
        t_d = x[4*i + 2]
        t_b = x[4*i + 3]
        
        # 约束检查
        if v_u < 70 or v_u > 140 or t_d < 0 or t_b < 0:
            return 0.0, [], [], []
        
        uav_params.append({
            'theta': theta,
            'v_u': v_u,
            't_d': t_d,
            't_b': t_b,
            'pos_0': uav_positions[i]
        })
    
    # 计算各无人机的投放点和起爆点
    dropp, bp = [], []
    for params in uav_params:
        dir_u = np.array([np.cos(params['theta']), np.sin(params['theta']), 0])
        
        # 投放点
        drop = params['pos_0'] + params['v_u'] * params['t_d'] * dir_u
        
        # 起爆点（考虑重力影响）
        v0_s = params['v_u'] * dir_u
        bpi = drop + v0_s * params['t_b'] + np.array([0, 0, -0.5 * g * params['t_b'] ** 2])
        
        dropp.append(drop)
        bp.append(bpi)
    
    # 生成目标点
    num_cp, num_h = 10, 5
    theta_c = np.linspace(0, 2*np.pi, num_cp)
    h_levels = np.linspace(0, target_h, num_h)
    target_p = []
    for h in h_levels:
        for ang in theta_c:
            target_p.append([target_r*np.cos(ang),
                             target_r*np.sin(ang)+target_c[1],
                             target_c[2]+h])
    target_p = np.array(target_p)
    
    # 遮蔽判定
    is_shielded = np.zeros_like(tau, dtype=bool)
    for i, t_local in enumerate(tau):
        shield_now = False
        
        # 检查每枚烟幕弹是否能提供遮蔽
        for k in range(3):
            # 烟幕云团位置（考虑下沉）
            sp = bp[k] + np.array([0, 0, -sink_v * t_local])
            
            # 导弹位置
            total_t = uav_params[k]['t_d'] + uav_params[k]['t_b'] + t_local
            mp = m1_0 + v_m1v * total_t
            
            # 检查是否所有目标点都被遮蔽
            all_ok = True
            for TP in target_p:
                if point_to_segment_dist(sp, mp, TP) > R_smoke:
                    all_ok = False
                    break
            
            if all_ok:
                shield_now = True
                break
        
        is_shielded[i] = shield_now
    
    # 统计遮蔽时长和区间
    total_time = np.sum(is_shielded) * dt
    intervals = []
    if np.any(is_shielded):
        starts = np.where(np.diff(np.concatenate(([0], is_shielded.astype(int)))) == 1)[0]
        ends   = np.where(np.diff(np.concatenate((is_shielded.astype(int), [0]))) == -1)[0]
        for s, e in zip(starts, ends):
            intervals.append((tau[s], tau[e]))
    
    return total_time, intervals, dropp, bp

# ============== 差分进化算法 ==============
def differential_evolution_multi(pop_size=80, gens=200, F=0.5, Cr=0.7):
    """多无人机协同的差分进化算法"""
    dim = 12  # 3架无人机，每架4个参数
    
    # 初始化种群 - 使用更智能的初始化策略
    Pop = []
    
    # 添加一些启发式解
    for _ in range(pop_size // 4):
        x = []
        for i in range(3):
            # 朝向假目标方向的角度作为基准
            base_angle = np.arctan2(-uav_positions[i][1], -uav_positions[i][0])
            theta = base_angle + np.random.normal(0, 0.5)  # 在基准角度附近扰动
            v_u = np.random.uniform(100, 130)              # 偏向较高速度
            t_d = np.random.uniform(0, 5)                  # 较早投放
            t_b = np.random.uniform(1, 8)                  # 适中的起爆延迟
            x.extend([theta, v_u, t_d, t_b])
        Pop.append(np.array(x))
    
    # 剩余个体随机初始化
    for _ in range(pop_size - pop_size // 4):
        x = []
        for i in range(3):  # 3架无人机
            theta = np.random.uniform(-np.pi, np.pi)  # 飞行角度
            v_u = np.random.uniform(70, 140)          # 飞行速度
            t_d = np.random.uniform(0, 10)            # 投放时间
            t_b = np.random.uniform(0, 15)            # 起爆延迟
            x.extend([theta, v_u, t_d, t_b])
        Pop.append(np.array(x))
    
    # 计算初始适应度
    Fit = []
    for x in Pop:
        t, _, _, _ = fitness_multi_uav(x)
        Fit.append(t)
    Fit = np.array(Fit)
    
    best_idx = np.argmax(Fit)
    best_vec, best_score = Pop[best_idx].copy(), Fit[best_idx]
    
    print(f"Initial best shielding = {best_score:.4f} s")
    
    # 自适应参数
    F_values = [0.3, 0.5, 0.8, 1.0]
    Cr_values = [0.5, 0.7, 0.9]
    
    # 进化过程
    for g in range(gens):
        for i in range(pop_size):
            # 自适应选择F和Cr
            F_curr = np.random.choice(F_values)
            Cr_curr = np.random.choice(Cr_values)
            
            # 选择三个不同的个体
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = [Pop[j] for j in np.random.choice(idxs, 3, replace=False)]
            
            # 变异
            mutant = a + F_curr * (b - c)
            
            # 交叉
            cross = np.array([mutant[j] if random.random() < Cr_curr else Pop[i][j] for j in range(dim)])
            
            # 评估
            t, _, _, _ = fitness_multi_uav(cross)
            
            # 选择
            if t > Fit[i]:
                Pop[i], Fit[i] = cross, t
                if t > best_score:
                    best_vec, best_score = cross, t
        
        # 输出进化信息
        if (g + 1) % 20 == 0:
            avg_fit = np.mean(Fit)
            print(f"Gen {g+1:3d}: best = {best_score:.4f} s, avg = {avg_fit:.4f} s")
    
    return best_vec, best_score

# ============== 结果输出和保存 ==============
def print_multi_uav_params(vec, score):
    """打印多无人机参数"""
    total_time, intervals, dropp, bp = fitness_multi_uav(vec)
    
    print(f"\n===== Multi-UAV Strategy Results =====")
    print(f"Total shielding time = {total_time:.4f} s")
    print(f"Number of intervals = {len(intervals)}")
    
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
        print(f"\n{uav_names[i]}:")
        print(f"  Flight angle = {theta:.4f} rad ({np.degrees(theta):.2f} deg)")
        print(f"  Flight speed = {v_u:.2f} m/s")
        print(f"  Drop time = {t_d:.4f} s")
        print(f"  Burst delay = {t_b:.4f} s")
        print(f"  Drop point = [{dropp[i][0]:.2f}, {dropp[i][1]:.2f}, {dropp[i][2]:.2f}]")
        print(f"  Burst point = [{bp[i][0]:.2f}, {bp[i][1]:.2f}, {bp[i][2]:.2f}]")
    
    print(f"\nShielding intervals:")
    for j, (s, e) in enumerate(intervals):
        print(f"  Interval {j+1}: [{s:.4f}, {e:.4f}] s, duration = {e-s:.4f} s")
    
    print("-" * 60)
    return total_time, intervals, dropp, bp

def save_results_to_excel(vec, filename="result2.xlsx"):
    """保存结果到Excel文件"""
    total_time, intervals, dropp, bp = fitness_multi_uav(vec)
    
    # 准备数据
    results = []
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
        # 计算飞行方向（单位向量）
        dir_x = np.cos(theta)
        dir_y = np.sin(theta)
        dir_z = 0.0
        
        results.append({
            '无人机编号': uav_names[i],
            '飞行方向x': dir_x,
            '飞行方向y': dir_y,
            '飞行方向z': dir_z,
            '飞行速度(m/s)': v_u,
            '投放时间(s)': t_d,
            '起爆延迟(s)': t_b,
            '投放点x': dropp[i][0],
            '投放点y': dropp[i][1],
            '投放点z': dropp[i][2],
            '起爆点x': bp[i][0],
            '起爆点y': bp[i][1],
            '起爆点z': bp[i][2]
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"\nResults saved to {filename}")
    
    # 打印总结信息
    print(f"\nSummary:")
    print(f"Total effective shielding time: {total_time:.4f} s")
    print(f"Number of shielding intervals: {len(intervals)}")

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== Problem 4: Multi-UAV Smoke Screen Strategy ====")
    print("Using FY1, FY2, FY3 to deploy smoke screens against M1")
    
    # 运行差分进化算法
    best_vec, best_score = differential_evolution_multi(pop_size=60, gens=150)
    
    # 输出最优结果
    print_multi_uav_params(best_vec, best_score)
    
    # 保存结果到Excel
    save_results_to_excel(best_vec, "result2.xlsx")
    
    print("\nOptimization completed!")