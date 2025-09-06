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

# ============== 修正的多无人机遮蔽时间计算 ==============
def fitness_multi_uav_corrected(x):
    """
    修正版：充分利用多烟幕弹的协同遮蔽优势
    x的结构：[theta1, v1, t_d1, t_b1, theta2, v2, t_d2, t_b2, theta3, v3, t_d3, t_b3]
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
    
    # 生成目标点（增加密度以提高精度）
    num_cp, num_h = 12, 6
    theta_c = np.linspace(0, 2*np.pi, num_cp, endpoint=False)
    h_levels = np.linspace(0, target_h, num_h)
    target_p = []
    for h in h_levels:
        for ang in theta_c:
            target_p.append([target_r*np.cos(ang),
                             target_r*np.sin(ang)+target_c[1],
                             target_c[2]+h])
    target_p = np.array(target_p)
    
    # ============== 关键修正：多烟幕弹联合遮蔽判定 ==============
    is_shielded = np.zeros_like(tau, dtype=bool)
    
    for i, t_global in enumerate(tau):
        # 收集当前时刻所有有效的烟幕云团
        active_smokes = []
        
        for k in range(3):
            # 检查烟幕弹是否已起爆且在有效时间内
            t_since_drop = t_global - uav_params[k]['t_d']
            t_since_burst = t_since_drop - uav_params[k]['t_b']
            
            if t_since_burst >= 0 and t_since_burst <= T_eff:
                # 烟幕云团当前位置（考虑下沉）
                sp = bp[k] + np.array([0, 0, -sink_v * t_since_burst])
                active_smokes.append(sp)
        
        if not active_smokes:
            is_shielded[i] = False
            continue
            
        # 导弹当前位置
        mp = m1_0 + v_m1v * t_global
        
        # ============== 协同遮蔽判定 ==============
        # 检查是否所有目标点都被至少一个烟幕云团遮蔽
        all_targets_covered = True
        for TP in target_p:
            target_covered = False
            for sp in active_smokes:
                if point_to_segment_dist(sp, mp, TP) <= R_smoke:
                    target_covered = True
                    break
            if not target_covered:
                all_targets_covered = False
                break
        
        is_shielded[i] = all_targets_covered
    
    # 统计遮蔽时长和区间
    total_time = np.sum(is_shielded) * dt
    intervals = []
    if np.any(is_shielded):
        starts = np.where(np.diff(np.concatenate(([0], is_shielded.astype(int)))) == 1)[0]
        ends   = np.where(np.diff(np.concatenate((is_shielded.astype(int), [0]))) == -1)[0]
        for s, e in zip(starts, ends):
            intervals.append((tau[s], tau[e]))
    
    return total_time, intervals, dropp, bp

# ============== 改进的差分进化算法 ==============
def differential_evolution_improved(pop_size=100, gens=300, F=0.5, Cr=0.7):
    """改进的差分进化算法，专门优化多无人机协同"""
    dim = 12  # 3架无人机，每架4个参数
    
    print("初始化种群...")
    Pop = []
    
    # 策略1：时间分段协同 - 让不同无人机在不同时段发挥作用
    for _ in range(pop_size // 3):
        x = []
        time_slots = np.sort(np.random.uniform(0, 8, 3))  # 分配不同时段
        for i in range(3):
            # 朝向目标区域
            target_dir = target_c - uav_positions[i]
            base_angle = np.arctan2(target_dir[1], target_dir[0])
            theta = base_angle + np.random.normal(0, 0.3)
            v_u = np.random.uniform(90, 130)
            t_d = time_slots[i] + np.random.uniform(0, 2)
            t_b = np.random.uniform(1, 6)
            x.extend([theta, v_u, t_d, t_b])
        Pop.append(np.array(x))
    
    # 策略2：空间分布协同 - 让烟幕弹在空间上形成更好的覆盖
    for _ in range(pop_size // 3):
        x = []
        for i in range(3):
            # 根据无人机位置调整策略
            if i == 0:  # FY1最接近，早期快速部署
                theta = np.random.uniform(-0.5, 0.5)
                t_d = np.random.uniform(0, 2)
            elif i == 1:  # FY2中距离，中期稳定
                theta = np.random.uniform(-np.pi, np.pi)
                t_d = np.random.uniform(1, 4)
            else:  # FY3远距离，可以更灵活
                theta = np.random.uniform(-np.pi, np.pi)
                t_d = np.random.uniform(0, 6)
            
            v_u = np.random.uniform(80, 130)
            t_b = np.random.uniform(0.5, 8)
            x.extend([theta, v_u, t_d, t_b])
        Pop.append(np.array(x))
    
    # 策略3：随机探索
    for _ in range(pop_size - 2 * (pop_size // 3)):
        x = []
        for i in range(3):
            theta = np.random.uniform(-np.pi, np.pi)
            v_u = np.random.uniform(70, 140)
            t_d = np.random.uniform(0, 10)
            t_b = np.random.uniform(0, 12)
            x.extend([theta, v_u, t_d, t_b])
        Pop.append(np.array(x))
    
    # 计算初始适应度
    print("计算初始适应度...")
    Fit = []
    for j, x in enumerate(Pop):
        if j % 20 == 0:
            print(f"  评估个体 {j+1}/{len(Pop)}")
        t, _, _, _ = fitness_multi_uav_corrected(x)
        Fit.append(t)
    Fit = np.array(Fit)
    
    best_idx = np.argmax(Fit)
    best_vec, best_score = Pop[best_idx].copy(), Fit[best_idx]
    
    print(f"初始最佳遮蔽时间 = {best_score:.4f} s")
    
    # 进化过程
    print("开始进化...")
    for g in range(gens):
        for i in range(pop_size):
            # 自适应参数
            F_curr = 0.3 + 0.7 * np.random.random()
            Cr_curr = 0.5 + 0.4 * np.random.random()
            
            # 选择三个不同的个体
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = [Pop[j] for j in np.random.choice(idxs, 3, replace=False)]
            
            # 变异
            mutant = a + F_curr * (b - c)
            
            # 交叉
            cross = np.array([mutant[j] if random.random() < Cr_curr else Pop[i][j] for j in range(dim)])
            
            # 评估
            t, _, _, _ = fitness_multi_uav_corrected(cross)
            
            # 选择
            if t > Fit[i]:
                Pop[i], Fit[i] = cross, t
                if t > best_score:
                    best_vec, best_score = cross, t
                    print(f"  新纪录！Gen {g+1}, 遮蔽时间 = {best_score:.4f} s")
        
        # 输出进化信息
        if (g + 1) % 25 == 0:
            avg_fit = np.mean(Fit)
            print(f"Gen {g+1:3d}: 最佳 = {best_score:.4f} s, 平均 = {avg_fit:.4f} s")
    
    return best_vec, best_score

# ============== 结果输出和保存 ==============
def print_improved_results(vec, score):
    """打印改进后的结果"""
    total_time, intervals, dropp, bp = fitness_multi_uav_corrected(vec)
    
    print(f"\n===== 改进的多无人机协同策略结果 =====")
    print(f"总遮蔽时间 = {total_time:.4f} s")
    print(f"遮蔽区间数 = {len(intervals)}")
    
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
        print(f"\n{uav_names[i]}:")
        print(f"  飞行角度 = {theta:.4f} rad ({np.degrees(theta):.2f}°)")
        print(f"  飞行速度 = {v_u:.2f} m/s")
        print(f"  投放时间 = {t_d:.4f} s")
        print(f"  起爆延迟 = {t_b:.4f} s")
        print(f"  投放点 = [{dropp[i][0]:.2f}, {dropp[i][1]:.2f}, {dropp[i][2]:.2f}]")
        print(f"  起爆点 = [{bp[i][0]:.2f}, {bp[i][1]:.2f}, {bp[i][2]:.2f}]")
    
    print(f"\n遮蔽时间区间:")
    total_duration = 0
    for j, (s, e) in enumerate(intervals):
        duration = e - s
        total_duration += duration
        print(f"  区间 {j+1}: [{s:.4f}, {e:.4f}] s, 持续时间 = {duration:.4f} s")
    
    print(f"\n总有效遮蔽时间: {total_duration:.4f} s")
    print("-" * 60)
    
    return total_time, intervals, dropp, bp

def save_improved_results(vec, filename="result2.xlsx"):
    """保存改进的结果"""
    total_time, intervals, dropp, bp = fitness_multi_uav_corrected(vec)
    
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
    print(f"\n结果已保存到 {filename}")
    
    # 打印总结
    print(f"\n=== 最终总结 ===")
    print(f"多无人机协同总遮蔽时间: {total_time:.4f} s")
    print(f"遮蔽区间数: {len(intervals)}")
    if len(intervals) > 1:
        print("实现了多段连续遮蔽，充分发挥了协同优势！")

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== 问题4：改进的多无人机协同烟幕干扰策略 ====")
    print("修正算法逻辑，充分利用多烟幕弹的协同遮蔽优势")
    
    # 运行改进的差分进化算法
    best_vec, best_score = differential_evolution_improved(pop_size=120, gens=400)
    
    # 输出最优结果
    print_improved_results(best_vec, best_score)
    
    # 保存结果
    save_improved_results(best_vec, "result2_improved.xlsx")
    
    print(f"\n优化完成！最终遮蔽时间: {best_score:.4f} s")
    print("现在应该能充分体现多无人机的协同优势了！")