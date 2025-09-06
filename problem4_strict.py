import numpy as np
import random, math
import pandas as pd

# ================= 严格按照题目给定的参数 =================
g = 9.801
v_m = 300  # 导弹飞行速度
target_c = np.array([0, 200, 0])  # 真目标位置
target_r, target_h = 7, 10  # 目标半径和高度
R_smoke = 10  # 烟幕有效遮蔽半径
sink_v = 3  # 烟幕下沉速度
T_eff = 20.0  # 烟幕有效时间
dt = 0.01
tau = np.arange(0, T_eff + dt, dt)

# 导弹M1的初始位置和飞行方向（严格按题目）
m1_0 = np.array([20000, 0, 2000])
# 导弹直指假目标(0,0,0)，所以方向向量是
dir_m1 = -m1_0 / np.linalg.norm(m1_0)
v_m1v = v_m * dir_m1

# 三架无人机的初始位置（严格按题目）
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

# ============== 严格的适应度函数 ==============
def fitness_strict_conditions(x):
    """
    严格按照题目条件的适应度函数
    - 烟幕必须遮蔽所有目标点才算有效
    - 不修改任何物理参数
    - 严格按照题目的几何关系
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
        
        # 严格的约束检查（按题目要求）
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
    
    # 生成目标点（严格按题目要求 - 圆柱形目标）
    num_cp, num_h = 8, 5  # 足够的密度确保准确性
    theta_c = np.linspace(0, 2*np.pi, num_cp, endpoint=False)
    h_levels = np.linspace(0, target_h, num_h)
    target_p = []
    for h in h_levels:
        for ang in theta_c:
            target_p.append([target_r*np.cos(ang),
                             target_r*np.sin(ang)+target_c[1],
                             target_c[2]+h])
    target_p = np.array(target_p)
    
    # ============== 严格的遮蔽判定 ==============
    is_shielded = np.zeros_like(tau, dtype=bool)
    
    for i, t_local in enumerate(tau):
        # 收集当前时刻所有有效的烟幕云团
        active_smokes = []
        
        for k in range(3):
            # 检查烟幕弹是否已起爆且在有效时间内
            t_since_drop = t_local - uav_params[k]['t_d']
            t_since_burst = t_since_drop - uav_params[k]['t_b']
            
            if t_since_burst >= 0 and t_since_burst <= T_eff:
                # 烟幕云团当前位置（考虑下沉）
                sp = bp[k] + np.array([0, 0, -sink_v * t_since_burst])
                active_smokes.append(sp)
        
        if not active_smokes:
            is_shielded[i] = False
            continue
            
        # 导弹当前位置
        mp = m1_0 + v_m1v * t_local
        
        # ============== 严格遮蔽判定：必须所有目标点都被遮蔽 ==============
        all_targets_covered = True
        for TP in target_p:
            target_covered = False
            # 检查是否至少有一个烟幕云团遮蔽了这个目标点
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

# ============== 改进的差分进化算法（但保持严格条件）==============
def differential_evolution_strict(pop_size=80, gens=200):
    """
    在严格条件下的差分进化算法
    重点优化策略而不是修改判定条件
    """
    print("=== 严格条件下的差分进化算法 ===")
    
    dim = 12
    lb = np.array([-np.pi, 70, 0, 0] * 3)
    ub = np.array([np.pi, 140, 15, 12] * 3)
    
    # 智能初始化策略
    Pop = []
    
    # 策略1：基于几何分析的初始化
    for _ in range(pop_size // 3):
        x = []
        for i in range(3):
            # 分析：导弹从(20000,0,2000)飞向(0,0,0)
            # 在不同时刻，导弹位置为 m1_0 + v_m1v * t
            
            # 为每架无人机选择不同的拦截时刻
            intercept_times = [3 + i*2, 6 + i*2, 9 + i*2]
            t_intercept = intercept_times[i] + np.random.uniform(-1, 1)
            
            # 预测该时刻的导弹位置
            missile_pos_future = m1_0 + v_m1v * t_intercept
            
            # 计算从无人机到拦截点的方向
            intercept_dir = missile_pos_future - uav_positions[i]
            base_angle = np.arctan2(intercept_dir[1], intercept_dir[0])
            
            # 在基础角度附近扰动
            theta = base_angle + np.random.normal(0, 0.3)
            v_u = np.random.uniform(90, 130)
            
            # 估算需要的投放时间
            dist = np.linalg.norm(intercept_dir)
            est_flight_time = dist / v_u
            t_d = max(0, t_intercept - est_flight_time - np.random.uniform(1, 4))
            t_b = np.random.uniform(1, 6)
            
            x.extend([theta, v_u, t_d, t_b])
        
        Pop.append(np.array(x))
    
    # 策略2：时间错开策略
    for _ in range(pop_size // 3):
        x = []
        time_offsets = [0, 4, 8]  # 错开时间
        np.random.shuffle(time_offsets)
        
        for i in range(3):
            # 朝向目标区域的大致方向
            target_dir = target_c - uav_positions[i]
            base_angle = np.arctan2(target_dir[1], target_dir[0])
            
            theta = base_angle + np.random.normal(0, 0.5)
            v_u = np.random.uniform(80, 130)
            t_d = time_offsets[i] + np.random.uniform(0, 3)
            t_b = np.random.uniform(1, 8)
            
            x.extend([theta, v_u, t_d, t_b])
        
        Pop.append(np.array(x))
    
    # 策略3：随机探索
    for _ in range(pop_size - 2*(pop_size//3)):
        x = np.random.uniform(lb, ub)
        Pop.append(x)
    
    # 评估初始种群
    Fit = []
    print("评估初始种群...")
    for j, x in enumerate(Pop):
        if j % 20 == 0:
            print(f"  评估 {j+1}/{len(Pop)}")
        score, _, _, _ = fitness_strict_conditions(x)
        Fit.append(score)
    Fit = np.array(Fit)
    
    best_idx = np.argmax(Fit)
    best_vec, best_score = Pop[best_idx].copy(), Fit[best_idx]
    
    print(f"初始最佳遮蔽时间: {best_score:.4f} s")
    
    # 进化过程
    print("\n开始进化优化...")
    for g in range(gens):
        for i in range(pop_size):
            # 自适应参数
            F = 0.5 + 0.3 * np.random.random()
            Cr = 0.7 + 0.2 * np.random.random()
            
            # 差分进化操作
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = [Pop[j] for j in np.random.choice(idxs, 3, replace=False)]
            
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, lb, ub)
            
            cross = np.array([mutant[j] if random.random() < Cr else Pop[i][j] for j in range(dim)])
            
            score, _, _, _ = fitness_strict_conditions(cross)
            
            if score > Fit[i]:
                Pop[i], Fit[i] = cross, score
                if score > best_score:
                    best_vec, best_score = cross, score
                    print(f"  新纪录！第{g+1}代: {best_score:.4f} s")
        
        if (g + 1) % 25 == 0:
            avg_score = np.mean(Fit)
            print(f"第{g+1:3d}代: 最佳={best_score:.4f} s, 平均={avg_score:.4f} s")
    
    return best_vec, best_score

# ============== 结果分析 ==============
def analyze_strict_result(vec):
    """分析严格条件下的结果"""
    score, intervals, dropp, bp = fitness_strict_conditions(vec)
    
    print(f"\n=== 严格条件下的结果分析 ===")
    print(f"总遮蔽时间: {score:.4f} s")
    print(f"遮蔽区间数: {len(intervals)}")
    
    # 分析各无人机
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
        print(f"\n{uav_names[i]}:")
        print(f"  飞行角度: {np.degrees(theta):.2f}°")
        print(f"  飞行速度: {v_u:.2f} m/s")
        print(f"  投放时间: {t_d:.4f} s")
        print(f"  起爆延迟: {t_b:.4f} s")
        print(f"  投放点: [{dropp[i][0]:.2f}, {dropp[i][1]:.2f}, {dropp[i][2]:.2f}]")
        print(f"  起爆点: [{bp[i][0]:.2f}, {bp[i][1]:.2f}, {bp[i][2]:.2f}]")
    
    print(f"\n遮蔽时间区间:")
    for j, (start, end) in enumerate(intervals):
        print(f"  区间{j+1}: [{start:.4f}, {end:.4f}] s, 时长={end-start:.4f} s")
    
    # 保存结果
    results = []
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
        results.append({
            '无人机编号': uav_names[i],
            '飞行方向x': np.cos(theta),
            '飞行方向y': np.sin(theta),
            '飞行方向z': 0.0,
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
    
    df = pd.DataFrame(results)
    df.to_excel("result2.xlsx", index=False)
    print(f"\n✅ 严格条件下的结果已保存到 result2.xlsx")
    
    return score

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== 第4问：严格条件下的多无人机协同策略 ====")
    print("严格遵循题目条件，不修改任何物理参数或判定标准")
    
    # 运行严格条件下的优化
    best_vec, best_score = differential_evolution_strict(pop_size=40, gens=100)
    
    # 分析结果
    final_score = analyze_strict_result(best_vec)
    
    print(f"\n📊 最终结果:")
    print(f"在严格条件下的最佳遮蔽时间: {final_score:.4f} s")
    
    if final_score < 10:
        print(f"\n🤔 分析:")
        print("在严格的遮蔽判定条件下，要求所有目标点都必须被遮蔽，")
        print("这确实是一个极具挑战性的优化问题。")
        print("可能需要进一步的策略创新或对问题的深入理解。")
    else:
        print(f"\n🎉 成功达到理想效果！")
    
    print("\n这是在不修改题目条件下的真实结果。")