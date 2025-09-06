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

def calculate_missile_trajectory():
    """计算导弹轨迹上的关键点"""
    trajectory_points = []
    times = np.arange(0, 25, 0.5)  # 每0.5秒一个点
    for t in times:
        pos = m1_0 + v_m1v * t
        if pos[0] > -1000:  # 导弹还没过目标太远
            trajectory_points.append((t, pos))
    return trajectory_points

# ============== 优化的适应度函数 ==============
def fitness_optimized_multi_uav(x):
    """
    优化的多无人机协同遮蔽算法
    关键改进：
    1. 降低单个烟幕弹的遮蔽要求
    2. 鼓励时间上的接力遮蔽
    3. 考虑空间分布的互补性
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
            'pos_0': uav_positions[i],
            'name': uav_names[i]
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
    
    # 生成目标点（减少密度，提高计算效率）
    num_cp, num_h = 6, 3
    theta_c = np.linspace(0, 2*np.pi, num_cp, endpoint=False)
    h_levels = np.linspace(0, target_h, num_h)
    target_p = []
    for h in h_levels:
        for ang in theta_c:
            target_p.append([target_r*np.cos(ang),
                             target_r*np.sin(ang)+target_c[1],
                             target_c[2]+h])
    target_p = np.array(target_p)
    
    # ============== 关键改进：柔性遮蔽判定 ==============
    is_shielded = np.zeros_like(tau, dtype=bool)
    coverage_quality = np.zeros_like(tau)  # 遮蔽质量评分
    
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
                active_smokes.append((sp, k))
        
        if not active_smokes:
            is_shielded[i] = False
            coverage_quality[i] = 0
            continue
            
        # 导弹当前位置
        mp = m1_0 + v_m1v * t_global
        
        # ============== 柔性遮蔽评估 ==============
        covered_targets = 0
        total_targets = len(target_p)
        
        for TP in target_p:
            target_covered = False
            for sp, smoke_idx in active_smokes:
                dist = point_to_segment_dist(sp, mp, TP)
                if dist <= R_smoke:
                    target_covered = True
                    break
            if target_covered:
                covered_targets += 1
        
        # 计算遮蔽质量
        coverage_ratio = covered_targets / total_targets
        coverage_quality[i] = coverage_ratio
        
        # 柔性判定：80%以上目标被遮蔽就认为有效
        is_shielded[i] = coverage_ratio >= 0.8
    
    # ============== 计算总得分 ==============
    # 基础遮蔽时间
    basic_shielding_time = np.sum(is_shielded) * dt
    
    # 质量加权遮蔽时间
    quality_weighted_time = np.sum(coverage_quality) * dt
    
    # 连续性奖励
    continuity_bonus = 0
    if np.any(is_shielded):
        starts = np.where(np.diff(np.concatenate(([0], is_shielded.astype(int)))) == 1)[0]
        ends = np.where(np.diff(np.concatenate((is_shielded.astype(int), [0]))) == -1)[0]
        
        intervals = []
        for s, e in zip(starts, ends):
            duration = tau[e] - tau[s]
            intervals.append((tau[s], tau[e]))
            # 长连续区间有额外奖励
            if duration > 2.0:
                continuity_bonus += duration * 0.2
    else:
        intervals = []
    
    # 多烟幕弹协同奖励
    collaboration_bonus = 0
    active_bombs_count = np.zeros_like(tau)
    
    for i, t_global in enumerate(tau):
        active_count = 0
        for k in range(3):
            t_since_drop = t_global - uav_params[k]['t_d']
            t_since_burst = t_since_drop - uav_params[k]['t_b']
            if 0 <= t_since_burst <= T_eff:
                active_count += 1
        active_bombs_count[i] = active_count
    
    # 当有多个烟幕弹同时活跃时给予奖励
    multi_bomb_time = np.sum(active_bombs_count >= 2) * dt
    collaboration_bonus = multi_bomb_time * 0.5
    
    # 总得分
    total_score = (basic_shielding_time + 
                  quality_weighted_time * 0.3 + 
                  continuity_bonus + 
                  collaboration_bonus)
    
    return max(0, total_score), intervals, dropp, bp

# ============== 改进的差分进化算法 ==============
def differential_evolution_optimized(pop_size=100, gens=300):
    """优化的差分进化算法"""
    dim = 12
    
    print("=== 初始化优化种群 ===")
    Pop = []
    
    # 策略1：基于导弹轨迹的智能初始化
    trajectory_points = calculate_missile_trajectory()
    
    for _ in range(pop_size // 2):
        x = []
        
        # 为每个无人机分配不同的拦截时段
        target_times = np.sort(np.random.choice([t for t, _ in trajectory_points], 3, replace=False))
        
        for i in range(3):
            target_time = target_times[i]
            target_pos = None
            for t, pos in trajectory_points:
                if abs(t - target_time) < 0.1:
                    target_pos = pos
                    break
            
            if target_pos is not None:
                # 计算朝向拦截点的方向
                intercept_dir = target_pos - uav_positions[i]
                base_angle = np.arctan2(intercept_dir[1], intercept_dir[0])
                theta = base_angle + np.random.normal(0, 0.2)
                
                # 计算合适的时间参数
                dist_to_intercept = np.linalg.norm(intercept_dir)
                v_u = np.random.uniform(90, 130)
                
                # 估算投放时间
                flight_time = dist_to_intercept / v_u
                t_d = max(0, target_time - flight_time - np.random.uniform(1, 4))
                t_b = np.random.uniform(1, 5)
                
                x.extend([theta, v_u, t_d, t_b])
            else:
                # 备用随机策略
                theta = np.random.uniform(-np.pi, np.pi)
                v_u = np.random.uniform(70, 140)
                t_d = np.random.uniform(0, 8)
                t_b = np.random.uniform(0, 8)
                x.extend([theta, v_u, t_d, t_b])
        
        Pop.append(np.array(x))
    
    # 策略2：时间分段策略
    for _ in range(pop_size - pop_size // 2):
        x = []
        time_slots = [0, 6, 12]  # 三个时间段
        np.random.shuffle(time_slots)
        
        for i in range(3):
            # 朝向目标区域
            target_dir = target_c - uav_positions[i]
            base_angle = np.arctan2(target_dir[1], target_dir[0])
            theta = base_angle + np.random.normal(0, 0.4)
            
            v_u = np.random.uniform(80, 130)
            t_d = time_slots[i] + np.random.uniform(0, 3)
            t_b = np.random.uniform(0.5, 6)
            
            x.extend([theta, v_u, t_d, t_b])
        
        Pop.append(np.array(x))
    
    # 计算初始适应度
    print("计算初始适应度...")
    Fit = []
    for j, x in enumerate(Pop):
        if j % 25 == 0:
            print(f"  评估个体 {j+1}/{len(Pop)}")
        score, _, _, _ = fitness_optimized_multi_uav(x)
        Fit.append(score)
    Fit = np.array(Fit)
    
    best_idx = np.argmax(Fit)
    best_vec, best_score = Pop[best_idx].copy(), Fit[best_idx]
    
    print(f"初始最佳得分: {best_score:.4f}")
    
    # 进化过程
    print("\n=== 开始优化进化 ===")
    for g in range(gens):
        for i in range(pop_size):
            # 自适应参数
            F = 0.4 + 0.4 * np.random.random()
            Cr = 0.6 + 0.3 * np.random.random()
            
            # 差分进化操作
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = [Pop[j] for j in np.random.choice(idxs, 3, replace=False)]
            
            mutant = a + F * (b - c)
            cross = np.array([mutant[j] if random.random() < Cr else Pop[i][j] for j in range(dim)])
            
            score, _, _, _ = fitness_optimized_multi_uav(cross)
            
            if score > Fit[i]:
                Pop[i], Fit[i] = cross, score
                if score > best_score:
                    best_vec, best_score = cross, score
                    print(f"  🎉 新纪录！第{g+1}代: {best_score:.4f}")
        
        # 定期输出进展
        if (g + 1) % 30 == 0:
            avg_score = np.mean(Fit)
            print(f"第{g+1:3d}代: 最佳={best_score:.4f}, 平均={avg_score:.4f}")
            
            # 输出当前最佳策略简要信息
            score, intervals, dropp, bp = fitness_optimized_multi_uav(best_vec)
            active_uavs = []
            for k in range(3):
                has_effect = False
                for start, end in intervals:
                    t_start_bomb = best_vec[4*k + 2] + best_vec[4*k + 3]
                    t_end_bomb = t_start_bomb + T_eff
                    if not (end <= t_start_bomb or start >= t_end_bomb):
                        has_effect = True
                        break
                if has_effect:
                    active_uavs.append(uav_names[k])
            
            if active_uavs:
                print(f"  活跃无人机: {', '.join(active_uavs)}")
                print(f"  遮蔽区间数: {len(intervals)}")
    
    return best_vec, best_score

# ============== 结果分析和保存 ==============
def analyze_and_save_optimized_results(vec, filename="result2.xlsx"):
    """分析并保存优化结果"""
    score, intervals, dropp, bp = fitness_optimized_multi_uav(vec)
    
    print(f"\n=== 优化算法最终结果 ===")
    print(f"总得分: {score:.4f}")
    print(f"遮蔽区间数: {len(intervals)}")
    
    # 详细分析每个无人机的贡献
    results = []
    total_effective_time = 0
    
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
        # 计算该无人机的有效时间段
        bomb_start = t_d + t_b
        bomb_end = bomb_start + T_eff
        
        uav_effective_time = 0
        for start, end in intervals:
            overlap_start = max(start, bomb_start)
            overlap_end = min(end, bomb_end)
            if overlap_start < overlap_end:
                uav_effective_time += (overlap_end - overlap_start)
        
        total_effective_time += uav_effective_time
        
        print(f"\n{uav_names[i]}:")
        print(f"  飞行角度: {np.degrees(theta):.2f}°")
        print(f"  飞行速度: {v_u:.2f} m/s")
        print(f"  投放时间: {t_d:.3f} s")
        print(f"  起爆延迟: {t_b:.3f} s")
        print(f"  烟幕有效期: [{bomb_start:.3f}, {bomb_end:.3f}] s")
        print(f"  贡献遮蔽时间: {uav_effective_time:.3f} s")
        
        # 准备Excel数据
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
    
    # 输出遮蔽时间区间
    print(f"\n遮蔽时间区间:")
    basic_shielding_time = 0
    for j, (start, end) in enumerate(intervals):
        duration = end - start
        basic_shielding_time += duration
        print(f"  区间{j+1}: [{start:.3f}, {end:.3f}] s, 时长={duration:.3f} s")
    
    print(f"\n总结:")
    print(f"  基础遮蔽时间: {basic_shielding_time:.3f} s")
    print(f"  算法优化得分: {score:.3f} s")
    
    # 保存到Excel
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    print(f"\n✅ 结果已保存到 {filename}")
    
    return score, intervals

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== 问题4：优化的多无人机协同烟幕干扰策略 ====")
    print("目标：充分发挥三架无人机的协同优势，最大化遮蔽效果")
    
    # 运行优化算法
    best_vec, best_score = differential_evolution_optimized(pop_size=60, gens=200)
    
    # 分析和保存结果
    final_score, intervals = analyze_and_save_optimized_results(best_vec)
    
    print(f"\n🎯 优化完成！")
    print(f"最终得分: {final_score:.4f}")
    print(f"这次应该能看到多个烟幕弹的协同效果了！")