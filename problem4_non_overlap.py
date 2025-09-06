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

# ============== 非重叠遮蔽策略的适应度函数 ==============
def fitness_non_overlap(x):
    """
    非重叠遮蔽策略：每个烟幕弹独立遮蔽，总时间为各自遮蔽时间之和
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
    
    # 生成目标点
    num_cp, num_h = 8, 4
    theta_c = np.linspace(0, 2*np.pi, num_cp, endpoint=False)
    h_levels = np.linspace(0, target_h, num_h)
    target_p = []
    for h in h_levels:
        for ang in theta_c:
            target_p.append([target_r*np.cos(ang),
                             target_r*np.sin(ang)+target_c[1],
                             target_c[2]+h])
    target_p = np.array(target_p)
    
    # ============== 分别计算每个烟幕弹的遮蔽时间 ==============
    individual_intervals = []
    total_individual_time = 0
    
    for k in range(3):  # 对每个烟幕弹
        uav_param = uav_params[k]
        smoke_intervals = []
        
        # 计算该烟幕弹的有效遮蔽时间段
        start_time = uav_param['t_d'] + uav_param['t_b']  # 起爆时间
        end_time = start_time + T_eff  # 有效结束时间
        
        if start_time >= 0 and start_time < len(tau) * dt:
            # 检查这个时间段内的遮蔽效果
            is_shielded_k = np.zeros_like(tau, dtype=bool)
            
            for i, t_global in enumerate(tau):
                if t_global < start_time or t_global > end_time:
                    is_shielded_k[i] = False
                    continue
                
                # 烟幕云团位置
                t_since_burst = t_global - start_time
                sp = bp[k] + np.array([0, 0, -sink_v * t_since_burst])
                
                # 导弹位置
                mp = m1_0 + v_m1v * t_global
                
                # 检查是否所有目标点都被这个烟幕弹遮蔽
                all_covered = True
                for TP in target_p:
                    if point_to_segment_dist(sp, mp, TP) > R_smoke:
                        all_covered = False
                        break
                
                is_shielded_k[i] = all_covered
            
            # 统计该烟幕弹的遮蔽区间
            if np.any(is_shielded_k):
                starts = np.where(np.diff(np.concatenate(([0], is_shielded_k.astype(int)))) == 1)[0]
                ends = np.where(np.diff(np.concatenate((is_shielded_k.astype(int), [0]))) == -1)[0]
                
                for s, e in zip(starts, ends):
                    interval = (tau[s], tau[e])
                    smoke_intervals.append(interval)
                    total_individual_time += (tau[e] - tau[s])
        
        individual_intervals.append(smoke_intervals)
    
    # ============== 计算重叠惩罚 ==============
    # 收集所有遮蔽区间
    all_intervals = []
    for k in range(3):
        for interval in individual_intervals[k]:
            all_intervals.append((interval[0], interval[1], k))  # (start, end, uav_index)
    
    # 按开始时间排序
    all_intervals.sort(key=lambda x: x[0])
    
    # 计算重叠时间
    overlap_penalty = 0
    for i in range(len(all_intervals)):
        for j in range(i+1, len(all_intervals)):
            start1, end1, uav1 = all_intervals[i]
            start2, end2, uav2 = all_intervals[j]
            
            # 计算重叠时间
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start < overlap_end:
                overlap_time = overlap_end - overlap_start
                overlap_penalty += overlap_time
    
    # 最终适应度 = 总遮蔽时间 - 重叠惩罚
    final_score = total_individual_time - 2.0 * overlap_penalty  # 重叠惩罚系数为2
    
    return max(0, final_score), individual_intervals, dropp, bp

# ============== 差分进化算法（带详细输出）==============
def differential_evolution_with_output(pop_size=80, gens=200):
    """带详细输出的差分进化算法"""
    dim = 12
    
    print("=== 初始化种群 ===")
    Pop = []
    
    # 策略：让三个无人机在时间上错开
    for _ in range(pop_size):
        x = []
        
        # 生成三个错开的时间窗口
        time_windows = np.sort(np.random.uniform(0, 15, 3))
        
        for i in range(3):
            # 朝向目标区域的角度
            target_dir = target_c - uav_positions[i]
            base_angle = np.arctan2(target_dir[1], target_dir[0])
            theta = base_angle + np.random.normal(0, 0.5)
            
            v_u = np.random.uniform(80, 130)
            t_d = time_windows[i] + np.random.uniform(0, 2)  # 在时间窗口内
            t_b = np.random.uniform(1, 6)
            
            x.extend([theta, v_u, t_d, t_b])
        
        Pop.append(np.array(x))
    
    # 计算初始适应度
    Fit = []
    for x in Pop:
        score, _, _, _ = fitness_non_overlap(x)
        Fit.append(score)
    Fit = np.array(Fit)
    
    best_idx = np.argmax(Fit)
    best_vec, best_score = Pop[best_idx].copy(), Fit[best_idx]
    
    print(f"初始最佳得分: {best_score:.4f} s")
    
    # 进化过程
    print("\n=== 开始进化 ===")
    for g in range(gens):
        for i in range(pop_size):
            # 差分进化操作
            F = 0.5 + 0.3 * np.random.random()
            Cr = 0.7 + 0.2 * np.random.random()
            
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = [Pop[j] for j in np.random.choice(idxs, 3, replace=False)]
            
            mutant = a + F * (b - c)
            cross = np.array([mutant[j] if random.random() < Cr else Pop[i][j] for j in range(dim)])
            
            score, _, _, _ = fitness_non_overlap(cross)
            
            if score > Fit[i]:
                Pop[i], Fit[i] = cross, score
                if score > best_score:
                    best_vec, best_score = cross, score
        
        # 每次迭代输出当前最佳结果
        print(f"\n--- 第 {g+1} 代 ---")
        print(f"最佳得分: {best_score:.4f} s")
        
        # 输出当前最佳策略的详细信息
        score, intervals, dropp, bp = fitness_non_overlap(best_vec)
        
        print("当前最佳策略:")
        total_time = 0
        for k in range(3):
            theta = best_vec[4*k]
            v_u = best_vec[4*k + 1]
            t_d = best_vec[4*k + 2]
            t_b = best_vec[4*k + 3]
            
            uav_time = 0
            for interval in intervals[k]:
                uav_time += (interval[1] - interval[0])
            total_time += uav_time
            
            print(f"  {uav_names[k]}: 角度={np.degrees(theta):.1f}°, 速度={v_u:.1f}m/s, "
                  f"投放={t_d:.2f}s, 起爆延迟={t_b:.2f}s, 遮蔽时间={uav_time:.3f}s")
            
            for j, (start, end) in enumerate(intervals[k]):
                print(f"    区间{j+1}: [{start:.3f}, {end:.3f}]s")
        
        print(f"总遮蔽时间: {total_time:.4f}s (考虑重叠惩罚后得分: {score:.4f}s)")
        
        if (g + 1) % 20 == 0:
            avg_score = np.mean(Fit)
            print(f"平均得分: {avg_score:.4f} s")
    
    return best_vec, best_score

# ============== 结果保存 ==============
def save_non_overlap_results(vec, filename="result2.xlsx"):
    """保存非重叠策略结果"""
    score, intervals, dropp, bp = fitness_non_overlap(vec)
    
    results = []
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
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
    
    df = pd.DataFrame(results)
    df.to_excel(filename, index=False)
    
    print(f"\n=== 最终结果保存到 {filename} ===")
    
    # 详细分析
    total_coverage = 0
    print("\n各无人机遮蔽分析:")
    for k in range(3):
        uav_coverage = 0
        print(f"\n{uav_names[k]}:")
        for j, (start, end) in enumerate(intervals[k]):
            duration = end - start
            uav_coverage += duration
            print(f"  区间{j+1}: [{start:.4f}, {end:.4f}]s, 时长={duration:.4f}s")
        total_coverage += uav_coverage
        print(f"  {uav_names[k]} 总遮蔽时间: {uav_coverage:.4f}s")
    
    print(f"\n三架无人机总遮蔽时间: {total_coverage:.4f}s")
    print(f"优化得分 (扣除重叠惩罚): {score:.4f}s")

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== 问题4：非重叠遮蔽策略的多无人机协同 ====")
    print("目标：让三个无人机的遮蔽时间尽可能不重叠，最大化总遮蔽时间")
    
    # 运行算法
    best_vec, best_score = differential_evolution_with_output(pop_size=60, gens=150)
    
    # 保存结果
    save_non_overlap_results(best_vec)
    
    print(f"\n=== 优化完成 ===")
    print(f"最终得分: {best_score:.4f} s")
    print("现在三个无人机应该能够实现更好的时间分工协作了！")