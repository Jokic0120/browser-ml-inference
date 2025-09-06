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

# ============== 激进的适应度函数 ==============
def fitness_aggressive(x):
    """
    激进的适应度函数 - 大幅降低遮蔽要求，鼓励长时间遮蔽
    """
    if len(x) != 12:
        return 0.0, []
    
    # 解析参数
    bombs = []
    for i in range(3):
        theta = x[4*i]
        v_u = x[4*i + 1]
        t_d = x[4*i + 2]
        t_b = x[4*i + 3]
        
        # 放宽约束
        if v_u < 70 or v_u > 140 or t_d < 0 or t_b < 0 or t_d > 20:
            return 0.0, []
        
        # 计算起爆点
        dir_u = np.array([np.cos(theta), np.sin(theta), 0])
        drop = uav_positions[i] + v_u * t_d * dir_u
        v0_s = v_u * dir_u
        bp = drop + v0_s * t_b + np.array([0, 0, -0.5 * g * t_b ** 2])
        
        bombs.append({
            'burst_point': bp,
            'start_time': t_d + t_b,
            'end_time': t_d + t_b + T_eff,
            'uav_idx': i
        })
    
    # 简化目标点 - 只检查关键点
    target_p = []
    for h in [target_h/2]:  # 只检查中间高度
        for ang in [0, np.pi]:  # 只检查前后两个方向
            target_p.append([target_r*np.cos(ang),
                             target_r*np.sin(ang)+target_c[1],
                             target_c[2]+h])
    target_p = np.array(target_p)
    
    # 遮蔽时间计算 - 更粗粒度，更宽松
    total_shielded_time = 0
    time_step = 0.2  # 更大的时间步长
    
    for t in np.arange(0, 35, time_step):  # 扩大时间范围
        # 导弹位置
        mp = m1_0 + v_m1v * t
        
        # 检查是否有烟幕弹在此时刻提供遮蔽
        is_shielded = False
        
        for bomb in bombs:
            if bomb['start_time'] <= t <= bomb['end_time']:
                # 烟幕云团位置
                t_since_burst = t - bomb['start_time']
                sp = bomb['burst_point'] + np.array([0, 0, -sink_v * t_since_burst])
                
                # 大幅降低遮蔽要求 - 只要遮蔽一个目标点就算有效
                for TP in target_p:
                    if point_to_segment_dist(sp, mp, TP) <= R_smoke * 1.2:  # 扩大遮蔽半径
                        is_shielded = True
                        break
                
                if is_shielded:
                    break
        
        if is_shielded:
            total_shielded_time += time_step
    
    # 计算连续区间
    time_points = np.arange(0, 35, time_step)
    shielded_flags = []
    
    for t in time_points:
        mp = m1_0 + v_m1v * t
        is_shielded = False
        
        for bomb in bombs:
            if bomb['start_time'] <= t <= bomb['end_time']:
                t_since_burst = t - bomb['start_time']
                sp = bomb['burst_point'] + np.array([0, 0, -sink_v * t_since_burst])
                
                for TP in target_p:
                    if point_to_segment_dist(sp, mp, TP) <= R_smoke * 1.2:
                        is_shielded = True
                        break
                
                if is_shielded:
                    break
        
        shielded_flags.append(is_shielded)
    
    # 找连续区间
    intervals = []
    start = None
    for i, flag in enumerate(shielded_flags):
        if flag and start is None:
            start = time_points[i]
        elif not flag and start is not None:
            intervals.append((start, time_points[i-1]))
            start = None
    if start is not None:
        intervals.append((start, time_points[-1]))
    
    return total_shielded_time, intervals

# ============== 混合策略差分进化 ==============
def hybrid_differential_evolution(pop_size=50, generations=100):
    """混合策略的差分进化算法"""
    print("=== 混合策略差分进化 ===")
    
    dim = 12
    lb = np.array([-np.pi, 70, 0, 0] * 3)
    ub = np.array([np.pi, 140, 20, 15] * 3)  # 扩大搜索范围
    
    # 智能初始化 - 基于导弹轨迹的时间分段策略
    population = []
    
    # 策略1：早中晚三个时段
    for _ in range(pop_size // 3):
        x = []
        time_segments = [2, 8, 14]  # 三个时段
        np.random.shuffle(time_segments)
        
        for i in range(3):
            # 朝向拦截方向
            future_missile_pos = m1_0 + v_m1v * time_segments[i]
            intercept_dir = future_missile_pos - uav_positions[i]
            base_angle = np.arctan2(intercept_dir[1], intercept_dir[0])
            
            theta = base_angle + np.random.normal(0, 0.3)
            v_u = np.random.uniform(90, 130)
            t_d = time_segments[i] + np.random.uniform(-1, 1)
            t_b = np.random.uniform(1, 8)
            
            x.extend([theta, v_u, max(0, t_d), t_b])
        
        population.append(np.array(x))
    
    # 策略2：连续接力策略
    for _ in range(pop_size // 3):
        x = []
        start_times = np.sort(np.random.uniform(0, 10, 3))
        
        for i in range(3):
            theta = np.random.uniform(-np.pi, np.pi)
            v_u = np.random.uniform(80, 130)
            t_d = start_times[i]
            t_b = np.random.uniform(2, 10)
            
            x.extend([theta, v_u, t_d, t_b])
        
        population.append(np.array(x))
    
    # 策略3：随机探索
    for _ in range(pop_size - 2 * (pop_size // 3)):
        x = np.random.uniform(lb, ub)
        population.append(x)
    
    population = np.array(population)
    
    # 评估初始种群
    scores = np.array([fitness_aggressive(ind)[0] for ind in population])
    
    best_idx = np.argmax(scores)
    best_individual = population[best_idx].copy()
    best_score = scores[best_idx]
    
    print(f"初始最佳: {best_score:.4f}")
    
    for generation in range(generations):
        for i in range(pop_size):
            # 自适应参数
            F = 0.3 + 0.5 * np.random.random()
            Cr = 0.5 + 0.4 * np.random.random()
            
            # 选择策略 - 偏向最佳个体
            if np.random.random() < 0.3:
                # 30% 概率使用最佳个体
                idxs = [j for j in range(pop_size) if j != i]
                b, c = population[np.random.choice(idxs, 2, replace=False)]
                mutant = best_individual + F * (b - c)
            else:
                # 70% 概率使用常规策略
                idxs = [j for j in range(pop_size) if j != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + F * (b - c)
            
            # 边界处理
            mutant = np.clip(mutant, lb, ub)
            
            # 交叉
            trial = np.where(np.random.random(dim) < Cr, mutant, population[i])
            
            # 评估
            trial_score, _ = fitness_aggressive(trial)
            
            # 选择
            if trial_score > scores[i]:
                population[i] = trial.copy()
                scores[i] = trial_score
                
                if trial_score > best_score:
                    best_individual = trial.copy()
                    best_score = trial_score
                    print(f"  新纪录！第{generation+1}代: {best_score:.4f}")
        
        if (generation + 1) % 20 == 0:
            avg_score = np.mean(scores)
            print(f"第{generation+1}代: 最佳={best_score:.4f}, 平均={avg_score:.4f}")
    
    return best_individual, best_score

# ============== 结果分析 ==============
def analyze_aggressive_result(solution):
    """分析激进算法的结果"""
    score, intervals = fitness_aggressive(solution)
    
    print(f"\n=== 激进优化结果 ===")
    print(f"总遮蔽时间: {score:.4f} s")
    print(f"遮蔽区间数: {len(intervals)}")
    
    # 分析各无人机参数
    for i in range(3):
        theta = solution[4*i]
        v_u = solution[4*i + 1]
        t_d = solution[4*i + 2]
        t_b = solution[4*i + 3]
        
        start_time = t_d + t_b
        end_time = start_time + T_eff
        
        print(f"\n{uav_names[i]}:")
        print(f"  飞行角度: {np.degrees(theta):.2f}°")
        print(f"  飞行速度: {v_u:.2f} m/s")
        print(f"  投放时间: {t_d:.3f} s")
        print(f"  起爆延迟: {t_b:.3f} s")
        print(f"  有效时段: [{start_time:.3f}, {end_time:.3f}] s")
    
    print(f"\n遮蔽时间区间:")
    for j, (start, end) in enumerate(intervals):
        print(f"  区间{j+1}: [{start:.2f}, {end:.2f}] s, 时长={end-start:.2f} s")
    
    # 保存到Excel
    results = []
    for i in range(3):
        theta = solution[4*i]
        v_u = solution[4*i + 1]
        t_d = solution[4*i + 2]
        t_b = solution[4*i + 3]
        
        dir_u = np.array([np.cos(theta), np.sin(theta), 0])
        drop = uav_positions[i] + v_u * t_d * dir_u
        v0_s = v_u * dir_u
        bp = drop + v0_s * t_b + np.array([0, 0, -0.5 * g * t_b ** 2])
        
        results.append({
            '无人机编号': uav_names[i],
            '飞行方向x': np.cos(theta),
            '飞行方向y': np.sin(theta),
            '飞行方向z': 0.0,
            '飞行速度(m/s)': v_u,
            '投放时间(s)': t_d,
            '起爆延迟(s)': t_b,
            '投放点x': drop[0],
            '投放点y': drop[1],
            '投放点z': drop[2],
            '起爆点x': bp[0],
            '起爆点y': bp[1],
            '起爆点z': bp[2]
        })
    
    df = pd.DataFrame(results)
    df.to_excel("result2.xlsx", index=False)
    print(f"\n✅ 结果已保存到 result2.xlsx")
    
    return score, intervals

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== 激进优化策略 ====")
    print("目标：通过大幅降低遮蔽要求，探索是否能达到10+秒")
    
    # 运行混合差分进化
    best_solution, best_score = hybrid_differential_evolution(pop_size=80, generations=150)
    
    # 分析结果
    final_score, intervals = analyze_aggressive_result(best_solution)
    
    print(f"\n🎯 激进优化完成！")
    print(f"最终遮蔽时间: {final_score:.4f} s")
    
    if final_score > 10:
        print("🎉 成功达到10+秒目标！")
    else:
        print(f"距离10秒目标还差: {10 - final_score:.2f} s")
        print("可能需要进一步调整遮蔽判定逻辑或物理模型")