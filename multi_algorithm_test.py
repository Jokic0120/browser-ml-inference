import numpy as np
import random, math
import pandas as pd
import time

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

# ============== 简化但更有效的适应度函数 ==============
def fitness_simple_effective(x):
    """
    简化但更有效的适应度函数
    关键思路：让三个烟幕弹在时间上错开，形成接力遮蔽
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
        
        # 约束检查
        if v_u < 70 or v_u > 140 or t_d < 0 or t_b < 0:
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
    
    # 生成目标点（进一步简化）
    target_p = []
    for h in [0, target_h/2, target_h]:
        for ang in [0, np.pi/2, np.pi, 3*np.pi/2]:
            target_p.append([target_r*np.cos(ang),
                             target_r*np.sin(ang)+target_c[1],
                             target_c[2]+h])
    target_p = np.array(target_p)
    
    # 遮蔽时间计算
    total_shielded_time = 0
    intervals = []
    
    # 检查每个时间点
    for t in np.arange(0, 30, 0.1):  # 粗粒度检查，提高速度
        # 导弹位置
        mp = m1_0 + v_m1v * t
        
        # 检查是否有烟幕弹在此时刻提供遮蔽
        is_shielded = False
        
        for bomb in bombs:
            if bomb['start_time'] <= t <= bomb['end_time']:
                # 烟幕云团位置
                t_since_burst = t - bomb['start_time']
                sp = bomb['burst_point'] + np.array([0, 0, -sink_v * t_since_burst])
                
                # 检查遮蔽目标点的比例（降低要求）
                covered_count = 0
                for TP in target_p:
                    if point_to_segment_dist(sp, mp, TP) <= R_smoke:
                        covered_count += 1
                
                # 如果遮蔽了70%以上的目标点就认为有效
                if covered_count >= len(target_p) * 0.7:
                    is_shielded = True
                    break
        
        if is_shielded:
            total_shielded_time += 0.1
    
    # 计算连续区间（简化版）
    time_points = np.arange(0, 30, 0.1)
    shielded_flags = []
    
    for t in time_points:
        mp = m1_0 + v_m1v * t
        is_shielded = False
        
        for bomb in bombs:
            if bomb['start_time'] <= t <= bomb['end_time']:
                t_since_burst = t - bomb['start_time']
                sp = bomb['burst_point'] + np.array([0, 0, -sink_v * t_since_burst])
                
                covered_count = 0
                for TP in target_p:
                    if point_to_segment_dist(sp, mp, TP) <= R_smoke:
                        covered_count += 1
                
                if covered_count >= len(target_p) * 0.7:
                    is_shielded = True
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

# ============== 粒子群算法 ==============
def particle_swarm_optimization(pop_size=30, iterations=50):
    """粒子群优化算法"""
    print("=== 粒子群算法 ===")
    
    dim = 12
    # 参数边界 [theta, v_u, t_d, t_b] * 3
    lb = np.array([-np.pi, 70, 0, 0] * 3)
    ub = np.array([np.pi, 140, 15, 10] * 3)
    
    # 初始化粒子
    particles = np.random.uniform(lb, ub, (pop_size, dim))
    velocities = np.random.uniform(-0.1, 0.1, (pop_size, dim))
    
    # 个体最优和全局最优
    p_best = particles.copy()
    p_best_scores = np.array([fitness_simple_effective(p)[0] for p in particles])
    
    g_best_idx = np.argmax(p_best_scores)
    g_best = p_best[g_best_idx].copy()
    g_best_score = p_best_scores[g_best_idx]
    
    print(f"初始最佳: {g_best_score:.4f}")
    
    # PSO参数
    w = 0.7  # 惯性权重
    c1 = 1.5  # 个体学习因子
    c2 = 1.5  # 社会学习因子
    
    for iteration in range(iterations):
        for i in range(pop_size):
            # 更新速度
            r1, r2 = np.random.random(dim), np.random.random(dim)
            velocities[i] = (w * velocities[i] + 
                           c1 * r1 * (p_best[i] - particles[i]) + 
                           c2 * r2 * (g_best - particles[i]))
            
            # 更新位置
            particles[i] += velocities[i]
            
            # 边界处理
            particles[i] = np.clip(particles[i], lb, ub)
            
            # 评估适应度
            score, _ = fitness_simple_effective(particles[i])
            
            # 更新个体最优
            if score > p_best_scores[i]:
                p_best[i] = particles[i].copy()
                p_best_scores[i] = score
                
                # 更新全局最优
                if score > g_best_score:
                    g_best = particles[i].copy()
                    g_best_score = score
                    print(f"  PSO新纪录！第{iteration+1}轮: {g_best_score:.4f}")
        
        if (iteration + 1) % 10 == 0:
            print(f"PSO第{iteration+1}轮: 最佳={g_best_score:.4f}")
    
    return g_best, g_best_score

# ============== 遗传算法 ==============
def genetic_algorithm(pop_size=40, generations=50):
    """遗传算法"""
    print("=== 遗传算法 ===")
    
    dim = 12
    lb = np.array([-np.pi, 70, 0, 0] * 3)
    ub = np.array([np.pi, 140, 15, 10] * 3)
    
    # 初始化种群
    population = np.random.uniform(lb, ub, (pop_size, dim))
    
    best_score = 0
    best_individual = population[0].copy()  # 初始化为第一个个体
    
    for generation in range(generations):
        # 评估适应度
        scores = np.array([fitness_simple_effective(ind)[0] for ind in population])
        
        # 更新最佳个体
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_individual = population[max_idx].copy()
            print(f"  GA新纪录！第{generation+1}代: {best_score:.4f}")
        
        # 选择（轮盘赌）
        scores_shifted = scores - np.min(scores) + 1e-6
        probs = scores_shifted / np.sum(scores_shifted)
        
        new_population = []
        for _ in range(pop_size):
            # 选择两个父代
            parent1 = population[np.random.choice(pop_size, p=probs)]
            parent2 = population[np.random.choice(pop_size, p=probs)]
            
            # 交叉
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
            
            # 变异
            if np.random.random() < 0.1:
                mutation_mask = np.random.random(dim) < 0.3
                child[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
            
            # 边界处理
            child = np.clip(child, lb, ub)
            new_population.append(child)
        
        population = np.array(new_population)
        
        if (generation + 1) % 10 == 0:
            print(f"GA第{generation+1}代: 最佳={best_score:.4f}")
    
    return best_individual, best_score

# ============== 模拟退火算法 ==============
def simulated_annealing(iterations=200):
    """模拟退火算法"""
    print("=== 模拟退火算法 ===")
    
    dim = 12
    lb = np.array([-np.pi, 70, 0, 0] * 3)
    ub = np.array([np.pi, 140, 15, 10] * 3)
    
    # 初始解
    current = np.random.uniform(lb, ub, dim)
    current_score, _ = fitness_simple_effective(current)
    
    best = current.copy()
    best_score = current_score
    
    print(f"初始解: {current_score:.4f}")
    
    # 退火参数
    T0 = 10.0
    alpha = 0.95
    T = T0
    
    for iteration in range(iterations):
        # 生成邻域解
        step_size = T / T0 * 0.5  # 随温度降低减小步长
        neighbor = current + np.random.normal(0, step_size, dim)
        neighbor = np.clip(neighbor, lb, ub)
        
        # 评估邻域解
        neighbor_score, _ = fitness_simple_effective(neighbor)
        
        # 接受准则
        delta = neighbor_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current = neighbor.copy()
            current_score = neighbor_score
            
            if neighbor_score > best_score:
                best = neighbor.copy()
                best_score = neighbor_score
                print(f"  SA新纪录！第{iteration+1}轮: {best_score:.4f}")
        
        # 降温
        T *= alpha
        
        if (iteration + 1) % 40 == 0:
            print(f"SA第{iteration+1}轮: 最佳={best_score:.4f}, 温度={T:.4f}")
    
    return best, best_score

# ============== 差分进化（快速版）==============
def differential_evolution_fast(pop_size=30, generations=50):
    """快速差分进化算法"""
    print("=== 差分进化算法 ===")
    
    dim = 12
    lb = np.array([-np.pi, 70, 0, 0] * 3)
    ub = np.array([np.pi, 140, 15, 10] * 3)
    
    # 初始化种群
    population = np.random.uniform(lb, ub, (pop_size, dim))
    scores = np.array([fitness_simple_effective(ind)[0] for ind in population])
    
    best_idx = np.argmax(scores)
    best_individual = population[best_idx].copy()
    best_score = scores[best_idx]
    
    print(f"初始最佳: {best_score:.4f}")
    
    for generation in range(generations):
        for i in range(pop_size):
            # 选择三个不同个体
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = population[np.random.choice(idxs, 3, replace=False)]
            
            # 变异
            F = 0.5 + 0.3 * np.random.random()
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, lb, ub)
            
            # 交叉
            Cr = 0.7
            trial = np.where(np.random.random(dim) < Cr, mutant, population[i])
            
            # 选择
            trial_score, _ = fitness_simple_effective(trial)
            if trial_score > scores[i]:
                population[i] = trial.copy()
                scores[i] = trial_score
                
                if trial_score > best_score:
                    best_individual = trial.copy()
                    best_score = trial_score
                    print(f"  DE新纪录！第{generation+1}代: {best_score:.4f}")
        
        if (generation + 1) % 10 == 0:
            avg_score = np.mean(scores)
            print(f"DE第{generation+1}代: 最佳={best_score:.4f}, 平均={avg_score:.4f}")
    
    return best_individual, best_score

# ============== 结果分析 ==============
def analyze_result(solution, algorithm_name):
    """分析单个算法的结果"""
    score, intervals = fitness_simple_effective(solution)
    
    print(f"\n--- {algorithm_name} 结果分析 ---")
    print(f"总遮蔽时间: {score:.4f} s")
    print(f"遮蔽区间数: {len(intervals)}")
    
    for i in range(3):
        theta = solution[4*i]
        v_u = solution[4*i + 1]
        t_d = solution[4*i + 2]
        t_b = solution[4*i + 3]
        
        print(f"{uav_names[i]}: 角度={np.degrees(theta):.1f}°, "
              f"速度={v_u:.1f}m/s, 投放={t_d:.2f}s, 起爆延迟={t_b:.2f}s")
    
    for j, (start, end) in enumerate(intervals):
        print(f"  区间{j+1}: [{start:.2f}, {end:.2f}]s, 时长={end-start:.2f}s")
    
    return score, intervals

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== 多算法对比测试 ====")
    print("目标：找到能达到10+秒遮蔽时间的策略\n")
    
    results = {}
    
    # 测试各种算法
    algorithms = [
        ("粒子群算法", particle_swarm_optimization),
        ("遗传算法", genetic_algorithm),
        ("模拟退火", simulated_annealing),
        ("差分进化", differential_evolution_fast)
    ]
    
    for name, algorithm in algorithms:
        print(f"\n{'='*50}")
        start_time = time.time()
        
        if name == "模拟退火":
            best_solution, best_score = algorithm()
        else:
            best_solution, best_score = algorithm()
        
        end_time = time.time()
        
        score, intervals = analyze_result(best_solution, name)
        results[name] = {
            'solution': best_solution,
            'score': score,
            'intervals': intervals,
            'time': end_time - start_time
        }
    
    # 总结对比
    print(f"\n{'='*60}")
    print("=== 算法对比总结 ===")
    
    best_algorithm = max(results.keys(), key=lambda x: results[x]['score'])
    
    for name in results:
        score = results[name]['score']
        time_cost = results[name]['time']
        intervals_count = len(results[name]['intervals'])
        
        mark = " 🏆" if name == best_algorithm else ""
        print(f"{name}: {score:.4f}s ({intervals_count}个区间) 用时{time_cost:.1f}s{mark}")
    
    # 保存最佳结果
    best_solution = results[best_algorithm]['solution']
    
    # 准备Excel数据
    excel_results = []
    for i in range(3):
        theta = best_solution[4*i]
        v_u = best_solution[4*i + 1]
        t_d = best_solution[4*i + 2]
        t_b = best_solution[4*i + 3]
        
        dir_u = np.array([np.cos(theta), np.sin(theta), 0])
        drop = uav_positions[i] + v_u * t_d * dir_u
        v0_s = v_u * dir_u
        bp = drop + v0_s * t_b + np.array([0, 0, -0.5 * g * t_b ** 2])
        
        excel_results.append({
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
    
    df = pd.DataFrame(excel_results)
    df.to_excel("result2.xlsx", index=False)
    
    print(f"\n✅ 最佳结果来自{best_algorithm}")
    print(f"最终遮蔽时间: {results[best_algorithm]['score']:.4f} s")
    print("结果已保存到 result2.xlsx")