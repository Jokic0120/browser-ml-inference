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
T_eff = 20.0  # 每个烟幕弹的有效时间（关键！）
dt = 0.01

# 导弹M1的初始位置和飞行方向
m1_0 = np.array([20000, 0, 2000])
dir_m1 = -m1_0 / np.linalg.norm(m1_0)  # 导弹飞向假目标(0,0,0)
v_m1v = v_m * dir_m1

# 三架无人机的初始位置
fy1_0 = np.array([17800, 0, 1800])
fy2_0 = np.array([12000, 1400, 1400])
fy3_0 = np.array([6000, -3000, 700])

uav_positions = [fy1_0, fy2_0, fy3_0]
uav_names = ['FY1', 'FY2', 'FY3']

# 计算导弹到达假目标的时间
missile_total_time = np.linalg.norm(m1_0) / v_m
print(f"导弹总飞行时间: {missile_total_time:.2f} s")

# 扩大时间范围以支持接力策略
max_time = min(missile_total_time + 10, 80)  # 导弹飞行时间 + 缓冲
tau = np.arange(0, max_time + dt, dt)

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

# ============== 接力遮蔽策略的适应度函数 ==============
def fitness_relay_strategy(x):
    """
    接力遮蔽策略适应度函数
    关键思路：三个烟幕弹按时间顺序接力，每个都有20秒有效期
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
        
        # 严格的约束检查
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
    
    # 生成目标点（严格按题目要求）
    num_cp, num_h = 8, 5
    theta_c = np.linspace(0, 2*np.pi, num_cp, endpoint=False)
    h_levels = np.linspace(0, target_h, num_h)
    target_p = []
    for h in h_levels:
        for ang in theta_c:
            target_p.append([target_r*np.cos(ang),
                             target_r*np.sin(ang)+target_c[1],
                             target_c[2]+h])
    target_p = np.array(target_p)
    
    # ============== 关键改进：接力遮蔽判定 ==============
    is_shielded = np.zeros_like(tau, dtype=bool)
    
    for i, t_global in enumerate(tau):
        # 检查导弹是否还在飞行
        if t_global > missile_total_time:
            break
            
        # 收集当前时刻所有有效的烟幕云团
        active_smokes = []
        
        for k in range(3):
            # 烟幕弹起爆时间
            smoke_start_time = uav_params[k]['t_d'] + uav_params[k]['t_b']
            smoke_end_time = smoke_start_time + T_eff
            
            # 检查烟幕弹是否在有效期内
            if smoke_start_time <= t_global <= smoke_end_time:
                # 计算烟幕云团当前位置（考虑下沉）
                t_since_burst = t_global - smoke_start_time
                sp = bp[k] + np.array([0, 0, -sink_v * t_since_burst])
                active_smokes.append(sp)
        
        if not active_smokes:
            is_shielded[i] = False
            continue
            
        # 导弹当前位置
        mp = m1_0 + v_m1v * t_global
        
        # 严格遮蔽判定：必须所有目标点都被遮蔽
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
            if s < len(tau) and e < len(tau):
                intervals.append((tau[s], tau[e]))
    
    return total_time, intervals, dropp, bp

# ============== 专门针对接力策略的差分进化算法 ==============
def differential_evolution_relay(pop_size=60, gens=200):
    """
    专门针对接力策略的差分进化算法
    重点：让三个烟幕弹在时间上形成接力
    """
    print("=== 接力策略差分进化算法 ===")
    print(f"目标：实现三个烟幕弹的时间接力，每个有效期{T_eff}秒")
    
    dim = 12
    lb = np.array([-np.pi, 70, 0, 0] * 3)
    ub = np.array([np.pi, 140, 30, 15] * 3)  # 扩大时间搜索范围
    
    # 接力策略的智能初始化
    Pop = []
    
    # 策略1：时间错开接力 - 核心策略
    for _ in range(pop_size // 2):
        x = []
        
        # 设计三个时间段：早期、中期、后期
        base_times = [5, 20, 35]  # 基础时间点
        np.random.shuffle(base_times)
        
        for i in range(3):
            # 计算该时段的导弹位置，设计拦截策略
            target_time = base_times[i]
            future_missile_pos = m1_0 + v_m1v * target_time
            
            # 从无人机到拦截点的方向
            intercept_dir = future_missile_pos - uav_positions[i]
            base_angle = np.arctan2(intercept_dir[1], intercept_dir[0])
            
            theta = base_angle + np.random.normal(0, 0.2)
            v_u = np.random.uniform(90, 130)
            
            # 计算合适的投放时间和起爆延迟
            dist = np.linalg.norm(intercept_dir)
            flight_time = dist / v_u
            
            # 让烟幕弹在目标时间前起爆
            desired_burst_time = target_time - 2
            t_d = max(0, desired_burst_time - flight_time - np.random.uniform(2, 8))
            t_b = max(0.5, desired_burst_time - t_d)
            
            x.extend([theta, v_u, t_d, t_b])
        
        Pop.append(np.array(x))
    
    # 策略2：连续接力策略
    for _ in range(pop_size // 4):
        x = []
        
        # 让三个烟幕弹形成连续的时间覆盖
        start_times = [2, 18, 34]  # 每个间隔16秒，形成部分重叠
        
        for i in range(3):
            target_time = start_times[i] + np.random.uniform(-2, 2)
            
            # 朝向目标区域
            target_dir = target_c - uav_positions[i] 
            base_angle = np.arctan2(target_dir[1], target_dir[0])
            
            theta = base_angle + np.random.normal(0, 0.3)
            v_u = np.random.uniform(80, 130)
            t_d = max(0, target_time - np.random.uniform(3, 10))
            t_b = max(0.5, target_time - t_d)
            
            x.extend([theta, v_u, t_d, t_b])
        
        Pop.append(np.array(x))
    
    # 策略3：随机探索
    for _ in range(pop_size - pop_size//2 - pop_size//4):
        x = np.random.uniform(lb, ub)
        Pop.append(x)
    
    # 评估初始种群
    Fit = []
    print("评估初始种群...")
    for j, x in enumerate(Pop):
        if j % 15 == 0:
            print(f"  评估 {j+1}/{len(Pop)}")
        score, _, _, _ = fitness_relay_strategy(x)
        Fit.append(score)
    Fit = np.array(Fit)
    
    best_idx = np.argmax(Fit)
    best_vec, best_score = Pop[best_idx].copy(), Fit[best_idx]
    
    print(f"初始最佳遮蔽时间: {best_score:.4f} s")
    
    # 进化过程
    print("\n开始接力策略优化...")
    for g in range(gens):
        for i in range(pop_size):
            # 自适应参数
            F = 0.4 + 0.4 * np.random.random()
            Cr = 0.6 + 0.3 * np.random.random()
            
            # 差分进化操作
            idxs = [j for j in range(pop_size) if j != i]
            a, b, c = [Pop[j] for j in np.random.choice(idxs, 3, replace=False)]
            
            mutant = a + F * (b - c)
            mutant = np.clip(mutant, lb, ub)
            
            cross = np.array([mutant[j] if random.random() < Cr else Pop[i][j] for j in range(dim)])
            
            score, _, _, _ = fitness_relay_strategy(cross)
            
            if score > Fit[i]:
                Pop[i], Fit[i] = cross, score
                if score > best_score:
                    best_vec, best_score = cross, score
                    print(f"  🎉 新纪录！第{g+1}代: {best_score:.4f} s")
        
        if (g + 1) % 25 == 0:
            avg_score = np.mean(Fit)
            print(f"第{g+1:3d}代: 最佳={best_score:.4f} s, 平均={avg_score:.4f} s")
    
    return best_vec, best_score

# ============== 结果分析 ==============
def analyze_relay_result(vec):
    """分析接力策略的结果"""
    score, intervals, dropp, bp = fitness_relay_strategy(vec)
    
    print(f"\n=== 接力策略结果分析 ===")
    print(f"总遮蔽时间: {score:.4f} s")
    print(f"遮蔽区间数: {len(intervals)}")
    
    # 分析各烟幕弹的时间安排
    smoke_schedules = []
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
        start_time = t_d + t_b
        end_time = start_time + T_eff
        
        smoke_schedules.append({
            'uav': uav_names[i],
            'start': start_time,
            'end': end_time,
            'theta': theta,
            'v_u': v_u,
            't_d': t_d,
            't_b': t_b
        })
        
        print(f"\n{uav_names[i]}:")
        print(f"  飞行角度: {np.degrees(theta):.2f}°")
        print(f"  飞行速度: {v_u:.2f} m/s")
        print(f"  投放时间: {t_d:.4f} s")
        print(f"  起爆延迟: {t_b:.4f} s")
        print(f"  烟幕有效期: [{start_time:.4f}, {end_time:.4f}] s (时长{T_eff}s)")
        print(f"  投放点: [{dropp[i][0]:.2f}, {dropp[i][1]:.2f}, {dropp[i][2]:.2f}]")
        print(f"  起爆点: [{bp[i][0]:.2f}, {bp[i][1]:.2f}, {bp[i][2]:.2f}]")
    
    # 按时间排序显示接力情况
    smoke_schedules.sort(key=lambda x: x['start'])
    print(f"\n⏰ 时间接力安排:")
    for i, schedule in enumerate(smoke_schedules):
        print(f"  {i+1}. {schedule['uav']}: [{schedule['start']:.2f}, {schedule['end']:.2f}] s")
    
    print(f"\n🎯 遮蔽时间区间:")
    total_interval_time = 0
    for j, (start, end) in enumerate(intervals):
        duration = end - start
        total_interval_time += duration
        print(f"  区间{j+1}: [{start:.4f}, {end:.4f}] s, 时长={duration:.4f} s")
    
    print(f"\n📊 效果统计:")
    print(f"  理论最大遮蔽时间: {3 * T_eff} s (3 × {T_eff}s)")
    print(f"  实际遮蔽时间: {score:.4f} s")
    print(f"  效率: {score/(3*T_eff)*100:.1f}%")
    
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
    print(f"\n✅ 接力策略结果已保存到 result2.xlsx")
    
    return score

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== 第4问：三无人机时间接力遮蔽策略 ====")
    print("核心思路：三个烟幕弹按时间顺序接力，每个有效期20秒")
    print(f"理论最大遮蔽时间：{3 * T_eff} s")
    
    # 运行接力策略优化
    best_vec, best_score = differential_evolution_relay(pop_size=80, gens=250)
    
    # 分析结果
    final_score = analyze_relay_result(best_vec)
    
    print(f"\n🏆 最终结果:")
    print(f"接力策略遮蔽时间: {final_score:.4f} s")
    
    if final_score > 10:
        print(f"🎉 成功超过10秒目标！")
    else:
        print(f"距离10秒目标: {10 - final_score:.2f} s")
    
    print("\n这次应该能看到真正的多无人机接力协同效果！")