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
T_eff = 20.0  # 每个烟幕弹的有效时间
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
max_time = min(missile_total_time + 10, 80)
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

# ============== 严格的适应度函数 ==============
def fitness_strict_conditions(x):
    """严格按照题目条件的适应度函数"""
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
    
    # 严格的遮蔽判定
    is_shielded = np.zeros_like(tau, dtype=bool)
    
    for i, t_global in enumerate(tau):
        if t_global > missile_total_time:
            break
            
        # 收集当前时刻所有有效的烟幕云团
        active_smokes = []
        
        for k in range(3):
            smoke_start_time = uav_params[k]['t_d'] + uav_params[k]['t_b']
            smoke_end_time = smoke_start_time + T_eff
            
            if smoke_start_time <= t_global <= smoke_end_time:
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

# ============== 用户初解输入函数 ==============
def get_user_initial_solution():
    """获取用户提供的初解"""
    print("=== 请输入初解参数 ===")
    print("每架无人机需要4个参数：飞行角度(度), 飞行速度(m/s), 投放时间(s), 起爆延迟(s)")
    print("约束条件：")
    print("  - 飞行速度：70-140 m/s")
    print("  - 投放时间：≥ 0 s")
    print("  - 起爆延迟：≥ 0 s")
    print()
    
    solution = []
    
    for i, uav_name in enumerate(uav_names):
        print(f"--- {uav_name} (初始位置: {uav_positions[i]}) ---")
        
        while True:
            try:
                angle_deg = float(input(f"  飞行角度 (度, -180到180): "))
                if -180 <= angle_deg <= 180:
                    theta = np.radians(angle_deg)
                    break
                else:
                    print("  角度必须在-180到180度之间")
            except ValueError:
                print("  请输入有效的数字")
        
        while True:
            try:
                v_u = float(input(f"  飞行速度 (m/s, 70-140): "))
                if 70 <= v_u <= 140:
                    break
                else:
                    print("  速度必须在70-140 m/s之间")
            except ValueError:
                print("  请输入有效的数字")
        
        while True:
            try:
                t_d = float(input(f"  投放时间 (s, ≥0): "))
                if t_d >= 0:
                    break
                else:
                    print("  投放时间必须≥0")
            except ValueError:
                print("  请输入有效的数字")
        
        while True:
            try:
                t_b = float(input(f"  起爆延迟 (s, ≥0): "))
                if t_b >= 0:
                    break
                else:
                    print("  起爆延迟必须≥0")
            except ValueError:
                print("  请输入有效的数字")
        
        solution.extend([theta, v_u, t_d, t_b])
        print()
    
    return np.array(solution)

def parse_parameter_string(param_str):
    """解析参数字符串"""
    try:
        # 移除多余的空格和换行符
        param_str = param_str.strip().replace('\n', ' ')
        
        # 分割参数
        params = []
        for part in param_str.split():
            try:
                params.append(float(part))
            except ValueError:
                continue
        
        if len(params) != 12:
            return None
        
        # 转换角度为弧度（假设输入的是度）
        solution = []
        for i in range(3):
            theta_deg = params[4*i]
            theta_rad = np.radians(theta_deg)
            v_u = params[4*i + 1]
            t_d = params[4*i + 2]
            t_b = params[4*i + 3]
            
            # 检查约束
            if not (70 <= v_u <= 140 and t_d >= 0 and t_b >= 0):
                return None
            
            solution.extend([theta_rad, v_u, t_d, t_b])
        
        return np.array(solution)
    except:
        return None

def get_user_initial_solution_batch():
    """批量输入初解"""
    print("=== 批量输入初解 ===")
    print("请输入12个参数，按以下顺序：")
    print("FY1: 角度(度) 速度(m/s) 投放时间(s) 起爆延迟(s)")
    print("FY2: 角度(度) 速度(m/s) 投放时间(s) 起爆延迟(s)")
    print("FY3: 角度(度) 速度(m/s) 投放时间(s) 起爆延迟(s)")
    print("可以用空格或换行分隔，例如：")
    print("0 120 1 2  90 100 3 4  -90 110 5 6")
    print()
    
    while True:
        param_str = input("请输入参数: ")
        solution = parse_parameter_string(param_str)
        
        if solution is not None:
            return solution
        else:
            print("输入格式错误或参数不满足约束条件，请重新输入")

# ============== 基于初解的优化算法 ==============
def optimize_from_initial_solution(initial_solution, pop_size=40, gens=100):
    """基于用户初解进行优化"""
    print(f"\n=== 基于初解进行优化 ===")
    
    # 首先评估初解
    init_score, init_intervals, init_dropp, init_bp = fitness_strict_conditions(initial_solution)
    print(f"初解遮蔽时间: {init_score:.4f} s")
    
    dim = 12
    lb = np.array([-np.pi, 70, 0, 0] * 3)
    ub = np.array([np.pi, 140, 30, 15] * 3)
    
    # 初始化种群：围绕用户初解
    Pop = []
    
    # 第一个个体就是用户初解
    Pop.append(initial_solution.copy())
    
    # 其余个体在初解附近扰动
    for _ in range(pop_size - 1):
        # 在初解基础上添加扰动
        perturbation = np.random.normal(0, 0.1, dim)  # 小幅扰动
        
        # 对不同参数使用不同的扰动幅度
        for i in range(3):
            perturbation[4*i] *= 0.2      # 角度扰动小一些
            perturbation[4*i + 1] *= 5    # 速度扰动
            perturbation[4*i + 2] *= 2    # 投放时间扰动
            perturbation[4*i + 3] *= 1    # 起爆延迟扰动
        
        candidate = initial_solution + perturbation
        candidate = np.clip(candidate, lb, ub)
        Pop.append(candidate)
    
    # 评估初始种群
    Fit = []
    print("评估初始种群...")
    for j, x in enumerate(Pop):
        if j % 10 == 0:
            print(f"  评估 {j+1}/{len(Pop)}")
        score, _, _, _ = fitness_strict_conditions(x)
        Fit.append(score)
    Fit = np.array(Fit)
    
    best_idx = np.argmax(Fit)
    best_vec, best_score = Pop[best_idx].copy(), Fit[best_idx]
    
    print(f"种群中最佳: {best_score:.4f} s")
    
    # 进化过程
    print("\n开始优化...")
    for g in range(gens):
        for i in range(pop_size):
            # 自适应参数
            F = 0.3 + 0.4 * np.random.random()
            Cr = 0.6 + 0.3 * np.random.random()
            
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
                    print(f"  🎉 新纪录！第{g+1}代: {best_score:.4f} s")
        
        if (g + 1) % 20 == 0:
            avg_score = np.mean(Fit)
            print(f"第{g+1:3d}代: 最佳={best_score:.4f} s, 平均={avg_score:.4f} s")
    
    return best_vec, best_score

# ============== 结果分析 ==============
def analyze_solution(vec, title="解决方案分析"):
    """分析解决方案"""
    score, intervals, dropp, bp = fitness_strict_conditions(vec)
    
    print(f"\n=== {title} ===")
    print(f"总遮蔽时间: {score:.4f} s")
    print(f"遮蔽区间数: {len(intervals)}")
    
    # 分析各无人机
    for i in range(3):
        theta = vec[4*i]
        v_u = vec[4*i + 1]
        t_d = vec[4*i + 2]
        t_b = vec[4*i + 3]
        
        start_time = t_d + t_b
        end_time = start_time + T_eff
        
        print(f"\n{uav_names[i]}:")
        print(f"  飞行角度: {np.degrees(theta):.2f}°")
        print(f"  飞行速度: {v_u:.2f} m/s")
        print(f"  投放时间: {t_d:.4f} s")
        print(f"  起爆延迟: {t_b:.4f} s")
        print(f"  烟幕有效期: [{start_time:.4f}, {end_time:.4f}] s")
        print(f"  投放点: [{dropp[i][0]:.2f}, {dropp[i][1]:.2f}, {dropp[i][2]:.2f}]")
        print(f"  起爆点: [{bp[i][0]:.2f}, {bp[i][1]:.2f}, {bp[i][2]:.2f}]")
    
    print(f"\n遮蔽时间区间:")
    for j, (start, end) in enumerate(intervals):
        print(f"  区间{j+1}: [{start:.4f}, {end:.4f}] s, 时长={end-start:.4f} s")
    
    return score, intervals, dropp, bp

def save_solution(vec, filename="result2.xlsx"):
    """保存解决方案"""
    score, intervals, dropp, bp = fitness_strict_conditions(vec)
    
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
    df.to_excel(filename, index=False)
    print(f"\n✅ 结果已保存到 {filename}")

# ============== 主程序 ==============
if __name__ == "__main__":
    print("==== 第4问：基于用户初解的优化 ====")
    print("您可以提供一个初解，程序将在此基础上进行优化")
    
    # 选择输入方式
    print("\n请选择输入方式：")
    print("1. 逐个参数输入（详细）")
    print("2. 批量输入（快速）")
    
    while True:
        choice = input("请选择 (1 或 2): ").strip()
        if choice in ['1', '2']:
            break
        print("请输入1或2")
    
    if choice == '1':
        initial_solution = get_user_initial_solution()
    else:
        initial_solution = get_user_initial_solution_batch()
    
    # 分析初解
    init_score, _, _, _ = analyze_solution(initial_solution, "初解分析")
    
    # 询问是否进行优化
    print(f"\n当前初解遮蔽时间: {init_score:.4f} s")
    optimize_choice = input("是否基于此初解进行优化？(y/n): ").strip().lower()
    
    if optimize_choice == 'y':
        # 基于初解优化
        best_vec, best_score = optimize_from_initial_solution(initial_solution)
        
        # 分析优化结果
        final_score, _, _, _ = analyze_solution(best_vec, "优化后结果")
        
        print(f"\n📊 对比结果:")
        print(f"初解遮蔽时间: {init_score:.4f} s")
        print(f"优化后遮蔽时间: {final_score:.4f} s")
        print(f"改进: {final_score - init_score:.4f} s")
        
        # 保存结果
        save_solution(best_vec)
    else:
        # 只分析初解
        save_solution(initial_solution)
        print("已保存初解结果")
    
    print("\n程序完成！")