function [q3, q4, info] = thumb_ik_q3q4(P_target, q1, q2)
    % THUMB_IK_Q3Q4 求解拇指机构的 q3 和 q4（平面内 2R 逆解）
    %
    % 功能：在已知 q1、q2 的基础上，求解平面内两关节的角度
    % 方法：几何法解析解（余弦定理 + 三角关系）
    %
    % 输入:
    %   P_target : 3x1 向量，末端目标位置 [x; y; z] (mm)
    %   q1       : 关节 1 角度 (度)，已由 thumb_ik_q1q2 求得
    %   q2       : 关节 2 角度 (度)，已由 thumb_ik_q1q2 求得
    %
    % 输出:
    %   q3       : 关节 3 角度 (度)
    %   q4       : 关节 4 角度 (度)
    %   info     : 结构体，包含求解状态和中间信息
    %              info.status       : 求解状态 (0=成功，1=无解，2=多解/警告)
    %              info.error_msg    : 错误/状态信息
    %              info.P_M          : M 点位置
    %              info.P_N          : N 点位置
    %              info.d_x, d_y, d_z: MT 向量在{1}系中的坐标
    %              info.r            : MT 在平面内的投影长度
    %              info.q3_all       : 所有可行的 q3 解
    %              info.q4_all       : 所有可行的 q4 解
    %              info.limit_exceeded : 关节限位超限标志
    %              info.exceed_joint   : 超限关节名称列表
    %
    % 示例:
    %   [q3, q4, info] = thumb_ik_q3q4([60; 15; 10], -1.85, -12.13);
    %
    % 数学原理:
    %   1. 计算 MT 距离 r 和方向角 β
    %   2. 余弦定理求 q4（注意是外角）
    %   3. 余弦定理求∠NMT
    %   4. q3 = β ∓ ∠NMT - α2

    %% 1. 参数定义
    % 从参数文件获取结构常数和配置参数
    p = params();
    
    alpha1 = p.alpha1;       % 基座固定偏角 (度)
    alpha2 = p.alpha2;       % M 系固定偏角 (度)
    alpha3 = p.alpha3;       % N 系固定偏角 (度)
    
    L1 = p.L1;               % OM 长度 (mm)
    L2 = p.L2;               % MN 长度 (mm)
    L3 = p.L3;               % NT 长度 (mm)
    
    % 关节限位
    joint_limits = [p.q3_limit_min, p.q3_limit_max;   % q3 限位
                    p.q4_limit_min, p.q4_limit_max];  % q4 限位
    limit_tolerance = p.limit_tolerance;  % 限位检查容差 (度)

    %% 2. 初始化输出
    q3 = [];
    q4 = [];
    info = struct();
    info.status = 0;                % 0=求解成功
    info.error_msg = '';            % 错误信息
    info.P_M = [];                  % M 点位置
    info.P_N = [];                  % N 点位置
    info.d_x = [];                  % MT 在{1}系 X 分量
    info.d_z = [];                  % MT 在{1}系 Z 分量
    info.r = [];                    % MT 在平面内投影长度
    info.q3_all = [];               % 所有可行的 q3 解
    info.q4_all = [];               % 所有可行的 q4 解
    info.limit_exceeded = false;    % 关节限位超限标志
    info.exceed_joint = {};         % 超限关节名称列表

    %% 3. 输入验证
    if nargin < 3
        info.status = 1;
        info.error_msg = '输入参数不足：需要 P_target, q1, q2';
        return;
    end

    if isempty(q1) || isempty(q2)
        info.status = 1;
        info.error_msg = 'q1 或 q2 为空，请先求解 q1, q2';
        return;
    end

    %% 4. 角度预处理
    deg2rad = pi / 180.0;
    alpha1_rad = alpha1 * deg2rad;
    alpha2_rad = alpha2 * deg2rad;
    alpha3_rad = alpha3 * deg2rad;

    %% 5. 计算基座旋转矩阵 R_01
    % R_01 描述{1}系相对于{0}系的姿态
    % 由三个旋转组成：Rx(q1) * Rz(q2) * Ry(alpha1)
    c1 = cos(q1 * deg2rad);
    s1 = sin(q1 * deg2rad);
    c2 = cos(q2 * deg2rad);
    s2 = sin(q2 * deg2rad);
    ca1 = cos(alpha1_rad);
    sa1 = sin(alpha1_rad);

    R_01 = [c2*ca1, -s2, c2*sa1;
            c1*s2*ca1 + s1*sa1, c1*c2, c1*s2*sa1 - s1*ca1;
            s1*s2*ca1 - c1*sa1, s1*c2, s1*s2*sa1 + c1*ca1];

    %% 6. 计算 M 点位置
    % M 点是 OM 向量的终点，在{1}系中为 [0; 0; L1]
    info.P_M = R_01 * [0; 0; L1];

    %% 7. 计算 MT 向量在{1}系中的坐标
    % MT = T - M，然后变换到{1}系
    MT = P_target - info.P_M;
    MT_local = R_01' * MT;  % R_01' 是逆变换

    info.d_x = MT_local(1);  % {1}系 X 分量
    info.d_y = MT_local(2);  % {1}系 Y 分量 (理论应为 0)
    info.d_z = MT_local(3);  % {1}系 Z 分量

    % d_y 验证：检查 q1/q2 求解是否正确
    % 如果 |d_y| > 容差，说明 q1/q2 求解存在误差
    if abs(info.d_y) > p.dy_tolerance
        warning('thumb_ik_q3q4: d_y=%.6f，q1/q2 可能存在误差', info.d_y);
    end

    % MT 在平面内的投影长度 (忽略 d_y)
    info.r = sqrt(info.d_x^2 + info.d_z^2);

    %% 8. 检查可达性
    % 计算工作空间边界
    r_min = abs(L2 - L3);  % 最小可达距离 (两杆重叠)
    r_max = L2 + L3;       % 最大可达距离 (两杆伸直)

    if info.r > r_max
        info.status = 1;
        info.error_msg = sprintf('目标点超出最大工作空间 (距离=%.2fmm > 最大=%.2fmm)', ...
            info.r, r_max);
        return;
    end

    if info.r < r_min
        info.status = 1;
        info.error_msg = sprintf('目标点超出最小工作空间 (距离=%.2fmm < 最小=%.2fmm)', ...
            info.r, r_min);
        return;
    end

    %% 9. 解析求解 q3, q4
    % 使用几何法：将问题分解为平面内 2R 机械臂逆解
    
    q3_solutions = [];
    q4_solutions = [];

    % ----- 步骤 1: 用余弦定理求关节 4 的角度 -----
    % 在三角形 MNT 中，已知三边 L2, L3, r
    % cos(θ4) = (r² - L2² - L3²) / (2·L2·L3)
    % θ4 是外角，q4 = θ4 - α3
    cos_theta4 = (info.r^2 - L2^2 - L3^2) / (2 * L2 * L3);
    
    % 限制在 [-1, 1] 范围内，避免数值误差
    cos_theta4 = max(-1, min(1, cos_theta4));
    
    % 两个可能的 θ4 解 (正负肘部配置)
    theta4_solutions = [acos(cos_theta4); -acos(cos_theta4)];

    % ----- 步骤 2: 计算∠NMT -----
    % cos(∠NMT) = (L2² + r² - L3²) / (2·L2·r)
    cos_NMT = (L2^2 + info.r^2 - L3^2) / (2 * L2 * info.r);
    cos_NMT = max(-1, min(1, cos_NMT));
    angle_NMT = acos(cos_NMT);

    % ----- 步骤 3: 计算 MT 的方向角 β -----
    % β = atan2(d_x, d_z)，在{1}系 X-Z 平面内
    beta = atan2(info.d_x, info.d_z);

    % ----- 步骤 4: 遍历所有θ4 解，计算对应的 q3, q4 -----
    for i = 1:length(theta4_solutions)
        theta4 = theta4_solutions(i);

        % q4 = θ4 - α3 (减去固定偏角)
        q4_candidate = theta4 / deg2rad - alpha3;

        % 根据θ4 的正负确定θ3 的计算方式
        % θ4 >= 0: "肘部向下" 配置，θ3 = β - ∠NMT
        % θ4 < 0:  "肘部向上" 配置，θ3 = β + ∠NMT
        if theta4 >= 0
            theta3 = beta - angle_NMT;
        else
            theta3 = beta + angle_NMT;
        end

        % q3 = θ3 - α2 (减去固定偏角)
        q3_candidate = theta3 / deg2rad - alpha2;

        % 关节限位检查 (含容差)
        if q3_candidate >= joint_limits(1,1) - limit_tolerance && ...
           q3_candidate <= joint_limits(1,2) + limit_tolerance && ...
           q4_candidate >= joint_limits(2,1) - limit_tolerance && ...
           q4_candidate <= joint_limits(2,2) + limit_tolerance
            q3_solutions = [q3_solutions; q3_candidate];
            q4_solutions = [q4_solutions; q4_candidate];
        end
    end

    %% 10. 处理求解结果
    if isempty(q3_solutions)
        info.status = 1;
        info.error_msg = '无解：所有候选解均超出关节限位';
        
        % 调试信息输出
        fprintf('\n[调试] q3/q4 无可行解\n');
        fprintf('  r = %.4f mm, 可达范围 [%.4f, %.4f] mm\n', info.r, r_min, r_max);
        fprintf('  beta = %.4f°, angle_NMT = %.4f°\n', beta*180/pi, angle_NMT*180/pi);
        fprintf('  d_x = %.4f mm, d_z = %.4f mm\n', info.d_x, info.d_z);
        for i = 1:length(theta4_solutions)
            theta4 = theta4_solutions(i);
            q4_cand = theta4 / deg2rad - alpha3;
            if theta4 >= 0
                theta3 = beta - angle_NMT;
            else
                theta3 = beta + angle_NMT;
            end
            q3_cand = theta3 / deg2rad - alpha2;
            fprintf('  候选解 %d: q3=%.4f°, q4=%.4f°\n', i, q3_cand, q4_cand);
        end
        return;
    end

    info.q3_all = q3_solutions;
    info.q4_all = q4_solutions;

    % 选择最优解
    if length(info.q3_all) > 1
        % 多解时选择关节角绝对值和最小的解
        info.status = 2;
        info.error_msg = sprintf('多解：共%d组可行解，已选择最优解', length(info.q3_all));
        cost = abs(info.q3_all) + abs(info.q4_all);
        [~, best_idx] = min(cost);
        q3 = info.q3_all(best_idx);
        q4 = info.q4_all(best_idx);
    else
        info.status = 0;
        info.error_msg = '求解成功';
        q3 = info.q3_all(1);
        q4 = info.q4_all(1);
    end

    % 检查最终解是否超限 (不含容差)
    info.exceed_joint = {};
    if q3 < joint_limits(1,1) || q3 > joint_limits(1,2)
        info.exceed_joint{end+1} = 'q3';
    end
    if q4 < joint_limits(2,1) || q4 > joint_limits(2,2)
        info.exceed_joint{end+1} = 'q4';
    end

    if ~isempty(info.exceed_joint)
        info.limit_exceeded = true;
        info.status = 2;
        info.error_msg = sprintf('⚠ 警告：关节 %s 超出限位', strjoin(info.exceed_joint, ', '));
    end

    %% 11. 计算 N 点位置
    % N 点是 OM + MN 向量的终点
    theta3 = (alpha2 + q3) * deg2rad;
    v2_local = [L2 * sin(theta3); 0; L2 * cos(theta3)];
    info.P_N = R_01 * ([0; 0; L1] + v2_local);
end
