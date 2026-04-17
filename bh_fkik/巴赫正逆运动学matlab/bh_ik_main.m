function [q_all, status, info] = bh_ik_main(P, p)
% BH_IK_MAIN 三自由度灵巧手逆运动学解算

    x = P(1); y = P(2); z = P(3);
    q_all = [];
    status = 0;
    
    % 初始化结构体
    info = struct();
    info.all_solutions = {};
    info.valid_solutions = {};
    info.distance = 0;
    info.num_solutions = 0;
    info.num_valid = 0;
    
    % 容差参数
    limit_tol = 0.5;
    pos_tol = 1e-4;
    
    % 1. 计算距离相关变量
    r = sqrt(x^2 + y^2); 
    D_sq = x^2 + y^2 + z^2; 
    D = sqrt(D_sq);
    info.distance = D;
    
    % 工作空间判断
    if D > p.D_max + pos_tol || D < p.D_min - pos_tol
        warning('目标点超出工作空间范围！D = %.2f', D);
        return;
    end
    
    % 2. 求解 q3（余弦定理）
    cos_theta3 = (D_sq - p.L1^2 - p.L2^2) / (2 * p.L1 * p.L2);
    
    if cos_theta3 > 1, cos_theta3 = 1; end
    if cos_theta3 < -1, cos_theta3 = -1; end
    
    theta3_rad_1 = acos(cos_theta3);
    theta3_rad_2 = -acos(cos_theta3);
    
    theta3_list = {theta3_rad_1, theta3_rad_2};
    
    % 3. 求解 q1
    q1_rad_base = atan2(x, -y);
    q1_list = {q1_rad_base, q1_rad_base + pi};
    
    % 4. 遍历所有组合：2 个 theta3 × 2 个 term_K 符号 = 4 组解
    solutions = {};
    sol_idx = 1;
    
    for i = 1:2
        theta3_rad = theta3_list{i};
        q3_deg = rad2deg(theta3_rad) - p.offset_deg;
        
        A = p.L1 + p.L2 * cos(theta3_rad);
        B = p.L2 * sin(theta3_rad);
        denom = A^2 + B^2;
        
        for sign_K = [1, -1]
            r_signed = sign_K * r;
            
            sin_q2 = (A * r_signed - B * z) / denom;
            cos_q2 = (B * r_signed + A * z) / denom;
            
            sin_q2 = max(min(sin_q2, 1), -1);
            cos_q2 = max(min(cos_q2, 1), -1);
            
            q2_rad = atan2(sin_q2, cos_q2);
            q2_deg = rad2deg(q2_rad);
            
            % 计算 q1
            if sign_K > 0
                q1_rad = atan2(x, -y);
            else
                q1_rad = atan2(-x, y);
            end
            
            q1_deg = rad2deg(q1_rad);
            
            % 【修复】q1 归一化到 (-180, 180]，允许 -180
            while q1_deg > 180, q1_deg = q1_deg - 360; end
            while q1_deg < -180, q1_deg = q1_deg + 360; end  % 改为 <
            
            % 【新增】处理边界等价：-180 和 180 物理相同，统一输出 180
            if q1_deg < -179.999
                q1_deg = 180;
            end
            
            solutions{sol_idx} = [q1_deg, q2_deg, q3_deg];
            sol_idx = sol_idx + 1;
        end
    end
    
    info.all_solutions = solutions;
    info.num_solutions = length(solutions);
    
    % 5. 筛选在限位内的解
    valid_solutions = {};
    for i = 1:length(solutions)
        q = solutions{i};
        
        % 【修复】限位检查使用容差，支持边界值
        in_limit = (q(1) >= p.limit.q1(1) - limit_tol && q(1) <= p.limit.q1(2) + limit_tol) && ...
                   (q(2) >= p.limit.q2(1) - limit_tol && q(2) <= p.limit.q2(2) + limit_tol) && ...
                   (q(3) >= p.limit.q3(1) - limit_tol && q(3) <= p.limit.q3(2) + limit_tol);
        
        if in_limit
            % 【修复】截断到限位边界，但不改变有效值
            q(1) = max(min(q(1), p.limit.q1(2)), p.limit.q1(1));
            q(2) = max(min(q(2), p.limit.q2(2)), p.limit.q2(1));
            q(3) = max(min(q(3), p.limit.q3(2)), p.limit.q3(1));
            valid_solutions{end+1} = q;
        end
    end
    
    info.valid_solutions = valid_solutions;
    info.num_valid = length(valid_solutions);
    
    % 6. 返回结果
    if ~isempty(valid_solutions)
        % 【修复】去重：移除三角函数等价的重复解
        valid_solutions = remove_duplicate_solutions(valid_solutions, p);
        
        if length(valid_solutions) == 1
            q_all = valid_solutions{1};
            q_all = reshape(q_all, 1, 3);
        else
            q_all = cell2mat(valid_solutions');
        end
        status = 1;
        fprintf('找到 %d 组有效解\n', info.num_valid);
    else
        warning('所有解均超出限位范围！');
        if ~isempty(solutions)
            q_all = solutions{1};
            q_all = reshape(q_all, 1, 3);
        end
        status = 0;
    end
end

% 【新增】去重函数：移除三角函数等价的重复解
function valid_solutions = remove_duplicate_solutions(valid_solutions, p)
    if length(valid_solutions) <= 1
        return;
    end
    
    unique_solutions = {};
    for i = 1:length(valid_solutions)
        q = valid_solutions{i};
        is_duplicate = false;
        
        for j = 1:length(unique_solutions)
            q_ref = unique_solutions{j};
            
            % 检查 q1 是否相差 360° 的整数倍
            q1_diff = abs(q(1) - q_ref(1));
            q1_same = (q1_diff < 0.01) || (abs(q1_diff - 360) < 0.01);
            
            % 检查 q2, q3 是否相同
            q2_same = abs(q(2) - q_ref(2)) < 0.01;
            q3_same = abs(q(3) - q_ref(3)) < 0.01;
            
            if q1_same && q2_same && q3_same
                is_duplicate = true;
                break;
            end
        end
        
        if ~is_duplicate
            unique_solutions{end+1} = q;
        end
    end
    
    valid_solutions = unique_solutions;
end
