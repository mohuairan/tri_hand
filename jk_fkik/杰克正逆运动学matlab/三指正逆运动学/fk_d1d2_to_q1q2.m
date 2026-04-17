function [q1, q2, info] = fk_d1d2_to_q1q2(d1_init, delta_d1, d2_init, delta_d2, params)
%FK_D1D2_TO_Q1Q2 根据 d1、d2 变化量正向求解 q1、q2
%   输入：d1_init - 初始 d1 值 (mm)。fk_solve中计算后传入
%         delta_d1 - d1 变化量 (mm)
%         d2_init - 初始 d2 值 (mm)。fk_solve中计算后传入
%         delta_d2 - d2 变化量 (mm)
%         params - 参数结构体 (来自 ik_params())
%   输出：q1 - 屈伸角 (度)
%         q2 - 外展角 (度)
%         info - 详细信息结构体

    %% ========== 加载参数 ==========
    ax = params.ax; ay = params.ay;
    bx = params.bx; by = params.by; bz = params.bz;
    P = params.P; px = P(1); py = P(2); pz = P(3);
    l = params.l1;
    
    q1_min = -90; q1_max = 0;
    q2_min = -45; q2_max = 45;
    
    %% ========== 计算当前 d1、d2 绝对值 ==========
    d1 = d1_init + delta_d1;
    d2 = d2_init + delta_d2;
    
    C1 = [ax; ay; d1];
    C2 = [-ax; ay; d2];
    
    %% ========== 步骤 1: q2 初值估计 ==========
    sin_q2_approx = (d2 - d1) / (2 * bx);
    
    if abs(sin_q2_approx) > 1
        info.success = false;
        info.error = '无解：|d2-d1| > 2*bx';
        q1 = []; q2 = [];
        return;
    end
    
    q2_0 = rad2deg(asin(sin_q2_approx));
    
    if q2_0 < q2_min || q2_0 > q2_max
        info.success = false;
        info.error = sprintf('无解：q2=%.2f° 超出限位', q2_0);
        q1 = []; q2 = [];
        return;
    end
    
    %% ========== 步骤 2: 求解 q1 ==========
    function F = q1_equation(q1_deg)
        q1_rad = deg2rad(q1_deg);
        q2_rad = deg2rad(q2_0);
        
        c1 = cos(q1_rad); s1 = sin(q1_rad);
        c2 = cos(q2_rad); s2 = sin(q2_rad);
        
        K = s1 * by + c1 * bz;
        
        B1x = px + c2 * bx + s2 * K;
        B1y = py + c1 * by - s1 * bz;
        B1z = pz - s2 * bx + c2 * K;
        
        F = (B1x - C1(1))^2 + (B1y - C1(2))^2 + (B1z - C1(3))^2 - l^2;
    end
    
    % 检查端点
    F_min = q1_equation(q1_min);
    F_max = q1_equation(q1_max);
    
    options = optimset('Display', 'off', 'TolX', 1e-10);
    
    if abs(F_min) < 1e-8
        q1_sol = q1_min;
    elseif abs(F_max) < 1e-8
        q1_sol = q1_max;
    elseif F_min * F_max < 0
        [q1_sol, ~, exitflag] = fzero(@q1_equation, [q1_min, q1_max], options);
        if exitflag <= 0
            info.success = false;
            info.error = '无解：q1 求解失败';
            q1 = []; q2 = [];
            return;
        end
    else
        % 尝试 fsolve
        q1_guess = (q1_min + q1_max) / 2;
        options_fs = optimoptions('fsolve', 'Display', 'off', 'FunctionTolerance', 1e-12);
        [q1_sol, fval] = fsolve(@q1_equation, q1_guess, options_fs);
        if abs(fval) > 1e-6
            info.success = false;
            info.error = '无解：q1 方程无根';
            q1 = []; q2 = [];
            return;
        end
    end
    
    %% ========== 步骤 3: 双变量精化 ==========
    function F = residual(q)
        q1_rad = deg2rad(q(1));
        q2_rad = deg2rad(q(2));
        
        c1 = cos(q1_rad); s1 = sin(q1_rad);
        c2 = cos(q2_rad); s2 = sin(q2_rad);
        
        K = s1 * by + c1 * bz;
        
        B1x = px + c2 * bx + s2 * K;
        B1y = py + c1 * by - s1 * bz;
        B1z = pz - s2 * bx + c2 * K;
        
        B2x = px - c2 * bx + s2 * K;
        B2y = B1y;
        B2z = pz + s2 * bx + c2 * K;
        
        F(1) = (B1x - C1(1))^2 + (B1y - C1(2))^2 + (B1z - C1(3))^2 - l^2;
        F(2) = (B2x - C2(1))^2 + (B2y - C2(2))^2 + (B2z - C2(3))^2 - l^2;
    end
    
    q_init = [q1_sol, q2_0];
    options_fs = optimoptions('fsolve', 'Display', 'off', ...
        'FunctionTolerance', 1e-12, 'StepTolerance', 1e-12);
    
    [q_sol, fval, exitflag] = fsolve(@residual, q_init, options_fs);
    
    if exitflag <= 0 || norm(fval) > 1e-6
        info.success = false;
        info.error = '无解：优化不收敛';
        q1 = []; q2 = [];
        return;
    end
    
    %% ========== 步骤 4: 限位检查 ==========
    q1 = q_sol(1);
    q2 = q_sol(2);
    
    if q1 < q1_min || q1 > q1_max || q2 < q2_min || q2 > q2_max
        info.success = false;
        info.error = sprintf('无解：结果超出限位 (q1=%.2f°, q2=%.2f°)', q1, q2);
        %sprintf 函数的格式化输出控制符：之前没用sprintf是因为没有变量需要插入，打印字符串就行；
        % 现在需要打印变量，所以用sprintf先格式化字符串，再赋值给error。
        q1 = []; q2 = [];
        return;
    end
    
    info.success = true;
    info.q1 = q1;
    info.q2 = q2;
end
