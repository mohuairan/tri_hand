function [q3, info] = ik_q3_from_PT(P, T, p, q3_init)
%IK_Q3_FROM_PT 根据 P、T 三维坐标求解 q3
%   输入：
%       P, T     - 三维坐标 [x,y,z] (mm)
%       p        - 参数结构体 (来自 params())
%       q3_init  - 迭代初值（度），可选
%   输出：
%       q3       - 求解角度（度）
%       info     - 求解状态信息结构体
%                  .D_PT: P 到 T 的距离
%                  .residual: 最终残差
%                  .success: 求解是否成功
%                  .q4: 对应的 q4 角度

    %% ========== 参数验证与默认值 ==========
    if nargin < 3 || isempty(p)
    %isempty是MATLAB内置函数，用于检查输入参数是否为空数组。
        p = params();  % 加载默认参数
    end
    
    if nargin < 4 || isempty(q3_init)
        q3_init = 0;  % 默认初始值
    end
    
    %% ========== 从 p 读取参数 ==========
    L_PM = p.L_PM;
    L_MN = p.L_MN;
    L_NT = p.L_NT;
    THETA1_0 = p.theta3_init;    % M 点初始角
    THETA2_0 = p.theta4_init;    % N 点初始角
    
    %% ========== 目标距离 ==========
    D_PT = norm(T - P);
    
    %% ========== 使用匿名函数定义残差 ==========
    % 将参数显式传递给嵌套函数
    residual = @(q3_deg) compute_residual(q3_deg, L_PM, L_MN, L_NT, THETA1_0, THETA2_0, D_PT, p);
    %%匿名函数语法：f = @(输入参数) 函数体;y=@(x) 2*x
    %创建一个单变量函数 residual(q3_deg)
    %将其他参数（L_PM, L_MN等）捕获到函数内部
    
    %% ========== 求解 ==========
    options = optimoptions('fsolve', ...
        'Display', 'off', ...% 不显示迭代过程，保持命令行干净
        'FunctionTolerance', p.tolerance, ...% 函数值收敛容限
        'StepTolerance', p.tolerance);% 步长收敛容限
    
    [q3, fval, exitflag] = fsolve(residual, q3_init, options);
    %exitflag：fsolve的输出参数，表示求解状态。>0表示成功，<=0表示失败。
    %fval：残差函数的最终值，理想情况下应该接近0。
    
    %% ========== 信息结构体 ==========
    info.D_PT = D_PT;
    info.residual = fval;
    info.success = (exitflag > 0) && (abs(fval) < 1e-5);
    info.q4 = q3_to_q4(q3, p);
end

%% ========== 辅助函数：计算残差 ==========
function F = compute_residual(q3_deg, L_PM, L_MN, L_NT, THETA1_0, THETA2_0, D_PT, p)
    % 调用耦合函数求 q4
    q4_deg = q3_to_q4(q3_deg, p);
    
    % 检查 q4 是否有效
    if isnan(q4_deg)
        F = 1e10;  % 返回大值表示不可达
        %isnan函数用于检查输入参数是否为 NaN（Not a Number）。如果 q4_deg 是 NaN，说明求解失败或不可达，
        % 此时返回一个非常大的残差值（如 1e10）来指示这个情况。
        return;
    end
    
    % 计算绝对角度
    theta1 = THETA1_0 + q3_deg;
    theta2 = THETA2_0 + q4_deg;
    
    % 平面链长计算（局部坐标系）
    % x 方向：PM 沿 z 轴，MN 和 NT 在平面内投影
    x_end = L_PM + L_MN*cosd(theta1) + L_NT*cosd(theta1+theta2);
    y_end = L_MN*sind(theta1) + L_NT*sind(theta1+theta2);
    
    % 残差 = 计算长度 - 目标长度
    F = sqrt(x_end^2 + y_end^2) - D_PT;
end
