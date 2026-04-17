function [q1_solutions, q2_solutions, info] = ik_q1q2_new(T, p)
%IK_Q1_Q2_NEW 已知 T 点坐标，解析求解 q1、q2
%   输入：T - 目标点坐标 [Tx, Ty, Tz]
%         p - 参数结构体（可选，来自 params()）
%   输出：q1、q2 是相对于工作零位的变化量（工作零位时 q1=q2=0）

    % ========== 加载参数 ==========
    P = p.P;
    PM = p.L_PM;
    MN = p.L_MN;
    NT = p.L_NT;
    
    % ========== 初始偏置 ==========
    theta_init = -5.41;  % 工作零位相对于几何零位的偏置
    
    % ========== 第一步：求解 q3、q4 ==========
    [q3, info_q3] = ik_q3_from_PT(P, T, p, 0);
    q4 = q3_to_q4(q3, p);
    
    if ~info_q3.success
        info.success = false;
        info.error = 'q3 求解失败';
        q1_solutions = [];
        q2_solutions = [];
        return;
    end

    [T0, ~, info] = fk_T0(q3, p);
    [q1_deg, q2_deg, error] = ik_solve_q1q2_byT0(P, T0, T);
           
    % ========== 信息输出 ==========
    q1_solutions = q1_deg;
    q2_solutions = q2_deg;
    info.success = true;
    info.q3 = q3;
    info.q4 = q4;
    info.num_solutions = 1;
end
