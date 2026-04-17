function [T_pos, T_rot, info] = fk_chain_to_T(delta_d1, delta_d2, q3, params)
%FK_CHAIN_TO_T 串联机构正运动学：计算末端 T 的位姿
%   输入：delta_d1 - d1 变化量 (mm)
%         delta_d2 - d2 变化量 (mm)
%         q3 - 关节角 q3 (度)
%         params - 参数结构体 (来自 ik_params())
%   输出：T_pos - T 点位置 [Tx, Ty, Tz] (mm)
%         T_rot - T 点姿态矩阵 (3x3)
%         info - 详细信息结构体

    %% ========== 步骤 1: 获取 d1、d2 初始值 ==========
    [~, ~, d1_init, d2_init, ~, ~, info_init] = ik_d1_d2(0, 0, params);
    %用 ~ 忽略不需要的输出变量，只保留 d1_init、d2_init 和 info_init。
    %info_init就是ik_d1_d2函数返回的第7个输出参数info，包含了初始状态计算的成功与否以及相关信息。
    if ~info_init.success
        info.success = false; 
        info.error = '初始状态计算失败';
        T_pos = []; T_rot = []; return;
        %return：结束 ‘当前’ 函数的执行，并返回到调用该函数的位置。
    end
    %%ik_d1_d2中检查可行性时，如果不达标，info结构体的success参数就是false；error参数是‘初始。。。不足’，
    % 然后将变量设为空值并return结束函数。
    % 然后ik_d1_d2中的info结构体在fk_chain_to_T中改名为info_init，fk_chain_to_T中的info是另一个结构体。
    % 此时用~判断info_init.success的布尔值，
    % 如果info_init.success是false（也就是ik_d1_d2中的info.success是false），
    % 则fk_chain_to_T中的info.success = false；info.error = '初始状态计算失败'。

    %% ========== 步骤 2: 求解 q1、q2（已知 d 求 q）==========
    [q1, q2, info_q] = fk_d1d2_to_q1q2(d1_init, delta_d1, d2_init, delta_d2, params);
    if ~info_q.success
        info.success = false; 
        info.error = ['q1/q2 求解失败: ', info_q.error];
        T_pos = []; T_rot = []; return;
    end
    
    %% ========== 步骤 3: 求解 q4 ==========
    q4 = q3_to_q4(q3);
    
    %% ========== 步骤 4: 初始偏置旋转（绕 x 轴 -5.41°）==========
    theta_init = -5.41;  % 初始位置时的 x 轴偏置角
    Rx_init = [1, 0, 0;
               0, cosd(theta_init), -sind(theta_init);
               0, sind(theta_init),  cosd(theta_init)];
    
    %% ========== 步骤 5: 计算 R 矩阵（q1、q2 旋转）==========

    c1 = cosd(q1); s1 = sind(q1);
    c2 = cosd(q2); s2 = sind(q2);
    
    R_q1q2 = [c2, s2*s1, s2*c1;
              0,  c1,   -s1;
             -s2, c2*s1, c2*c1];
    
    % 总旋转：先初始偏置，再 q1/q2 旋转
    R = R_q1q2 * Rx_init;
    
    %% ========== 步骤 6: 计算后续旋转矩阵 ==========
    theta3 = q3 + 5.41;
    theta4 = q4 - 8.15;
    
    Rx3 = [1, 0, 0;
           0, cosd(theta3), -sind(theta3);
           0, sind(theta3),  cosd(theta3)];
    
    Rx4 = [1, 0, 0;
           0, cosd(theta4), -sind(theta4);
           0, sind(theta4),  cosd(theta4)];
    
    %% ========== 步骤 7: 计算 T 点位置 ==========
    P = params.P(:);
    PM = params.L_PM;
    MN = params.L_MN;
    NT = params.L_NT;
    
    v_NT = [0; 0; NT];
    v_MN = [0; 0; MN] + Rx4 * v_NT;
    v_PM = [0; 0; PM] + Rx3 * v_MN;
    
    T_pos = (P + R * v_PM)';
    
    %% ========== 步骤 8: 计算 T 点姿态 ==========
    T_rot = R * Rx3 * Rx4;
    
    %% ========== 信息输出 ==========
    info.success = true;
    info.q1 = q1;
    info.q2 = q2;
    info.q3 = q3;
    info.q4 = q4;
    info.theta_init = theta_init;
    info.theta3 = theta3;
    info.theta4 = theta4;
end
