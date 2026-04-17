function [T_pos, T_rot, info] = fk_T0(q3, params)
%FK_T0 串联机构正运动学：已知q3，计算在q1=0，q2=0时，末端 T 的位姿
%   输入：
%         q3 - 关节角 q3 (度)
%         params - 参数结构体 (来自 ik_params())
%   输出：T_pos - T 点位置 [Tx, Ty, Tz] (mm)
%         T_rot - T 点姿态矩阵 (3x3)
%         info - 详细信息结构体

    
    %% ========== 步骤 1: 求解 q4 ==========
    q4 = q3_to_q4(q3);
    
    %% ========== 步骤 2: 初始偏置旋转（绕 x 轴 -5.41°）==========
    theta_init = -5.41;  % 初始位置时的 x 轴偏置角
    Rx_init = [1, 0, 0;
               0, cosd(theta_init), -sind(theta_init);
               0, sind(theta_init),  cosd(theta_init)];
    
    %% ========== 步骤 3: 计算 R 矩阵（q1、q2 旋转）==========
    q1 = 0.0; q2 = 0.0;

    c1 = cosd(q1); s1 = sind(q1);
    c2 = cosd(q2); s2 = sind(q2);
    
    R_q1q2 = [c2, s2*s1, s2*c1;
              0,  c1,   -s1;
             -s2, c2*s1, c2*c1];
    
    % 总旋转：先初始偏置，再 q1/q2 旋转
    R = R_q1q2 * Rx_init;
    
    %% ========== 步骤 4: 计算后续旋转矩阵 ==========
    theta3 = q3 + 5.41;
    theta4 = q4 - 8.15;
    
    Rx3 = [1, 0, 0;
           0, cosd(theta3), -sind(theta3);
           0, sind(theta3),  cosd(theta3)];
    
    Rx4 = [1, 0, 0;
           0, cosd(theta4), -sind(theta4);
           0, sind(theta4),  cosd(theta4)];
    
    %% ========== 步骤 5: 计算 T 点位置 ==========
    P = params.P(:);
    PM = params.L_PM;
    MN = params.L_MN;
    NT = params.L_NT;
    
    v_NT = [0; 0; NT];
    v_MN = [0; 0; MN] + Rx4 * v_NT;
    v_PM = [0; 0; PM] + Rx3 * v_MN;
    
    T_pos = (P + R * v_PM)';
    
    %% ========== 步骤 6: 计算 T 点姿态 ==========
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
