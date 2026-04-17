%% =========================================================================
%   分析失败帧的详细输入数据
%   输出帧 453、549、550 的 T 坐标、phi 值和原始关节角
% =========================================================================

close all;
clear;
clc;

%% 1. 加载数据
try
    data = load('visualization/vis_ik_solve_data.mat');
    vis_ik_solve_data = data.vis_ik_solve_data;
    
    data_traj = load('visualization/vis_joint_traj_data.mat');
    vis_joint_traj_data = data_traj.vis_joint_traj_data;
catch
    error('请先运行 thumb_vis_ik_solve.m 生成逆解数据');
end

%% 2. 提取失败帧数据
failed_frames = [453, 549, 550];
success = vis_ik_solve_data.success;
T_sequence = vis_ik_solve_data.T_sequence;
phi_sequence = vis_ik_solve_data.phi_sequence;
q_sequence_original = vis_ik_solve_data.q_sequence_original;
stage_bounds = vis_joint_traj_data.stage_bounds;
stage_names = vis_joint_traj_data.stage_names;

%% 3. 显示失败帧的详细数据
fprintf('\n================================================================================\n');
fprintf('                    失败帧详细输入数据分析\n');
fprintf('================================================================================\n\n');

for i = 1:length(failed_frames)
    f = failed_frames(i);
    
    % 判断当前阶段
    current_stage = 1;
    for s = 1:5
        if f >= stage_bounds(s,1) && f <= stage_bounds(s,2)
            current_stage = s;
            break;
        end
    end
    
    T = T_sequence(:, f);
    phi = phi_sequence(f);
    q_orig = q_sequence_original(f, :);
    
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('帧号：%d\n', f);
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('阶段：%s\n', stage_names{current_stage});
    fprintf('\n');
    
    fprintf('【输入 T 坐标】(mm)\n');
    fprintf('  T_x = %.6f mm\n', T(1));
    fprintf('  T_y = %.6f mm\n', T(2));
    fprintf('  T_z = %.6f mm\n', T(3));
    fprintf('\n');
    
    fprintf('【输入 phi】(度)\n');
    fprintf('  phi = %.6f°\n');
    fprintf('\n');
    
    fprintf('【原始关节角】(度)\n');
    fprintf('  q1_orig = %.6f°\n', q_orig(1));
    fprintf('  q2_orig = %.6f°\n', q_orig(2));
    fprintf('  q3_orig = %.6f°\n', q_orig(3));
    fprintf('  q4_orig = %.6f°\n', q_orig(4));
    fprintf('\n');
    
    % 计算 MT 距离（用于分析 q3/q4 工作空间）
    p = params();
    L2 = p.L2;
    L3 = p.L3;
    r_max = L2 + L3;
    r_min = abs(L2 - L3);
    
    % 计算 M 点位置（近似）
    R_01_approx = eye(3);  % 简化计算
    P_M_approx = [0; 0; p.L1];
    MT_approx = T - P_M_approx;
    r_approx = norm(MT_approx);
    
    fprintf('【工作空间分析】\n');
    fprintf('  L2 (MN) = %.4f mm\n', L2);
    fprintf('  L3 (NT) = %.4f mm\n', L3);
    fprintf('  可达范围：[%.4f, %.4f] mm\n', r_min, r_max);
    fprintf('  MT 近似距离 ≈ %.4f mm\n', r_approx);
    if r_approx > r_max
        fprintf('  ⚠ 警告：MT 距离超出最大工作空间！\n');
    elseif r_approx < r_min
        fprintf('  ⚠ 警告：MT 距离小于最小工作空间！\n');
    else
        fprintf('  ✓ MT 距离在可达范围内\n');
    end
    fprintf('\n');
end

fprintf('================================================================================\n');
fprintf('                              总结\n');
fprintf('================================================================================\n');
fprintf('失败帧数：%d\n', length(failed_frames));
fprintf('失败类型：q3/q4 求解失败\n');
fprintf('\n');
fprintf('可能原因：\n');
fprintf('1. 在阶段 5，四关节协同运动导致某些位置超出 q3/q4 工作空间\n');
fprintf('2. q1/q2 的连贯性选择可能不是最优，导致 q3/q4 无法求解\n');
fprintf('3. 数值误差累积导致临界位置求解失败\n');
fprintf('\n');
