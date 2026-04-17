%% =========================================================================
%   诊断帧 432 和 433 的跳变问题
%   分析 T/phi 数值精度对逆解的影响
% =========================================================================

close all;
clear;
clc;

fprintf('\n============================================================\n');
fprintf('        帧 432-433 跳变问题诊断\n');
fprintf('============================================================\n\n');

%% 1. 加载数据
try
    data = load('visualization/vis_ik_solve_data.mat');
    vis_ik_solve_data = data.vis_ik_solve_data;
    
    data_traj = load('visualization/vis_joint_traj_data.mat');
    vis_joint_traj_data = data_traj.vis_joint_traj_data;
catch
    error('请先运行可视化流程生成数据');
end

%% 2. 提取帧 432 和 433 的数据
frame_432 = 432;
frame_433 = 433;

T_432 = vis_ik_solve_data.T_sequence(:, frame_432);
T_433 = vis_ik_solve_data.T_sequence(:, frame_433);
phi_432 = vis_ik_solve_data.phi_sequence(frame_432);
phi_433 = vis_ik_solve_data.phi_sequence(frame_433);

q_ik_432 = vis_ik_solve_data.q_sequence_ik(frame_432, :);
q_ik_433 = vis_ik_solve_data.q_sequence_ik(frame_433, :);
q_orig_432 = vis_ik_solve_data.q_sequence_original(frame_432, :);
q_orig_433 = vis_ik_solve_data.q_sequence_original(frame_433, :);

fprintf('【帧 432 数据】\n');
fprintf('  T = [%.10f, %.10f, %.10f] mm\n', T_432(1), T_432(2), T_432(3));
fprintf('  phi = %.10f°\n', phi_432);
fprintf('  q_ik = [%.6f, %.6f, %.6f, %.6f]°\n', q_ik_432(1), q_ik_432(2), q_ik_432(3), q_ik_432(4));
fprintf('  q_orig = [%.6f, %.6f, %.6f, %.6f]°\n', q_orig_432(1), q_orig_432(2), q_orig_432(3), q_orig_432(4));
fprintf('\n');

fprintf('【帧 433 数据】\n');
fprintf('  T = [%.10f, %.10f, %.10f] mm\n', T_433(1), T_433(2), T_433(3));
fprintf('  phi = %.10f°\n', phi_433);
fprintf('  q_ik = [%.6f, %.6f, %.6f, %.6f]°\n', q_ik_433(1), q_ik_433(2), q_ik_433(3), q_ik_433(4));
fprintf('  q_orig = [%.6f, %.6f, %.6f, %.6f]°\n', q_orig_433(1), q_orig_433(2), q_orig_433(3), q_orig_433(4));
fprintf('\n');

%% 3. 计算相邻帧的变化量
fprintf('【相邻帧变化量】\n');
fprintf('  ΔT = [%.6f, %.6f, %.6f] mm\n', T_433(1)-T_432(1), T_433(2)-T_432(2), T_433(3)-T_432(3));
fprintf('  Δphi = %.6f°\n', phi_433 - phi_432);
fprintf('  Δq_ik = [%.6f, %.6f, %.6f, %.6f]°\n', ...
    q_ik_433(1)-q_ik_432(1), q_ik_433(2)-q_ik_432(2), ...
    q_ik_433(3)-q_ik_432(3), q_ik_433(4)-q_ik_432(4));
fprintf('  Δq_orig = [%.6f, %.6f, %.6f, %.6f]°\n', ...
    q_orig_433(1)-q_orig_432(1), q_orig_433(2)-q_orig_432(2), ...
    q_orig_433(3)-q_orig_432(3), q_orig_433(4)-q_orig_432(4));
fprintf('\n');

%% 4. 测试不同精度的 T/phi 对逆解的影响
fprintf('【精度测试】\n');

% 测试 1: 使用完整精度的 T 和 phi
fprintf('\n测试 1: 使用完整精度的 T 和 phi\n');
fprintf('  T = [%.10f, %.10f, %.10f], phi = %.10f°\n', T_433(1), T_433(2), T_433(3), phi_433);
[q1_all, q2_all, info_q1q2] = thumb_ik_q1q2(T_433, phi_433);
if ~isempty(q1_all)
    fprintf('  q1/q2 解：\n');
    for k = 1:length(q1_all)
        fprintf('    解%d: q1=%.6f°, q2=%.6f°\n', k, q1_all(k), q2_all(k));
        [q3, q4, ~] = thumb_ik_q3q4(T_433, q1_all(k), q2_all(k));
        if ~isempty(q3)
            fprintf('         → q3=%.6f°, q4=%.6f° ✓\n', q3, q4);
        else
            fprintf('         → q3/q4 无解 ✗\n');
        end
    end
else
    fprintf('  q1/q2 无解\n');
end

% 测试 2: 使用 2 位小数的 T 和 phi（模拟动画显示值）
T_433_rounded = round(T_433 * 100) / 100;
phi_433_rounded = round(phi_433 * 100) / 100;
fprintf('\n测试 2: 使用 2 位小数的 T 和 phi（模拟动画显示）\n');
fprintf('  T = [%.2f, %.2f, %.2f], phi = %.2f°\n', T_433_rounded(1), T_433_rounded(2), T_433_rounded(3), phi_433_rounded);
[q1_all_r, q2_all_r, info_q1q2_r] = thumb_ik_q1q2(T_433_rounded, phi_433_rounded);
if ~isempty(q1_all_r)
    fprintf('  q1/q2 解：\n');
    for k = 1:length(q1_all_r)
        fprintf('    解%d: q1=%.6f°, q2=%.6f°\n', k, q1_all_r(k), q2_all_r(k));
        [q3, q4, ~] = thumb_ik_q3q4(T_433_rounded, q1_all_r(k), q2_all_r(k));
        if ~isempty(q3)
            fprintf('         → q3=%.6f°, q4=%.6f° ✓\n', q3, q4);
        else
            fprintf('         → q3/q4 无解 ✗\n');
        end
    end
else
    fprintf('  q1/q2 无解\n');
end

% 测试 3: 使用原始关节角通过正解计算的 T 和 phi
fprintf('\n测试 3: 使用正解计算的 T 和 phi（最高精度）\n');
[T_fk, P_fk, R_fk, fk_info] = thumb_fk_main(q_orig_433(1), q_orig_433(2), q_orig_433(3), q_orig_433(4));
fprintf('  T_fk = [%.10f, %.10f, %.10f], phi_fk = %.10f°\n', T_fk(1), T_fk(2), T_fk(3), fk_info.phi);
[q1_all_fk, q2_all_fk, info_q1q2_fk] = thumb_ik_q1q2(T_fk, fk_info.phi);
if ~isempty(q1_all_fk)
    fprintf('  q1/q2 解：\n');
    for k = 1:length(q1_all_fk)
        fprintf('    解%d: q1=%.6f°, q2=%.6f°\n', k, q1_all_fk(k), q2_all_fk(k));
        [q3, q4, ~] = thumb_ik_q3q4(T_fk, q1_all_fk(k), q2_all_fk(k));
        if ~isempty(q3)
            fprintf('         → q3=%.6f°, q4=%.6f° ✓\n', q3, q4);
            % 计算与原始值的偏差
            dev = abs(q1_all_fk(k) - q_orig_433(1)) + abs(q2_all_fk(k) - q_orig_433(2)) + ...
                  abs(q3 - q_orig_433(3)) + abs(q4 - q_orig_433(4));
            fprintf('         → 与原始值偏差：%.6f°\n', dev);
        else
            fprintf('         → q3/q4 无解 ✗\n');
        end
    end
else
    fprintf('  q1/q2 无解\n');
end

%% 5. 分析结论
fprintf('\n============================================================\n');
fprintf('                        分析结论\n');
fprintf('============================================================\n');
fprintf('1. 数值精度影响：T/phi 的小数位数对逆解结果有显著影响\n');
fprintf('2. 连贯性跳变：当多组解存在时，可能选择了不连续的解\n');
fprintf('3. phi 容差：当前 tol_phi=%.4f° 可能过小\n', params().tol_phi);
fprintf('\n建议修复步骤：\n');
fprintf('  1. 增加 phi 容差至 0.5° 或更大\n');
fprintf('  2. 在可视化流程中使用原始关节角而非逆解结果\n');
fprintf('  3. 改进连贯性选择逻辑，考虑所有关节的总偏差\n');
