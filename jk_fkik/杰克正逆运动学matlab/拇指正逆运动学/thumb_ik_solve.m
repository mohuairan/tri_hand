% ================================================================================
%                    拇指机构逆运动学求解测试脚本（法向量输入版本）
% ================================================================================
% 功能：用于调试和验证逆运动学算法
% 说明：本文件仅包含输入、打印和验证部分，不含运算核心代码
%       核心算法在 thumb_ik_q1q2.m 和 thumb_ik_q3q4.m 中实现
% 
% 修改：输入从 phi 角度改为法向量 n_c
% ================================================================================

clear; clc; close all;

%% 1. 输入目标位置和法向量
% 在此处修改目标位置和法向量进行测试
P_target = [ -48.65; 64.89;1.44];   % 末端目标位置 [x; y; z] (mm)

% 法向量输入（单位向量）
% 示例：从正解得到的法向量
n_c = [0.638; 0.492; -0.592];   % 法向量 [nx; ny; nz]
% 注意：法向量应该是单位向量，如果不是会自动归一化

%% 2. 第一步：求解 q1, q2（基于法向量）
fprintf('==================== 步骤 1: 求解 q1, q2 ====================\n');
fprintf('输入：P_target = [%.4f; %.4f; %.4f] mm\n', P_target(1), P_target(2), P_target(3));
fprintf('输入：n_c = [%.6f; %.6f; %.6f]\n', n_c(1), n_c(2), n_c(3));
fprintf('\n');

[q1, q2, info1] = thumb_ik_q1q2(P_target, n_c);

% 显示求解结果
fprintf('\n【步骤 1 结果】\n');
if ~isempty(info1.all_candidates)
    fprintf('所有候选解:\n');
    for i = 1:size(info1.all_candidates, 1)
        if isnan(info1.all_candidates(i,1))
            fprintf('  候选解 %d: 无效解\n', i);
        else
            if info1.all_candidates(i,3)
                limit_flag = '✓';
            else
                limit_flag = '✗';
            end
            fprintf('  候选解 %d: q1=%7.4f°, q2=%7.4f° [%s]\n', ...
                i, info1.all_candidates(i,1), info1.all_candidates(i,2), limit_flag);
        end
    end
    fprintf('\n');
end

% 检查求解结果
if isempty(q1)
    fprintf('❌ q1/q2 求解失败\n');
    fprintf('错误信息：%s\n', info1.error_msg);
    return;
end

if info1.limit_exceeded
    fprintf('⚠ %s\n', info1.error_msg);
else
    fprintf('✅ q1/q2 求解成功\n');
end
fprintf('选用解：q1 = %.4f°, q2 = %.4f°\n', q1, q2);

% 显示中间输出 phi
fprintf('平面夹角 phi = %.4f°（中间输出）\n', info1.phi);
fprintf('\n');

%% 3. 求解 q3, q4
fprintf('==================== 步骤 2: 求解 q3, q4 ====================\n');
all_ik_solutions = [];  % 存储所有完整逆解
info2 = [];  % 初始化 info2
q3 = []; q4 = [];  % 初始化 q3, q4

% 法向量输入版本：q1/q2 只有唯一解，q3/q4 有两组解（正负肘部）
% 选择与原始姿态最接近的解（肘部向下配置，theta4 >= 0）

q1_cand = info1.q1_all(1);
q2_cand = info1.q2_all(1);

[q3, q4, info_q3q4] = thumb_ik_q3q4(P_target, q1_cand, q2_cand);

if isempty(q3)
    fprintf('❌ q3/q4 求解失败\n');
    fprintf('错误信息：%s\n', info_q3q4.error_msg);
    return;
end

% 检查是否有多个 q3/q4 解
num_q3q4_solutions = length(info_q3q4.q3_all);

if num_q3q4_solutions > 1
    % 多解时选择肘部向下配置（theta4 >= 0，即 q4 更接近 0 的解）
    % 或者选择关节角绝对值和最小的解
    cost = abs(info_q3q4.q3_all) + abs(info_q3q4.q4_all);
    [~, best_idx] = min(cost);
    q3 = info_q3q4.q3_all(best_idx);
    q4 = info_q3q4.q4_all(best_idx);
    
    fprintf('  检测到 %d 组 q3/q4 解，选择最优解：\n', num_q3q4_solutions);
    for k = 1:num_q3q4_solutions
        flag = '';
        if k == best_idx
            flag = ' ← 已选择';
        end
        fprintf('    解%d: q3=%7.4f°, q4=%7.4f°%s\n', ...
            k, info_q3q4.q3_all(k), info_q3q4.q4_all(k), flag);
    end
else
    fprintf('  解：q3=%7.4f°, q4=%7.4f° ✓\n', q3, q4);
end

all_ik_solutions = [q1_cand, q2_cand, q3, q4];
info2 = info_q3q4;

fprintf('\n');

% 检查 info2 是否有效
if isempty(info2)
    fprintf('❌ q3/q4 求解失败：info2 为空\n');
    return;
end

% 检查求解结果
if isempty(q3)
    fprintf('❌ q3/q4 求解失败\n');
    fprintf('错误信息：%s\n', info2.error_msg);
    return;
end

if info2.limit_exceeded
    fprintf('⚠ %s\n', info2.error_msg);
else
    fprintf('✅ q3/q4 求解成功\n');
end
fprintf('q3 = %.4f°, q4 = %.4f°\n', q3, q4);
fprintf('\n');

%% 4. 输出完整结果
fprintf('==================== 逆解完整结果 ====================\n');
fprintf('关节角度：q1=%.2f°, q2=%.2f°, q3=%.2f°, q4=%.2f°\n', q1, q2, q3, q4);
fprintf('M 点位置：[%.4f, %.4f, %.4f] (mm)\n', info2.P_M);
fprintf('N 点位置：[%.4f, %.4f, %.4f] (mm)\n', info2.P_N);

%% 5. 正解验证
fprintf('\n==================== 正解验证 ====================\n');
[T, P_ver, R_ver, fk_info] = thumb_fk_main(q1, q2, q3, q4);
pos_error = norm(P_ver - P_target);
fprintf('目标位置：[%.4f, %.4f, %.4f] (mm)\n', P_target);
fprintf('验证位置：[%.4f, %.4f, %.4f] (mm)\n', P_ver);
fprintf('位置误差：%.6f mm\n', pos_error);

% 获取容差参数
p = params();
if pos_error < p.position_tolerance
    fprintf('✅ 验证通过\n');
else
    fprintf('⚠ 验证警告（误差 >= %.2fmm）\n', p.position_tolerance);
end

fprintf('\n约束平面验证:\n');
fprintf('  输入法向量 n_c = [%.6f, %.6f, %.6f]\n', n_c(1), n_c(2), n_c(3));
fprintf('  正解法向量 n_c = [%.6f, %.6f, %.6f]\n', fk_info.n_c(1), fk_info.n_c(2), fk_info.n_c(3));
fprintf('  输入 phi = %.4f°\n', info1.phi);
fprintf('  正解 phi = %.4f°\n', fk_info.phi);

% 法向量误差
n_error = norm(fk_info.n_c - n_c);
fprintf('\n法向量误差：%.6f\n', n_error);
if n_error < p.tol_normal * 10
    fprintf('✅ 法向量验证通过\n');
else
    fprintf('⚠ 法向量误差较大\n');
end

%% 6. 与测量值对比（可选）
fprintf('\n==================== 与测量值对比 ====================\n');
% 取消注释并填入测量值进行对比
% measure_q = [-16.17, -35.35, -30.6, -43.93];
% if exist('measure_q', 'var') && ~isempty(all_ik_solutions)
%     for i = 1:size(all_ik_solutions, 1)
%         diff = norm(all_ik_solutions(i,:) - measure_q);
%         fprintf('候选解 %d 与测量值偏差：%.4f°\n', i, diff);
%     end
% end

%% 7. 可视化（可选）
% 取消注释以启用 3D 可视化
%{
figure('Name', '拇指机构逆解可视化');
hold on; grid on; axis equal;

O = [0; 0; 0];
M = info2.P_M;
N = info2.P_N;
T_point = P_target;

% 绘制各点
plot3(0, 0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot3(M(1), M(2), M(3), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot3(N(1), N(2), N(3), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot3(T_point(1), T_point(2), T_point(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% 绘制连杆
plot3([0, M(1)], [0, M(2)], [0, M(3)], 'b-', 'LineWidth', 2);
plot3([M(1), N(1)], [M(2), N(2)], [M(3), N(3)], 'g-', 'LineWidth', 2);
plot3([N(1), T_point(1)], [N(2), T_point(2)], [N(3), T_point(3)], 'r-', 'LineWidth', 2);

% 标注
text(0, 0, 0, ' O', 'VerticalAlignment', 'bottom');
text(M(1), M(2), M(3), ' M', 'VerticalAlignment', 'bottom');
text(N(1), N(2), N(3), ' N', 'VerticalAlignment', 'bottom');
text(T_point(1), T_point(2), T_point(3), ' T', 'VerticalAlignment', 'bottom');

xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('拇指机构逆解可视化 (O-M-N-T)');
view(3);
%}
