%% =========================================================================
%   拇指机构逆解求解验证（从正解生成的 T、n_c 还原关节角）
%   不修改任何核心代码，只读调用逆解函数
% =========================================================================
%
%   输入：vis_joint_traj_data.mat
%   输出：vis_ik_solve_data.mat
%
%   依赖（只读调用）：
%     - thumb_ik_q1q2.m（法向量输入版本）
%     - thumb_ik_q3q4.m
%     - params.m
%
%   【关键改进】改进连贯性选择逻辑：
%   1. 对于每个 q1/q2 解，都尝试求解 q3/q4
%   2. 收集所有完整的 q1/q2/q3/q4 解
%   3. 选择与前一帧四关节总偏差最小的解
%
%   【修改】输入从 phi 改为法向量 n_c
% =========================================================================

close all;
clear;
clc;

%% 1. 加载正解轨迹数据
try
    data = load('visualization/vis_joint_traj_data.mat');
    vis_joint_traj_data = data.vis_joint_traj_data;
catch
    error('请先运行 thumb_vis_joint_traj.m 生成关节轨迹数据');
end

T_sequence = vis_joint_traj_data.T_sequence;
phi_sequence = vis_joint_traj_data.phi_sequence;
q_sequence_original = vis_joint_traj_data.q_sequence;
stage_bounds = vis_joint_traj_data.stage_bounds;
stage_names = vis_joint_traj_data.stage_names;
num_frames = vis_joint_traj_data.num_frames;

% 从正解数据中获取法向量序列（如果存在）
if isfield(vis_joint_traj_data, 'n_c_sequence')
    n_c_sequence = vis_joint_traj_data.n_c_sequence;
    use_normal_vector = true;
    fprintf('使用法向量作为输入\n');
else
    % 如果没有法向量序列，从 phi 计算近似法向量
    use_normal_vector = false;
    fprintf('未找到法向量序列，从 phi 计算近似法向量\n');
end

%% 2. 批量逆解求解（改进的连贯性多解选择 + 法向量输入）
q_sequence_ik = zeros(num_frames, 4);
ik_info_seq = struct('q1q2',{}, 'q3q4',{});
success = false(num_frames, 1);

% 初始关节角（用于连续性选择）- 使用第一帧的原始值
q_prev = q_sequence_original(1, :);

fprintf('=== 批量逆解求解 (%d 帧，改进的连贯性多解选择) ===\n', num_frames);

error_q1 = zeros(num_frames, 1);
error_q2 = zeros(num_frames, 1);
error_q3 = zeros(num_frames, 1);
error_q4 = zeros(num_frames, 1);

% 用于统计
multi_solution_count = 0;
continuity_selected_count = 0;
backtrack_count = 0;

for i = 1:num_frames
    T = T_sequence(:, i);
    phi = phi_sequence(i);
    q_orig = q_sequence_original(i, :);
    
    % 获取法向量输入
    if use_normal_vector
        n_c = n_c_sequence(:, i);
    else
        % 从 phi 计算近似法向量（假设 q1≈0, q2≈0 时的法向量）
        % n_c = [-sin(q2); cos(q1)*cos(q2); sin(q1)*cos(q2)]
        % 近似：n_c ≈ [0; cos(phi); sin(phi)] 或 [0; cos(phi); 0]
        phi_rad = phi * pi / 180;
        n_c = [0; cos(phi_rad); 0];  % 简化近似
    end
    
    % 步骤 1: 求解 q1, q2（使用法向量输入）
    [q1_all, q2_all, info_q1q2] = thumb_ik_q1q2(T, n_c);
    
    if isempty(q1_all) || info_q1q2.status ~= 0
        fprintf('帧 %d: q1/q2 求解失败\n', i);
        continue;
    end
    
    % 【关键改进】收集所有完整的 q1/q2/q3/q4 解
    all_full_solutions = [];  % 每行：[q1, q2, q3, q4, continuity_cost]
    info_q3q4_cell = {};  % 使用 cell 数组存储结构体
    
    for k = 1:length(q1_all)
        q1_cand = q1_all(k);
        q2_cand = q2_all(k);
        
        % 尝试求解 q3, q4
        [q3, q4, info_q3q4] = thumb_ik_q3q4(T, q1_cand, q2_cand);
        
        if ~isempty(q3)
            % 计算与前一帧的连贯性成本（四关节总偏差）
            continuity_cost = abs(q1_cand - q_prev(1)) + abs(q2_cand - q_prev(2)) + ...
                              abs(q3 - q_prev(3)) + abs(q4 - q_prev(4));
            
            all_full_solutions = [all_full_solutions; 
                                  q1_cand, q2_cand, q3, q4, continuity_cost];
            info_q3q4_cell{end+1} = info_q3q4;
        end
    end
    
    % 检查是否有完整解
    if isempty(all_full_solutions)
        fprintf('帧 %d: 所有 q1/q2 解对应的 q3/q4 均无解\n', i);
        continue;
    end
    
    % 【关键改进】选择连贯性最好的完整解
    if size(all_full_solutions, 1) > 1
        multi_solution_count = multi_solution_count + 1;
        
        % 找到连贯性成本最小的解
        [~, best_idx] = min(all_full_solutions(:, 5));
        
        % 检查是否需要回溯
        if best_idx ~= 1
            backtrack_count = backtrack_count + 1;
            fprintf('帧 %d: 连贯性最优解不是第一组，回溯选择第 %d 组解\n', i, best_idx);
        end
        
        q1 = all_full_solutions(best_idx, 1);
        q2 = all_full_solutions(best_idx, 2);
        q3 = all_full_solutions(best_idx, 3);
        q4 = all_full_solutions(best_idx, 4);
        continuity_selected_count = continuity_selected_count + 1;
        
        fprintf('帧 %d: 检测到 %d 组完整解，选择连贯性最优解\n', i, size(all_full_solutions, 1));
    else
        % 只有一组完整解
        q1 = all_full_solutions(1, 1);
        q2 = all_full_solutions(1, 2);
        q3 = all_full_solutions(1, 3);
        q4 = all_full_solutions(1, 4);
    end
    
    % 存储结果
    q_sequence_ik(i, :) = [q1, q2, q3, q4];
    ik_info_seq(i).q1q2 = info_q1q2;
    % 从 cell 数组中取出第一个 info_q3q4
    if ~isempty(info_q3q4_cell)
        ik_info_seq(i).q3q4 = info_q3q4_cell{1};
    end
    success(i) = true;
    
    % 更新前一帧关节角（用于下一帧的连贯性选择）
    q_prev = [q1, q2, q3, q4];
    
    % 计算误差
    error_q1(i) = abs(q1 - q_orig(1));
    error_q2(i) = abs(q2 - q_orig(2));
    error_q3(i) = abs(q3 - q_orig(3));
    error_q4(i) = abs(q4 - q_orig(4));
    
    if mod(i, 20) == 0
        fprintf('帧 %d: q_ik=[%.2f,%.2f,%.2f,%.2f] | 误差=[%.4f,%.4f,%.4f,%.4f] ✓\n', ...
            i, q1, q2, q3, q4, error_q1(i), error_q2(i), error_q3(i), error_q4(i));
    end
end

%% 3. 误差分析
fprintf('\n=== 逆解误差统计 ===\n');
valid_frames = sum(success);
fprintf('有效帧数：%d/%d (%.1f%%)\n', valid_frames, num_frames, valid_frames/num_frames*100);
fprintf('q1 误差：均值=%.4f°, 最大值=%.4f°\n', mean(error_q1(success)), max(error_q1(success)));
fprintf('q2 误差：均值=%.4f°, 最大值=%.4f°\n', mean(error_q2(success)), max(error_q2(success)));
fprintf('q3 误差：均值=%.4f°, 最大值=%.4f°\n', mean(error_q3(success)), max(error_q3(success)));
fprintf('q4 误差：均值=%.4f°, 最大值=%.4f°\n', mean(error_q4(success)), max(error_q4(success)));
fprintf('多解情况统计：%d 帧出现多解，其中 %d 次使用连贯性选择\n', ...
    multi_solution_count, continuity_selected_count);
fprintf('回溯统计：%d 次连贯性最优解不是第一组\n', backtrack_count);

%% 4. 保存数据
vis_ik_solve_data.q_sequence_ik = q_sequence_ik;
vis_ik_solve_data.q_sequence_original = q_sequence_original;
vis_ik_solve_data.T_sequence = T_sequence;
vis_ik_solve_data.phi_sequence = phi_sequence;
vis_ik_solve_data.ik_info = ik_info_seq;
vis_ik_solve_data.success = success;
vis_ik_solve_data.num_frames = num_frames;
vis_ik_solve_data.error_q1 = error_q1;
vis_ik_solve_data.error_q2 = error_q2;
vis_ik_solve_data.error_q3 = error_q3;
vis_ik_solve_data.error_q4 = error_q4;
vis_ik_solve_data.stage_bounds = stage_bounds;
vis_ik_solve_data.stage_names = stage_names;
vis_ik_solve_data.multi_solution_count = multi_solution_count;
vis_ik_solve_data.continuity_selected_count = continuity_selected_count;
vis_ik_solve_data.backtrack_count = backtrack_count;
vis_ik_solve_data.use_normal_vector = use_normal_vector;

save('visualization/vis_ik_solve_data.mat', 'vis_ik_solve_data');
fprintf('\n逆解数据已保存：visualization/vis_ik_solve_data.mat\n');

%% 5. 绘制误差分析图
fig = figure('Color','w', 'Position',[100,100,1200,700]);

subplot(2, 2, 1);
plot(error_q1, 'b-', 'LineWidth', 1.5); hold on;
plot(error_q2, 'g-', 'LineWidth', 1.5);
ylabel('误差 (°)'); title('q1, q2 逆解误差'); grid on;
legend('q1 误差', 'q2 误差');

subplot(2, 2, 2);
plot(error_q3, 'm-', 'LineWidth', 1.5); hold on;
plot(error_q4, 'c-', 'LineWidth', 1.5);
ylabel('误差 (°)'); title('q3, q4 逆解误差'); grid on;
legend('q3 误差', 'q4 误差');

subplot(2, 2, 3);
total_error = error_q1 + error_q2 + error_q3 + error_q4;
plot(total_error, 'r-', 'LineWidth', 1.5); hold on;
ylabel('总误差 (°)'); title('四关节总误差'); grid on;

subplot(2, 2, 4);
% 根据 stage_bounds 计算各阶段平均误差
num_stages = size(stage_bounds, 1);
phase_errors = zeros(num_stages, 1);

for p = 1:num_stages
    start_frame = stage_bounds(p, 1);
    end_frame = stage_bounds(p, 2);
    frame_idx = start_frame:end_frame;
    valid_in_phase = success(frame_idx);
    if any(valid_in_phase)
        phase_errors(p) = mean(total_error(frame_idx(valid_in_phase)));
    else
        phase_errors(p) = NaN;
    end
end

bar(phase_errors, 'FaceColor', [0.2 0.6 0.8]);
set(gca, 'XTickLabel', stage_names);
xtickangle(15);
ylabel('平均总误差 (°)'); title('各阶段误差对比'); grid on;

sgtitle('逆解误差分析（正解→逆解闭环验证）');
saveas(fig, 'visualization/vis_ik_error_analysis.png');
fprintf('误差分析图已保存：visualization/vis_ik_error_analysis.png\n');
