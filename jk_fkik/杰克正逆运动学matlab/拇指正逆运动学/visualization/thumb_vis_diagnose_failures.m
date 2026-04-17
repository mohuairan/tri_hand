%% =========================================================================
%   拇指机构逆解失败帧诊断
%   提取所有失败帧的 T 坐标和 phi 值
% =========================================================================
%
%   输入：vis_ik_solve_data.mat, vis_joint_traj_data.mat
%   输出：vis_failure_data.mat, vis_failure_diagnosis.png
%
%   使用时机：逆解失败率 > 5% 时使用
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

success = vis_ik_solve_data.success;
T_sequence = vis_ik_solve_data.T_sequence;
phi_sequence = vis_ik_solve_data.phi_sequence;
q_sequence_original = vis_ik_solve_data.q_sequence_original;
stage_bounds = vis_joint_traj_data.stage_bounds;
stage_names = vis_joint_traj_data.stage_names;
num_frames = vis_ik_solve_data.num_frames;

%% 2. 找出失败帧
failed_frames = find(~success);
num_failed = length(failed_frames);
num_total = num_frames;

fprintf('=== 逆解失败帧诊断 ===\n');
fprintf('总帧数：%d\n', num_total);
fprintf('失败帧数：%d (%.1f%%)\n', num_failed, num_failed/num_total*100);
fprintf('成功帧数：%d (%.1f%%)\n', num_total-num_failed, (num_total-num_failed)/num_total*100);

%% 3. 输出失败帧详情
if num_failed > 0
    fprintf('\n=== 失败帧详细信息 ===\n');
    fprintf('%-6s | %-12s | %-12s | %-12s | %-8s | %-6s | %-10s\n', ...
        '帧号', 'T_X (mm)', 'T_Y (mm)', 'T_Z (mm)', 'phi (°)', '阶段', '原始 q1q2');
    fprintf('%s\n', repmat('-', 1, 80));
    
    for i = 1:min(num_failed, 50)  % 最多显示 50 个
        f = failed_frames(i);
        T = T_sequence(:, f);
        phi = phi_sequence(f);
        q_orig = q_sequence_original(f, :);
        
        % 判断阶段
        current_stage = 1;
        for s = 1:5
            if f >= stage_bounds(s,1) && f <= stage_bounds(s,2)
                current_stage = s;
                break;
            end
        end
        
        fprintf('%-6d | %-12.3f | %-12.3f | %-12.3f | %-8.3f | %-6d | [%-5.1f, %-5.1f]\n', ...
            f, T(1), T(2), T(3), phi, current_stage, q_orig(1), q_orig(2));
    end
    
    if num_failed > 50
        fprintf('... 还有 %d 个失败帧\n', num_failed - 50);
    end
end

%% 4. 按阶段统计失败率
fprintf('\n=== 各阶段失败统计 ===\n');
for s = 1:5
    start_f = stage_bounds(s, 1);
    end_f = stage_bounds(s, 2);
    stage_frames = start_f:end_f;
    
    stage_success = success(stage_frames);
    stage_failed = sum(~stage_success);
    stage_total = length(stage_frames);
    
    fprintf('阶段 %d (%s): 失败 %d/%d (%.1f%%)\n', ...
        s, stage_names{s}, stage_failed, stage_total, stage_failed/stage_total*100);
end

%% 5. 分析失败原因
fprintf('\n=== 失败原因分析 ===\n');

% 提取失败帧的 phi 值
phi_failed = phi_sequence(failed_frames);
phi_success = phi_sequence(success);

fprintf('失败帧 phi 统计:\n');
fprintf('  均值：%.3f°, 最小值：%.3f°, 最大值：%.3f°\n', ...
    mean(phi_failed), min(phi_failed), max(phi_failed));

fprintf('成功帧 phi 统计:\n');
fprintf('  均值：%.3f°, 最小值：%.3f°, 最大值：%.3f°\n', ...
    mean(phi_success), min(phi_success), max(phi_success));

% 提取失败帧的 T 坐标
T_failed = T_sequence(:, failed_frames);
T_success = T_sequence(:, success);

fprintf('\n失败帧 T 坐标范围:\n');
fprintf('  X: [%.3f, %.3f] mm\n', min(T_failed(1,:)), max(T_failed(1,:)));
fprintf('  Y: [%.3f, %.3f] mm\n', min(T_failed(2,:)), max(T_failed(2,:)));
fprintf('  Z: [%.3f, %.3f] mm\n', min(T_failed(3,:)), max(T_failed(3,:)));

fprintf('\n成功帧 T 坐标范围:\n');
fprintf('  X: [%.3f, %.3f] mm\n', min(T_success(1,:)), max(T_success(1,:)));
fprintf('  Y: [%.3f, %.3f] mm\n', min(T_success(2,:)), max(T_success(2,:)));
fprintf('  Z: [%.3f, %.3f] mm\n', min(T_success(3,:)), max(T_success(3,:)));

%% 6. 绘制 phi 分布对比图
fig = figure('Color','w', 'Position',[100,100,1000,600]);

subplot(2, 2, 1);
histogram(phi_failed, 'FaceColor', [1 0.4 0.4], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
hold on;
histogram(phi_success, 'FaceColor', [0.2 0.6 0.8], 'EdgeColor', 'none', 'FaceAlpha', 0.7);
xlabel('phi (°)'); ylabel('帧数');
title('phi 值分布对比');
legend('失败帧', '成功帧');
grid on;

subplot(2, 2, 2);
plot(1:num_frames, phi_sequence, 'b-', 'LineWidth', 1);
hold on;
plot(failed_frames, phi_failed, 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('帧号'); ylabel('phi (°)');
title('phi 值随帧变化（×标记为失败帧）');
grid on;

subplot(2, 2, 3);
plot3(T_success(1,:), T_success(2,:), T_success(3,:), 'b.', 'MarkerSize', 8);
hold on;
plot3(T_failed(1,:), T_failed(2,:), T_failed(3,:), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('T 点空间分布（×标记为失败帧）');
grid on; axis equal; view(3);

subplot(2, 2, 4);
failure_rate = zeros(5, 1);
for s = 1:5
    start_f = stage_bounds(s, 1);
    end_f = stage_bounds(s, 2);
    stage_frames = start_f:end_f;
    failure_rate(s) = sum(~success(stage_frames)) / length(stage_frames) * 100;
end
bar(failure_rate, 'FaceColor', [1 0.4 0.4]);
set(gca, 'XTickLabel', stage_names);
xtickangle(15);
ylabel('失败率 (%)'); title('各阶段失败率');
grid on;

sgtitle('逆解失败帧诊断分析');
saveas(fig, 'visualization/vis_failure_diagnosis.png');
fprintf('\n诊断图已保存：visualization/vis_failure_diagnosis.png\n');

%% 7. 保存失败帧列表
failure_data.failed_frames = failed_frames;
failure_data.T_failed = T_failed;
failure_data.phi_failed = phi_failed;
failure_data.num_failed = num_failed;
failure_data.failure_rate = num_failed/num_total*100;

save('visualization/vis_failure_data.mat', 'failure_data');
fprintf('失败帧数据已保存：visualization/vis_failure_data.mat\n');
