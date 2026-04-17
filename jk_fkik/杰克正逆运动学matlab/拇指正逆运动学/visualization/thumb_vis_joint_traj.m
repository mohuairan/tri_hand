%% =========================================================================
%   拇指机构关节轨迹生成（正解生成 T、phi 序列）
%   仿照 ILDA-MCP 五阶段运动规划
% =========================================================================
%
%   输入：无（内置轨迹规划）
%   输出：vis_joint_traj_data.mat
%
%   运动阶段：
%     阶段 1: 仅 q2 变化 (0→-20→0)，q1,q3,q4 不变
%     阶段 2: 仅 q1 变化 (0→-15→0)，q2,q3,q4 不变
%     阶段 3: 仅 q3,q4 变化 (0→-60→0, 0→-40→0)，q1,q2 不变
%     阶段 4: q1,q2 同步变化 (0→-15→0, 0→-20→0)，q3,q4 不变
%     阶段 5: q1,q2,q3,q4 四关节协同变化
%
%   依赖（只读调用）：
%     - thumb_fk_main.m
%     - params.m
%     - vis_params.m (可视化参数)
% =========================================================================

close all;
clear;

%% 1. 加载参数
% 机械参数（用于获取关节限位）
p = params();
q_limits = [p.q1_limit_min, p.q1_limit_max;
            p.q2_limit_min, p.q2_limit_max;
            p.q3_limit_min, p.q3_limit_max;
            p.q4_limit_min, p.q4_limit_max];

% 可视化参数（用于轨迹规划）
vp = vis_params();

%% 2. 定义输入序列（动画路径）- 五阶段运动规划
n_half = vp.n_half;  % 每个半阶段帧数

% ---- 阶段 1：仅 q2 变化 (0→-20→0)，q1,q3,q4 不变 ----
stage1_q1 = zeros(1, 2*n_half);
stage1_q2 = [linspace(vp.stage1_q2_min, vp.stage1_q2_max, n_half), ...
             linspace(vp.stage1_q2_max, vp.stage1_q2_min, n_half)];
stage1_q3 = zeros(1, 2*n_half);
stage1_q4 = zeros(1, 2*n_half);

% ---- 阶段 2：仅 q1 变化 (0→-15→0)，q2,q3,q4 不变 ----
stage2_q1 = [linspace(vp.stage2_q1_min, vp.stage2_q1_max, n_half), ...
             linspace(vp.stage2_q1_max, vp.stage2_q1_min, n_half)];
stage2_q2 = zeros(1, 2*n_half);
stage2_q3 = zeros(1, 2*n_half);
stage2_q4 = zeros(1, 2*n_half);

% ---- 阶段 3：仅 q3,q4 变化 (0→-60→0, 0→-40→0)，q1,q2 不变 ----
stage3_q1 = zeros(1, 2*n_half);
stage3_q2 = zeros(1, 2*n_half);
stage3_q3 = [linspace(vp.stage3_q3_min, vp.stage3_q3_max, n_half), ...
             linspace(vp.stage3_q3_max, vp.stage3_q3_min, n_half)];
stage3_q4 = [linspace(vp.stage3_q4_min, vp.stage3_q4_max, n_half), ...
             linspace(vp.stage3_q4_max, vp.stage3_q4_min, n_half)];

% ---- 阶段 4：q1,q2 同步变化 (0→-15→0, 0→-20→0)，q3,q4 不变 ----
stage4_q1 = [linspace(vp.stage4_q1_min, vp.stage4_q1_max, n_half), ...
             linspace(vp.stage4_q1_max, vp.stage4_q1_min, n_half)];
stage4_q2 = [linspace(vp.stage4_q2_min, vp.stage4_q2_max, n_half), ...
             linspace(vp.stage4_q2_max, vp.stage4_q2_min, n_half)];
stage4_q3 = zeros(1, 2*n_half);
stage4_q4 = zeros(1, 2*n_half);

% ---- 阶段 5：q1,q2,q3,q4 四关节协同变化 (120 帧) ----
n5 = vp.n5;  % 120 帧
t5 = linspace(0, 2*pi, n5);

% q1: 较慢变化，从 0 开始
stage5_q1 = vp.stage5_q1_amp * (1 - cos(t5));
% q2: 较快变化（频率 1.2 倍），从 0 开始
stage5_q2 = vp.stage5_q2_amp * (1 - cos(vp.stage5_q2_freq * t5));
% q3: 从 0 弯到 -50 再回来，从 0 开始
stage5_q3 = vp.stage5_q3_amp * (1 - cos(t5));
% q4: 从 0 弯到 -35 再回来，从 0 开始
stage5_q4 = vp.stage5_q4_amp * (1 - cos(vp.stage5_q4_freq * t5));

% 确保不超出限位
for i = 1:n5
    stage5_q1(i) = max(q_limits(1,1), min(q_limits(1,2), stage5_q1(i)));
    stage5_q2(i) = max(q_limits(2,1), min(q_limits(2,2), stage5_q2(i)));
    stage5_q3(i) = max(q_limits(3,1), min(q_limits(3,2), stage5_q3(i)));
    stage5_q4(i) = max(q_limits(4,1), min(q_limits(4,2), stage5_q4(i)));
end

% ---- 拼接所有阶段 ----
q1_seq = [stage1_q1, stage2_q1, stage3_q1, stage4_q1, stage5_q1];
q2_seq = [stage1_q2, stage2_q2, stage3_q2, stage4_q2, stage5_q2];
q3_seq = [stage1_q3, stage2_q3, stage3_q3, stage4_q3, stage5_q3];
q4_seq = [stage1_q4, stage2_q4, stage3_q4, stage4_q4, stage5_q4];

num_frames = length(q1_seq);

% 阶段边界（用于标注）
stage_bounds = [1, 2*n_half;                                    % 阶段 1
                2*n_half+1, 4*n_half;                           % 阶段 2
                4*n_half+1, 6*n_half;                           % 阶段 3
                6*n_half+1, 8*n_half;                           % 阶段 4
                8*n_half+1, 8*n_half+n5];                       % 阶段 5

stage_names = {'阶段 1: 仅 q2 变化', ...
               '阶段 2: 仅 q1 变化', ...
               '阶段 3: 仅 q3+q4 变化', ...
               '阶段 4: q1+q2 同步', ...
               '阶段 5: 四关节协同'};

fprintf('=== 运动规划 ===\n');
fprintf('总帧数：%d\n', num_frames);
for s = 1:5
    fprintf('  %s: 帧 %d-%d (%d 帧)\n', stage_names{s}, ...
            stage_bounds(s,1), stage_bounds(s,2), ...
            stage_bounds(s,2)-stage_bounds(s,1)+1);
end

%% 3. 调用正解生成 T、phi、n_c 序列
T_sequence = zeros(3, num_frames);
phi_sequence = zeros(num_frames, 1);
n_c_sequence = zeros(3, num_frames);  % 法向量序列
fk_info_seq = struct('P_M',{}, 'P_N',{}, 'n_c',{}, 'phi',{});

fprintf('\n=== 正解计算 T、phi、n_c 序列 (%d 帧) ===\n', num_frames);
for i = 1:num_frames
    q = [q1_seq(i), q2_seq(i), q3_seq(i), q4_seq(i)];
    
    % 调用核心正解函数（只读）
    [T_end, P_end, R_end, fk_info] = thumb_fk_main(q(1), q(2), q(3), q(4));
    
    T_sequence(:, i) = P_end;
    phi_sequence(i) = fk_info.phi;
    n_c_sequence(:, i) = fk_info.n_c;  % 保存法向量
    fk_info_seq(i) = fk_info;
    
    if mod(i, 20) == 0
        fprintf('帧 %d: T=[%.2f,%.2f,%.2f], phi=%.2f°, n_c=[%.4f,%.4f,%.4f] ✓\n', ...
            i, P_end(1), P_end(2), P_end(3), fk_info.phi, fk_info.n_c(1), fk_info.n_c(2), fk_info.n_c(3));
    end
end

%% 4. 保存数据
vis_joint_traj_data.q1_seq = q1_seq;
vis_joint_traj_data.q2_seq = q2_seq;
vis_joint_traj_data.q3_seq = q3_seq;
vis_joint_traj_data.q4_seq = q4_seq;
vis_joint_traj_data.q_sequence = [q1_seq', q2_seq', q3_seq', q4_seq'];
vis_joint_traj_data.T_sequence = T_sequence;
vis_joint_traj_data.phi_sequence = phi_sequence;
vis_joint_traj_data.n_c_sequence = n_c_sequence;  % 法向量序列
vis_joint_traj_data.fk_info = fk_info_seq;
vis_joint_traj_data.stage_bounds = stage_bounds;
vis_joint_traj_data.stage_names = stage_names;
vis_joint_traj_data.num_frames = num_frames;
vis_joint_traj_data.q_limits = q_limits;

save('visualization/vis_joint_traj_data.mat', 'vis_joint_traj_data');
fprintf('\n关节轨迹数据已保存：visualization/vis_joint_traj_data.mat\n');

%% 5. 绘制关节轨迹预览图
fig = figure('Color','w', 'Position', vp.fig_position);

subplot(2, 2, 1);
plot(q1_seq, 'b-', 'LineWidth', 1.5); hold on;
yline([q_limits(1,1), q_limits(1,2)], 'r--');
ylabel('q1 (°)'); title('关节 1 轨迹'); grid on;

subplot(2, 2, 2);
plot(q2_seq, 'g-', 'LineWidth', 1.5); hold on;
yline([q_limits(2,1), q_limits(2,2)], 'r--');
ylabel('q2 (°)'); title('关节 2 轨迹'); grid on;

subplot(2, 2, 3);
plot(q3_seq, 'm-', 'LineWidth', 1.5); hold on;
yline([q_limits(3,1), q_limits(3,2)], 'r--');
ylabel('q3 (°)'); title('关节 3 轨迹'); grid on;

subplot(2, 2, 4);
plot(q4_seq, 'c-', 'LineWidth', 1.5); hold on;
yline([q_limits(4,1), q_limits(4,2)], 'r--');
ylabel('q4 (°)'); title('关节 4 轨迹'); grid on;

sgtitle('关节轨迹预览（正解生成 T、phi）');

% 修复 saveas 兼容性问题
try
    saveas(fig, 'visualization/vis_joint_traj_preview.png');
    fprintf('轨迹预览图已保存：visualization/vis_joint_traj_preview.png\n');
catch ME
    % 如果 saveas 失败，尝试使用 print
    fprintf('saveas 失败，尝试使用 print 保存...\n');
    print(fig, '-dpng', 'visualization/vis_joint_traj_preview.png');
    fprintf('轨迹预览图已保存：visualization/vis_joint_traj_preview.png (使用 print)\n');
end

%% 6. 显示末端位姿摘要
fprintf('\n=== 末端 T 点位姿摘要 ===\n');
fprintf('帧数：%d 帧\n', num_frames);
fprintf('T 点范围:\n');
fprintf('  X: [%.3f, %.3f] mm\n', min(T_sequence(1,:)), max(T_sequence(1,:)));
fprintf('  Y: [%.3f, %.3f] mm\n', min(T_sequence(2,:)), max(T_sequence(2,:)));
fprintf('  Z: [%.3f, %.3f] mm\n', min(T_sequence(3,:)), max(T_sequence(3,:)));
fprintf('phi 范围：[%.3f, %.3f]°\n', min(phi_sequence), max(phi_sequence));
fprintf('法向量范围:\n');
fprintf('  n_x: [%.4f, %.4f]\n', min(n_c_sequence(1,:)), max(n_c_sequence(1,:)));
fprintf('  n_y: [%.4f, %.4f]\n', min(n_c_sequence(2,:)), max(n_c_sequence(2,:)));
fprintf('  n_z: [%.4f, %.4f]\n', min(n_c_sequence(3,:)), max(n_c_sequence(3,:)));
fprintf('关节角范围:\n');
fprintf('  q1: [%.3f, %.3f]°\n', min(q1_seq), max(q1_seq));
fprintf('  q2: [%.3f, %.3f]°\n', min(q2_seq), max(q2_seq));
fprintf('  q3: [%.3f, %.3f]°\n', min(q3_seq), max(q3_seq));
fprintf('  q4: [%.3f, %.3f]°\n', min(q4_seq), max(q4_seq));

fprintf('\n=== 关节轨迹生成完成 ===\n');
