%% =========================================================================
%   ILDA 手 MCP 关节正运动学 - 三维可视化与 MP4 视频生成
%   在三维视角下可视化前向运动学，并将参数保存到 MAT 文件
%   依赖：params.m, fk_generate_points.m
% =========================================================================
%
%   输入：d1 变化量、d2 变化量、q3 角度序列
%   输出：三维 MP4 视频文件 + 参数 MAT 文件
%
%   运动阶段：
%     阶段1: 仅 d2 变化 (0→-10→0)，d1=0, q3=0
%     阶段2: 仅 d1 变化 (0→-10→0)，d2=0, q3=0
%     阶段3: 仅 q3 变化 (0→-60→0)，d1=0, d2=0
%     阶段4: d1,d2 同步变化 (0→-10→0)，q3=0
%     阶段5: d1,d2,q3 三者协同变化
%
% =========================================================================

close all;
clear;

%% 1. 定义输入序列（动画路径）- 全面运动规划
% 每个半阶段 40 帧，阶段5 为 160 帧
n_half = 40;  % 每个半阶段帧数

% ---- 阶段 1：仅 d2 变化 (0→-10→0)，d1=0, q3=0 ----
stage1_d1 = zeros(1, 2*n_half);
stage1_d2 = [linspace(0, -10, n_half), linspace(-10, 0, n_half)];
%linspace函数的语法：y = linspace(a, b, n) 在区间 [a, b] 上生成 n 个等间距的点。
stage1_q3 = zeros(1, 2*n_half);

% ---- 阶段 2：仅 d1 变化 (0→-10→0)，d2=0, q3=0 ----
stage2_d1 = [linspace(0, -10, n_half), linspace(-10, 0, n_half)];
stage2_d2 = zeros(1, 2*n_half);
stage2_q3 = zeros(1, 2*n_half);

% ---- 阶段 3：仅 q3 变化 (0→-60→0)，d1=0, d2=0 ----
stage3_d1 = zeros(1, 2*n_half);
stage3_d2 = zeros(1, 2*n_half);
stage3_q3 = [linspace(0, -60, n_half), linspace(-60, 0, n_half)];

% ---- 阶段 4：d1,d2 同步变化 (0→-10→0)，q3=0 ----
stage4_d1 = [linspace(0, -10, n_half), linspace(-10, 0, n_half)];
stage4_d2 = [linspace(0, -10, n_half), linspace(-10, 0, n_half)];
stage4_q3 = zeros(1, 2*n_half);

% ---- 阶段 5：d1, d2, q3 三者协同变化 (160帧) ----
% 设计：d1 和 d2 不同速度变化（均从0开始），q3 同步弯曲
% d1 较慢，d2 较快，产生差异化运动
n5 = 4 * n_half;  % 160 帧
t5 = linspace(0, 2*pi, n5);
% d1: 较慢变化，用 (1-cos) 映射到 [0, -8]，从0开始
stage5_d1 = -4 * (1 - cos(t5));
% d2: 较快变化（频率1.5倍），用 (1-cos) 映射到 [0, -10]，从0开始
stage5_d2 = -5 * (1 - cos(1.5 * t5));
% 限制 d2 在 [-10, 0] 范围内
stage5_d2 = max(stage5_d2, -10);
stage5_d2 = min(stage5_d2, 0);
% q3: 从 0 弯到 -50 再回来，从0开始
stage5_q3 = -25 * (1 - cos(t5));

% ---- 拼接所有阶段 ----
delta_d1_seq = [stage1_d1, stage2_d1, stage3_d1, stage4_d1, stage5_d1];
delta_d2_seq = [stage1_d2, stage2_d2, stage3_d2, stage4_d2, stage5_d2];
q3_seq       = [stage1_q3, stage2_q3, stage3_q3, stage4_q3, stage5_q3];

num_frames = length(delta_d1_seq);

% 阶段边界（用于标注）
stage_bounds = [1, 2*n_half;                                    % 阶段1
                2*n_half+1, 4*n_half;                           % 阶段2
                4*n_half+1, 6*n_half;                           % 阶段3
                6*n_half+1, 8*n_half;                           % 阶段4
                8*n_half+1, 8*n_half+n5];                       % 阶段5

stage_names = {'阶段1: 仅d2变化', ...
               '阶段2: 仅d1变化', ...
               '阶段3: 仅q3变化', ...
               '阶段4: d1+d2同步', ...
               '阶段5: d1+d2+q3协同'};

fprintf('=== 运动规划 ===\n');
fprintf('总帧数: %d\n', num_frames);
for s = 1:5
    fprintf('  %s: 帧 %d-%d (%d帧)\n', stage_names{s}, ...
            stage_bounds(s,1), stage_bounds(s,2), ...
            stage_bounds(s,2)-stage_bounds(s,1)+1);
end

%% 2. 生成所有点坐标
fprintf('\n=== 三维正运动学坐标生成 ===\n');

[all_points, all_info] = fk_generate_points(delta_d1_seq, delta_d2_seq, q3_seq);

%% 3. 加载参数
p = params();

%% 4. 颜色与样式定义
% 各组连杆颜色
color_base    = [0.4, 0.4, 0.4];   % 基座框架 - 深灰
color_parallel = [0.0, 0.45, 0.74]; % MCP 并联机构 - 蓝色
color_motor1  = [0.85, 0.33, 0.10]; % 电机驱动 1 - 橙色
color_motor2  = [0.47, 0.67, 0.19]; % 电机驱动 2 - 绿色
color_chain   = [0.64, 0.08, 0.18]; % 三杆机构 - 深红

% 关键点颜色
color_O  = [0, 0, 0];              % 原点 - 黑色
color_P  = [1, 0, 0];              % P 点 - 红色
color_AB = [0, 0.75, 0.75];        % A/B 点 - 青色
color_C  = [0.93, 0.69, 0.13];     % C 点 - 金色
color_M  = [0, 0.8, 0];            % M 点 - 绿色
color_N  = [0.8, 0.8, 0];          % N 点 - 黄色
color_T  = [0.8, 0, 0.8];          % T 点 - 品红

% 线宽
lw_base    = 2.0;
lw_parallel = 2.5;
lw_motor   = 2.0;
lw_chain   = 3.0;

% 点大小
ms_small = 7;
ms_large = 10;

%% 5. 三维视角设置
% 初始视角（方位角、仰角）
az_init = -45;
el_init = 25;

% 视角缓慢旋转（每帧旋转角度）
az_rotate_per_frame = 360 / num_frames;  % 整个动画旋转一圈

%% 6. 生成三维动画帧
fprintf('\n=== 生成三维 MP4 视频 ===\n');

% 创建视频写入器（逐帧写入，节省内存）
video_filename = 'fk_animation_3d.mp4';
v = VideoWriter(video_filename, 'MPEG-4');
v.FrameRate = 8;    % 帧率（约对应原 DelayTime=0.12）
v.Quality = 95;     % 视频质量
open(v);

% 收集末端轨迹
T_trail = nan(num_frames, 3);

for j = 1:num_frames
    if isempty(all_points(j).O)
        fprintf('跳过帧 %d（求解失败）\n', j);
        continue;
    end
    
    pts  = all_points(j);
    info = all_info(j);
    
    % 记录末端轨迹
    T_trail(j, :) = pts.T(:)';
    
    % 创建图形
    fig = figure('Visible', 'off', 'Color', 'w');
    %fig的类型是matlab.ui.Figure，是MATLAB中的一个类，
    % 用于创建和管理图形窗口。函数句柄figure()用于创建一个新的图形窗口，并返回一个指向该窗口的句柄。
    set(fig, 'Units', 'pixels', 'Position', [100, 50, 1000, 800]);
    
    ax = axes('Parent', fig);
    hold(ax, 'on');
    
    %% 6.1 绘制基座框架（灰色）
    draw_link(ax, pts.O, pts.P,  color_base, lw_base);
    draw_link(ax, pts.O, pts.A1, color_base, lw_base);
    draw_link(ax, pts.O, pts.A2, color_base, lw_base);
    draw_link(ax, pts.P, pts.A1, color_base, lw_base);
    draw_link(ax, pts.P, pts.A2, color_base, lw_base);
    
    %% 6.2 绘制 MCP 并联机构（蓝色）
    draw_link(ax, pts.P,  pts.B1, color_parallel, lw_parallel);
    draw_link(ax, pts.P,  pts.B2, color_parallel, lw_parallel);
    draw_link(ax, pts.B1, pts.B2, color_parallel, lw_parallel);
    
    %% 6.3 绘制电机驱动连杆
    % 电机 1 侧（橙色）
    draw_link(ax, pts.A1, pts.C1, color_motor1, lw_motor, '--');
    draw_link(ax, pts.C1, pts.B1, color_motor1, lw_motor);
    
    % 电机 2 侧（绿色）
    draw_link(ax, pts.A2, pts.C2, color_motor2, lw_motor, '--');
    draw_link(ax, pts.C2, pts.B2, color_motor2, lw_motor);
    
    %% 6.4 绘制三杆机构（深红色，加粗）
    draw_link(ax, pts.P, pts.M, color_chain, lw_chain);
    draw_link(ax, pts.M, pts.N, color_chain, lw_chain);
    draw_link(ax, pts.N, pts.T, color_chain, lw_chain);
    
    %% 6.5 绘制关键点
    draw_point(ax, pts.O,  color_O,  ms_small, 'o');   % O
    draw_point(ax, pts.P,  color_P,  ms_large, 'o');   % P
    draw_point(ax, pts.A1, color_AB, ms_small, 's');    % A1
    draw_point(ax, pts.A2, color_AB, ms_small, 's');    % A2
    draw_point(ax, pts.B1, color_AB, ms_small, 'd');    % B1
    draw_point(ax, pts.B2, color_AB, ms_small, 'd');    % B2
    draw_point(ax, pts.C1, color_C,  ms_small, '^');    % C1
    draw_point(ax, pts.C2, color_C,  ms_small, '^');    % C2
    draw_point(ax, pts.M,  color_M,  ms_small, 'o');    % M
    draw_point(ax, pts.N,  color_N,  ms_small, 'o');    % N
    draw_point(ax, pts.T,  color_T,  ms_large, 'p');    % T（五角星）
    
    %% 6.6 绘制点标签
    offset = 2;  % 标签偏移量 (mm)
    add_label(ax, pts.O,  'O',  offset);
    add_label(ax, pts.P,  'P',  offset);
    add_label(ax, pts.A1, 'A1', offset);
    add_label(ax, pts.A2, 'A2', offset);
    add_label(ax, pts.B1, 'B1', offset);
    add_label(ax, pts.B2, 'B2', offset);
    add_label(ax, pts.C1, 'C1', offset);
    add_label(ax, pts.C2, 'C2', offset);
    add_label(ax, pts.M,  'M',  offset);
    add_label(ax, pts.N,  'N',  offset);
    add_label(ax, pts.T,  'T',  offset);
    
    %% 6.7 绘制末端轨迹（已走过的部分，阶段5重新开始）
    stage5_start = stage_bounds(5, 1);
    if j >= stage5_start
        % 阶段5：清除之前轨迹，从阶段5起点重新画
        trail_range = T_trail(stage5_start:j, :);
    else
        trail_range = T_trail(1:j, :);
    end
    valid_trail = trail_range(~any(isnan(trail_range), 2), :);
    if size(valid_trail, 1) > 1
        plot3(ax, valid_trail(:,1), valid_trail(:,2), valid_trail(:,3), ...
              '-', 'Color', [color_T, 0.5], 'LineWidth', 1.5);
    end
    
    %% 6.8 绘制坐标轴参考（在原点处画小坐标轴）
    arrow_len = 8;
    quiver3(ax, 0, 0, 0, arrow_len, 0, 0, 0, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
    quiver3(ax, 0, 0, 0, 0, arrow_len, 0, 0, 'g', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
    quiver3(ax, 0, 0, 0, 0, 0, arrow_len, 0, 'b', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
    text(ax, arrow_len+1, 0, 0, 'X', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 9);
    text(ax, 0, arrow_len+1, 0, 'Y', 'Color', [0 0.6 0], 'FontWeight', 'bold', 'FontSize', 9);
    text(ax, 0, 0, arrow_len+1, 'Z', 'Color', 'b', 'FontWeight', 'bold', 'FontSize', 9);
    
    %% 6.9 绘制半透明基座平面（Z=0 平面参考）
    base_size = 20;
    fill3(ax, [-base_size, base_size, base_size, -base_size], ...
              [-base_size, -base_size, base_size, base_size], ...
              [0, 0, 0, 0], ...
              [0.9, 0.9, 0.9], 'FaceAlpha', 0.3, 'EdgeColor', [0.7, 0.7, 0.7]);
    
    %% 6.10 设置坐标轴属性
    axis(ax, 'equal');
    xlim(ax, [-60, 60]);
    ylim(ax, [-40, 100]);
    zlim(ax, [-10, 150]);
    
    grid(ax, 'on');
    ax.GridAlpha = 0.3;
    ax.GridLineStyle = ':';
    
    xlabel(ax, 'X (mm)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel(ax, 'Y (mm)', 'FontSize', 11, 'FontWeight', 'bold');
    zlabel(ax, 'Z (mm)', 'FontSize', 11, 'FontWeight', 'bold');
    
    % 三维视角（缓慢旋转）
    az_current = az_init + (j - 1) * az_rotate_per_frame;
    view(ax, az_current, el_init);
    
    % 光照效果
    lighting(ax, 'gouraud');
    camlight(ax, 'headlight');
    
    %% 6.11 标题 - 包含阶段名称
    % 判断当前阶段
    current_stage = 1;
    for s = 1:5
        if j >= stage_bounds(s,1) && j <= stage_bounds(s,2)
            current_stage = s;
            break;
        end
    end
    
    title(ax, sprintf('三指手 MCP 正运动学 - %s (帧 %d/%d)', ...
          stage_names{current_stage}, j, num_frames), ...
          'FontSize', 13, 'FontWeight', 'bold');
    
    %% 6.12 信息文本框 - 因变量 d1, d2, q3 醒目显示
    
    % ---- 顶部醒目区域：因变量 Δd1, Δd2, q3（大字体、彩色背景）----
    % 根据变化量决定颜色：变化中的变量用醒目颜色
    dd1_val = delta_d1_seq(j);
    dd2_val = delta_d2_seq(j);
    q3_val  = q3_seq(j);
    
    % 构建醒目的控制量 + 末端T 字符串
    str_input = sprintf(['\\bf\\fontsize{14}控制量:  ' ...
                         '\\color[rgb]{0.85,0.33,0.10}\\Deltad1 = %+.2f mm    ' ...
                         '\\color[rgb]{0.47,0.67,0.19}\\Deltad2 = %+.2f mm    ' ...
                         '\\color[rgb]{0.64,0.08,0.18}q3 = %+.1f°\n' ...
                         '\\color{black}\\fontsize{12}末端T:  [%.2f, %.2f, %.2f] mm'], ...
                         dd1_val, dd2_val, q3_val, ...
                         pts.T(1), pts.T(2), pts.T(3));
    
    % 标题下方：控制量 + 末端T 醒目显示（向下移动避免遮挡标题）
    annotation(fig, 'textbox', [0.05, 0.83, 0.90, 0.10], ...
        'String', str_input, ...
        'Interpreter', 'tex', ...
        'FitBoxToText', 'on', 'FontSize', 12, ...
        'HorizontalAlignment', 'center', ...
        'BackgroundColor', [1 1 0.85], 'EdgeColor', [0.8 0.6 0.0], ...
        'LineWidth', 2);
    
    % ---- 左下角：关节角（因变量结果）----
    str_joints = sprintf(['关节角 (°):\n' ...
                          'q1 = %+.2f   q2 = %+.2f\n' ...
                          'q3 = %+.2f   q4 = %+.2f'], ...
                          info.q1, info.q2, info.q3, info.q4);
    
    annotation(fig, 'textbox', [0.01, 0.01, 0.28, 0.12], ...
        'String', str_joints, ...
        'FitBoxToText', 'on', 'FontSize', 9, ...
        'BackgroundColor', 'w', 'EdgeColor', [0.5 0.5 0.5]);
    
    % ---- 右下角：电机绝对位移 ----
    str_motor = sprintf(['电机绝对位移 (mm):\n' ...
                         'd1 = %.3f\n' ...
                         'd2 = %.3f'], ...
                         info.d1, info.d2);
    %sprintf和fprintf的区别：sprintf函数将格式化的数据转换为字符串，而fprintf函数将格式化的数据输出到命令窗口或文件中。
    annotation(fig, 'textbox', [0.78, 0.01, 0.21, 0.10], ...
        'String', str_motor, ...
        'FitBoxToText', 'on', 'FontSize', 9, ...
        'BackgroundColor', 'w', 'EdgeColor', [0.5 0.5 0.5]);
    
    % ---- 中下方：阶段信息 ----
    str_stage = sprintf('%s\n帧 %d/%d (阶段内 %d/%d)', ...
                        stage_names{current_stage}, j, num_frames, ...
                        j - stage_bounds(current_stage,1) + 1, ...
                        stage_bounds(current_stage,2) - stage_bounds(current_stage,1) + 1);
    
    annotation(fig, 'textbox', [0.35, 0.01, 0.30, 0.08], ...
        'String', str_stage, ...
        'FitBoxToText', 'on', 'FontSize', 9, ...
        'HorizontalAlignment', 'center', ...
        'BackgroundColor', [1 1 0.9], 'EdgeColor', [0.5 0.5 0.5]);
    
    %% 6.13 图例（右上角）
    % 用不可见的线创建图例
    h_leg = [];
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', color_base,     'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', color_parallel, 'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', color_motor1,   'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', color_motor2,   'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', color_chain,    'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', color_T,        'LineWidth', 1.5);
    
    legend(ax, h_leg, {'基座框架', 'MCP并联机构', '电机1驱动', '电机2驱动', '三杆机构', '末端轨迹'}, ...
           'Location', 'northeast', 'FontSize', 8, 'FontName', 'Microsoft YaHei');
    
    %% 6.14 捕获帧并写入视频
    drawnow;
    frame = getframe(fig);
    writeVideo(v, frame);
    fprintf('帧 %d/%d 完成\n', j, num_frames);
    close(fig);
end

%% 7. 关闭视频文件
close(v);
fprintf('\n三维正运动学视频已保存：%s\n', video_filename);

%% 8. 保存所有参数到 MAT 文件
fprintf('\n=== 保存可视化参数到 MAT 文件 ===\n');

% 输入序列参数
vis_params.num_frames    = num_frames;
vis_params.q3_seq        = q3_seq;
vis_params.delta_d1_seq  = delta_d1_seq;
vis_params.delta_d2_seq  = delta_d2_seq;

% 阶段信息
vis_params.stage_bounds  = stage_bounds;
vis_params.stage_names   = {stage_names};

% 机构参数
vis_params.mechanism_params = p;

% 所有帧的点坐标
vis_params.all_points = all_points;

% 所有帧的求解信息（关节角、电机位移等）
vis_params.all_info = all_info;

% 末端轨迹数据
T_data = zeros(num_frames, 3);
q1_data = zeros(num_frames, 1);
q2_data = zeros(num_frames, 1);
q3_data = zeros(num_frames, 1);
q4_data = zeros(num_frames, 1);
d1_data = zeros(num_frames, 1);
d2_data = zeros(num_frames, 1);

for i = 1:num_frames
    if ~isempty(all_points(i).O)
        T_data(i, :) = all_points(i).T(:)';
        q1_data(i) = all_info(i).q1;
        q2_data(i) = all_info(i).q2;
        q3_data(i) = all_info(i).q3;
        q4_data(i) = all_info(i).q4;
        d1_data(i) = all_info(i).d1;
        d2_data(i) = all_info(i).d2;
    end
end

vis_params.T_data  = T_data;
vis_params.q1_data = q1_data;
vis_params.q2_data = q2_data;
vis_params.q3_data = q3_data;
vis_params.q4_data = q4_data;
vis_params.d1_data = d1_data;
vis_params.d2_data = d2_data;

% 视角参数
vis_params.az_init = az_init;
vis_params.el_init = el_init;
vis_params.az_rotate_per_frame = az_rotate_per_frame;

% 保存
mat_filename = 'fk_visualize_3d_params.mat';
save(mat_filename, '-struct', 'vis_params');
fprintf('可视化参数已保存：%s\n', mat_filename);

% 显示保存内容摘要
fprintf('\nMAT 文件包含以下字段：\n');
fields = fieldnames(vis_params);
for i = 1:length(fields)
    val = vis_params.(fields{i});
    if isnumeric(val)
        fprintf('  %-25s  [%s]  size: %s\n', fields{i}, class(val), mat2str(size(val)));
    elseif isstruct(val)
        fprintf('  %-25s  [struct]  %d 个字段\n', fields{i}, length(fieldnames(val)));
    else
        fprintf('  %-25s  [%s]\n', fields{i}, class(val));
    end
end

%% 9. 显示末端位姿摘要
fprintf('\n=== 末端 T 点位姿摘要 ===\n');
fprintf('帧数：%d 帧\n', num_frames);
fprintf('T 点范围:\n');
fprintf('  X: [%.3f, %.3f] mm\n', min(T_data(:,1)), max(T_data(:,1)));
fprintf('  Y: [%.3f, %.3f] mm\n', min(T_data(:,2)), max(T_data(:,2)));
fprintf('  Z: [%.3f, %.3f] mm\n', min(T_data(:,3)), max(T_data(:,3)));
fprintf('关节角范围:\n');
fprintf('  q1: [%.3f, %.3f]°\n', min(q1_data), max(q1_data));
fprintf('  q2: [%.3f, %.3f]°\n', min(q2_data), max(q2_data));
fprintf('  q3: [%.3f, %.3f]°\n', min(q3_data), max(q3_data));
fprintf('  q4: [%.3f, %.3f]°\n', min(q4_data), max(q4_data));
fprintf('因变量范围:\n');
fprintf('  Δd1: [%.3f, %.3f] mm\n', min(delta_d1_seq), max(delta_d1_seq));
fprintf('  Δd2: [%.3f, %.3f] mm\n', min(delta_d2_seq), max(delta_d2_seq));
fprintf('  q3:  [%.3f, %.3f]°\n', min(q3_seq), max(q3_seq));

fprintf('\n=== 三维可视化完成 ===\n');

%% =========================================================================
%   辅助绘图函数
% =========================================================================

function draw_link(ax, p1, p2, color, linewidth, linestyle)
%DRAW_LINK 绘制连杆（3D 线段）
    if nargin < 6
        linestyle = '-';
    end
    p1 = p1(:)'; p2 = p2(:)';
    plot3(ax, [p1(1), p2(1)], [p1(2), p2(2)], [p1(3), p2(3)], ...
          linestyle, 'Color', color, 'LineWidth', linewidth);
end

function draw_point(ax, pt, color, markersize, marker)
%DRAW_POINT 绘制关键点
    pt = pt(:)';
    plot3(ax, pt(1), pt(2), pt(3), marker, ...
          'MarkerSize', markersize, 'MarkerFaceColor', color, ...
          'MarkerEdgeColor', color * 0.6, 'LineWidth', 1.2);
end

function add_label(ax, pt, label_str, offset)
%ADD_LABEL 在点旁添加文字标签
    pt = pt(:)';
    text(ax, pt(1) + offset, pt(2) + offset, pt(3) + offset, label_str, ...
         'FontSize', 8, 'FontWeight', 'bold', 'Color', [0.2, 0.2, 0.2]);
end
