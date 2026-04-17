%% =========================================================================
%   拇指机构逆运动学 - 三维可视化与 AVI 视频生成
%   仿照 ILDA-MCP 正解可视化代码结构
% =========================================================================
%
%   输入：vis_points_data.mat, vis_ik_solve_data.mat, vis_joint_traj_data.mat
%   输出：thumb_ik_animation_3d.avi + thumb_ik_vis_params.mat
%
%   依赖（只读调用）：
%     - params.m
%     - vis_params.m
% =========================================================================

close all;
clear;

%% 1. 加载数据
try
    data_points = load('visualization/vis_points_data.mat');
    vis_points_data = data_points.vis_points_data;
    
    data_ik = load('visualization/vis_ik_solve_data.mat');
    vis_ik_solve_data = data_ik.vis_ik_solve_data;
    
    data_traj = load('visualization/vis_joint_traj_data.mat');
    vis_joint_traj_data = data_traj.vis_joint_traj_data;
catch
    error('请先运行 thumb_vis_generate_points.m 生成坐标数据');
end

all_points = vis_points_data.all_points;
q_sequence = vis_points_data.q_sequence;
success = vis_points_data.success;
num_frames = vis_points_data.num_frames;
stage_bounds = vis_ik_solve_data.stage_bounds;
stage_names = vis_ik_solve_data.stage_names;
error_total = vis_ik_solve_data.error_q1 + vis_ik_solve_data.error_q2 + ...
              vis_ik_solve_data.error_q3 + vis_ik_solve_data.error_q4;

%% 2. 加载参数
p = params();   % 机械参数
vp = vis_params();  % 可视化参数

%% 3. 计算每帧旋转角度
az_rotate_per_frame = 360 / num_frames;  % 整个动画旋转一圈

%% 4. 生成三维动画帧
fprintf('\n=== 生成三维 AVI 视频 ===\n');

% 创建视频写入对象
v = VideoWriter(vp.video_filename, vp.video_format);
v.FrameRate = vp.frame_rate;
open(v);

% 收集末端轨迹
T_trail = nan(num_frames, 3);

for j = 1:num_frames
    if ~success(j) || isempty(all_points(j).O)
        fprintf('跳过帧 %d（求解失败）\n', j);
        continue;
    end
    
    pts  = all_points(j);
    q    = q_sequence(j, :);
    
    % 记录末端轨迹
    T_trail(j, :) = pts.T(:)';
    
    % 创建图形
    fig = figure('Visible', vp.fig_visible, 'Color', vp.fig_color);
    set(fig, 'Units', 'pixels', 'Position', vp.fig_position);
    
    ax = axes('Parent', fig);
    hold(ax, 'on');
    
    %% 5.1 绘制连杆 O-M-N-T
    draw_link(ax, pts.O, pts.M, vp.color_OM, vp.lw_OM);
    draw_link(ax, pts.M, pts.N, vp.color_MN, vp.lw_MN);
    draw_link(ax, pts.N, pts.T, vp.color_NT, vp.lw_NT);
    
    %% 5.2 绘制关键点
    draw_point(ax, pts.O, vp.color_O, vp.ms_small, 'o');
    draw_point(ax, pts.M, vp.color_M, vp.ms_large, 'o');
    draw_point(ax, pts.N, vp.color_N, vp.ms_small, 'o');
    draw_point(ax, pts.T, vp.color_T, vp.ms_large, 'p');
    
    %% 5.3 绘制法向量（从 O 点出发）
    quiver3(ax, pts.O(1), pts.O(2), pts.O(3), ...
            pts.n_c(1)*30, pts.n_c(2)*30, pts.n_c(3)*30, ...
            'r-', 'LineWidth', vp.lw_nc, 'MaxHeadSize', 0.5);
    
    %% 5.4 绘制点标签
    offset = 3;
    add_label(ax, pts.O, 'O', offset);
    add_label(ax, pts.M, 'M', offset);
    add_label(ax, pts.N, 'N', offset);
    add_label(ax, pts.T, 'T', offset);
    
    %% 5.5 绘制末端轨迹（已走过的部分）
    trail_range = T_trail(1:j, :);
    valid_trail = trail_range(~any(isnan(trail_range), 2), :);
    if size(valid_trail, 1) > 1
        plot3(ax, valid_trail(:,1), valid_trail(:,2), valid_trail(:,3), ...
              '-', 'Color', [vp.color_traj, vp.trail_alpha], 'LineWidth', vp.lw_traj);
    end
    
    %% 5.6 绘制坐标轴参考
    quiver3(ax, 0, 0, 0, vp.arrow_len, 0, 0, 'r', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
    quiver3(ax, 0, 0, 0, 0, vp.arrow_len, 0, 'g', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
    quiver3(ax, 0, 0, 0, 0, 0, vp.arrow_len, 'b', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
    text(ax, vp.arrow_len+2, 0, 0, 'X', 'Color', vp.color_X, 'FontWeight', 'bold', 'FontSize', 9);
    text(ax, 0, vp.arrow_len+2, 0, 'Y', 'Color', vp.color_Y, 'FontWeight', 'bold', 'FontSize', 9);
    text(ax, 0, 0, vp.arrow_len+2, 'Z', 'Color', vp.color_Z, 'FontWeight', 'bold', 'FontSize', 9);
    
    %% 5.7 绘制半透明基座平面（Z=0 平面参考）
    fill3(ax, [-vp.base_size, vp.base_size, vp.base_size, -vp.base_size], ...
              [-vp.base_size, -vp.base_size, vp.base_size, vp.base_size], ...
              [0, 0, 0, 0], ...
              vp.color_base, 'FaceAlpha', 0.3, 'EdgeColor', vp.color_grid);
    
    %% 5.8 设置坐标轴属性
    if vp.axis_equal
        axis(ax, 'equal');
    end
    xlim(ax, vp.xlim);
    ylim(ax, vp.ylim);
    zlim(ax, vp.zlim);
    
    if vp.grid_on
        grid(ax, 'on');
        ax.GridAlpha = vp.grid_alpha;
        ax.GridLineStyle = vp.grid_linestyle;
    end
    
    xlabel(ax, 'X (mm)', 'FontSize', vp.font_size_label, 'FontWeight', vp.font_weight_label);
    ylabel(ax, 'Y (mm)', 'FontSize', vp.font_size_label, 'FontWeight', vp.font_weight_label);
    zlabel(ax, 'Z (mm)', 'FontSize', vp.font_size_label, 'FontWeight', vp.font_weight_label);
    
    % 三维视角（缓慢旋转）
    az_current = vp.az_init + (j - 1) * az_rotate_per_frame;
    view(ax, az_current, vp.el_init);
    
    lighting(ax, vp.lighting_method);
    camlight(ax, vp.camlight_type);
    
    %% 5.9 判断当前阶段
    current_stage = 1;
    for s = 1:5
        if j >= stage_bounds(s,1) && j <= stage_bounds(s,2)
            current_stage = s;
            break;
        end
    end
    
    %% 5.10 标题
    title(ax, sprintf('拇指机构逆运动学 - %s (帧 %d/%d)', ...
          stage_names{current_stage}, j, num_frames), ...
          'FontSize', vp.font_size_title, 'FontWeight', vp.font_weight_title);
    
    %% 5.11 信息文本框
    
    % ---- 顶部醒目区域：输入 T、phi ----
    T_val = pts.T;
    phi_val = pts.phi;
    
    % 修复：移除无效的 \color 转义字符，改用纯文本格式
    str_input = sprintf(['\\bf\\fontsize{14}输入：  ' ...
                         'T = [%.2f, %.2f, %.2f] mm    ' ...
                         '\\phi = %.2f°'], ...
                         T_val(1), T_val(2), T_val(3), phi_val);
    
    annotation(fig, 'textbox', vp.input_box_position, ...
        'String', str_input, ...
        'Interpreter', 'tex', ...
        'FitBoxToText', 'on', 'FontSize', vp.input_box_font_size, ...
        'HorizontalAlignment', 'center', ...
        'BackgroundColor', vp.input_box_bg_color, 'EdgeColor', vp.input_box_edge_color, ...
        'LineWidth', 2);
    
    % ---- 左下角：关节角 ----
    str_joints = sprintf(['关节角 (°):\n' ...
                          'q1 = %+.2f   q2 = %+.2f\n' ...
                          'q3 = %+.2f   q4 = %+.2f'], ...
                          q(1), q(2), q(3), q(4));
    
    annotation(fig, 'textbox', vp.joint_box_position, ...
        'String', str_joints, ...
        'FitBoxToText', 'on', 'FontSize', vp.joint_box_font_size, ...
        'BackgroundColor', vp.joint_box_bg_color, 'EdgeColor', vp.joint_box_edge_color);
    
    % ---- 中下方：阶段信息 + 误差 ----
    str_stage = sprintf('%s\n帧 %d/%d (阶段内 %d/%d)\n逆解误差：%.4f°', ...
                        stage_names{current_stage}, j, num_frames, ...
                        j - stage_bounds(current_stage,1) + 1, ...
                        stage_bounds(current_stage,2) - stage_bounds(current_stage,1) + 1, ...
                        error_total(j));
    
    annotation(fig, 'textbox', vp.stage_box_position, ...
        'String', str_stage, ...
        'FitBoxToText', 'on', 'FontSize', vp.stage_box_font_size, ...
        'HorizontalAlignment', 'center', ...
        'BackgroundColor', vp.stage_box_bg_color, 'EdgeColor', vp.stage_box_edge_color);
    
    % ---- 右下角：法向量信息 ----
    str_nc = sprintf(['OMNT 平面法向量:\n' ...
                      'n_c = [%.3f, %.3f, %.3f]'], ...
                      pts.n_c(1), pts.n_c(2), pts.n_c(3));
    
    annotation(fig, 'textbox', vp.nc_box_position, ...
        'String', str_nc, ...
        'FitBoxToText', 'on', 'FontSize', vp.nc_box_font_size, ...
        'BackgroundColor', vp.nc_box_bg_color, 'EdgeColor', vp.nc_box_edge_color);
    
    %% 5.12 图例（右上角）
    h_leg = [];
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', vp.color_OM, 'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', vp.color_MN, 'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', vp.color_NT, 'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', vp.color_nc, 'LineWidth', 2);
    h_leg(end+1) = plot3(ax, nan, nan, nan, '-', 'Color', vp.color_traj, 'LineWidth', 1.5);
    
    legend(ax, h_leg, vp.legend_items, ...
           'Location', vp.legend_location, 'FontSize', vp.font_size_legend, 'FontName', vp.font_name);
    
    %% 5.13 捕获帧并写入视频
    drawnow;
    frame = getframe(fig);
    writeVideo(v, frame);
    fprintf('帧 %d/%d 完成\n', j, num_frames);
    close(fig);
end

%% 6. 关闭视频文件
close(v);
fprintf('\n三维逆运动学视频已保存：%s\n', vp.video_filename);

%% 7. 保存所有参数到 MAT 文件
fprintf('\n=== 保存可视化参数到 MAT 文件 ===\n');

vis_params_struct.num_frames    = num_frames;
vis_params_struct.q1_seq        = q_sequence(:,1)';
vis_params_struct.q2_seq        = q_sequence(:,2)';
vis_params_struct.q3_seq        = q_sequence(:,3)';
vis_params_struct.q4_seq        = q_sequence(:,4)';
vis_params_struct.stage_bounds  = stage_bounds;
vis_params_struct.stage_names   = stage_names;
vis_params_struct.mechanism_params = p;
vis_params_struct.all_points    = all_points;
vis_params_struct.T_sequence    = vis_ik_solve_data.T_sequence;
vis_params_struct.phi_sequence  = vis_ik_solve_data.phi_sequence;
vis_params_struct.az_init       = vp.az_init;
vis_params_struct.el_init       = vp.el_init;
vis_params_struct.az_rotate_per_frame = az_rotate_per_frame;

mat_filename = vp.mat_filename;
save(mat_filename, '-struct', 'vis_params_struct');
fprintf('可视化参数已保存：%s\n', mat_filename);

%% 8. 显示末端位姿摘要
fprintf('\n=== 末端 T 点位姿摘要 ===\n');
T_data = vis_ik_solve_data.T_sequence';
fprintf('帧数：%d 帧\n', num_frames);
fprintf('T 点范围:\n');
fprintf('  X: [%.3f, %.3f] mm\n', min(T_data(:,1)), max(T_data(:,1)));
fprintf('  Y: [%.3f, %.3f] mm\n', min(T_data(:,2)), max(T_data(:,2)));
fprintf('  Z: [%.3f, %.3f] mm\n', min(T_data(:,3)), max(T_data(:,3)));
fprintf('关节角范围:\n');
fprintf('  q1: [%.3f, %.3f]°\n', min(q_sequence(:,1)), max(q_sequence(:,1)));
fprintf('  q2: [%.3f, %.3f]°\n', min(q_sequence(:,2)), max(q_sequence(:,2)));
fprintf('  q3: [%.3f, %.3f]°\n', min(q_sequence(:,3)), max(q_sequence(:,3)));
fprintf('  q4: [%.3f, %.3f]°\n', min(q_sequence(:,4)), max(q_sequence(:,4)));

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
