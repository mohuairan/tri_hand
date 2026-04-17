%% =========================================================================
%   ILDA 手 MCP 关节逆运动学 - 可视化与 GIF 生成
%   根据末端位置 T 的变化生成动画
%   依赖：params.m, ik_generate_points.m
% =========================================================================
%
%   输入：T_pose_reference.mat 文件中的 T 点坐标序列
%   输出：ik_animation.gif 动画文件
%
% =========================================================================

close all;
clear;

%% 1. 加载末端位姿参考文件（由 fk_generate_points.m 生成）
fprintf('=== 加载末端位姿参考文件 ===\n');

try
    data = load('T_pose_reference.mat');
    if isfield(data, 'T_data')
        T_sequence = data.T_data;
        if isfield(data, 'num_frames')
            num_frames = data.num_frames;
        else
            num_frames = size(T_sequence, 1);
        end
        fprintf('成功加载 T_pose_reference.mat：%d 帧\n', num_frames);
    else
        error('T_pose_reference.mat 中未找到 T_data 变量');
    end
catch
    error('未找到 T_pose_reference.mat 文件！请先运行 fk_generate_points.m 生成参考文件。');
end

%% 2. 生成所有点坐标
fprintf('=== 逆运动学坐标生成 ===\n');
[all_points, all_info] = ik_generate_points(T_sequence);

%% 3. 加载参数（用于绘图设置）
p = params();

%% 4. 生成动画帧
fprintf('\n=== 生成 GIF 动画 ===\n');
im = cell(1, num_frames);

for j = 1:num_frames
    if isempty(all_points(j))
        fprintf('跳过帧 %d（求解失败）\n', j);
        continue;
    end
    
    pts = all_points(j);
    info = all_info(j);
    
    % 定义可视化线段
    lines = {};
    line_idx = 0;
    
    % 基座框架
    line_idx = line_idx + 1; lines{line_idx} = [pts.O(:)'; pts.P(:)'];          % O-P
    line_idx = line_idx + 1; lines{line_idx} = [pts.O(:)'; pts.A1(:)'];         % O-A1
    line_idx = line_idx + 1; lines{line_idx} = [pts.O(:)'; pts.A2(:)'];         % O-A2
    line_idx = line_idx + 1; lines{line_idx} = [pts.P(:)'; pts.A1(:)'];         % P-A1
    line_idx = line_idx + 1; lines{line_idx} = [pts.P(:)'; pts.A2(:)'];         % P-A2
    
    % MCP 并联机构
    line_idx = line_idx + 1; lines{line_idx} = [pts.P(:)'; pts.B1(:)'];         % P-B1
    line_idx = line_idx + 1; lines{line_idx} = [pts.P(:)'; pts.B2(:)'];         % P-B2
    line_idx = line_idx + 1; lines{line_idx} = [pts.B1(:)'; pts.B2(:)'];        % B1-B2
    
    % 电机驱动
    line_idx = line_idx + 1; lines{line_idx} = [pts.A1(:)'; pts.C1(:)'];        % A1-C1
    line_idx = line_idx + 1; lines{line_idx} = [pts.A2(:)'; pts.C2(:)'];        % A2-C2
    line_idx = line_idx + 1; lines{line_idx} = [pts.C1(:)'; pts.B1(:)'];        % C1-B1
    line_idx = line_idx + 1; lines{line_idx} = [pts.C2(:)'; pts.B2(:)'];        % C2-B2
    
    % 三杆机构
    line_idx = line_idx + 1; lines{line_idx} = [pts.P(:)'; pts.M(:)'];          % P-M
    line_idx = line_idx + 1; lines{line_idx} = [pts.M(:)'; pts.N(:)'];          % M-N
    line_idx = line_idx + 1; lines{line_idx} = [pts.N(:)'; pts.T(:)'];          % N-T
    
    % 创建图形
    fig = figure('Visible', 'off');
    set(gcf, 'unit', 'normalized', 'position', [0.1, 0.05, 0.8, 0.5]);
    
    % 创建双视图
    for k = 1:2
        subplot(1, 2, k);
        
        % 绘制原点
        plot3(pts.O(1), pts.O(2), pts.O(3), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k');
        hold on;
        
        % 绘制所有线段
        for i = 1:length(lines)
            plot3(lines{i}(:,1), lines{i}(:,2), lines{i}(:,3), ...
                  'b-', 'LineWidth', 2);
        end
        
        % 绘制关键点（用不同颜色标记）
        plot3(pts.P(1), pts.P(2), pts.P(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');  % P
        plot3(pts.M(1), pts.M(2), pts.M(3), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');   % M
        plot3(pts.N(1), pts.N(2), pts.N(3), 'yo', 'MarkerSize', 8, 'MarkerFaceColor', 'y');   % N
        plot3(pts.T(1), pts.T(2), pts.T(3), 'mo', 'MarkerSize', 10, 'MarkerFaceColor', 'm');  % T
        plot3(pts.A1(1), pts.A1(2), pts.A1(3), 'cs', 'MarkerSize', 8, 'MarkerFaceColor', 'c'); % A1
        plot3(pts.A2(1), pts.A2(2), pts.A2(3), 'cs', 'MarkerSize', 8, 'MarkerFaceColor', 'c'); % A2
        plot3(pts.C1(1), pts.C1(2), pts.C1(3), 'c^', 'MarkerSize', 8, 'MarkerFaceColor', 'c'); % C1
        plot3(pts.C2(1), pts.C2(2), pts.C2(3), 'c^', 'MarkerSize', 8, 'MarkerFaceColor', 'c'); % C2
        
        % 设置坐标轴
        axis([-60 60 -60 60 0 150]);
        grid on;
        
        if k == 1
            view(0, 0);   % 正视图（YZ 平面）
            title(sprintf('逆运动学 - 正视图 (帧 %d/%d)', j, num_frames));
            xlabel('X (mm)');
            ylabel('Y (mm)');
            zlabel('Z (mm)');
        else
            view(90, 0);  % 侧视图（XZ 平面）
            title(sprintf('逆运动学 - 侧视图 (帧 %d/%d)', j, num_frames));
            xlabel('X (mm)');
            ylabel('Y (mm)');
            zlabel('Z (mm)');
        end
        
        % 添加图例 - 左侧显示关节角，右侧显示电机位移和末端位置
        % 左侧：关节角
        legend_left_pos = [0.02 0.02 0.48 0.12];
        annotation('textbox', legend_left_pos, 'String', ...
            sprintf('关节角 (度):\nq1=%.2f  q2=%.2f\nq3=%.2f  q4=%.2f', ...
                info.q1, info.q2, info.q3, info.q4), ...
            'FitBoxToText', 'on', 'FontSize', 9, 'BackgroundColor', 'white');
        
        % 右侧：电机位移和末端位置
        legend_right_pos = [0.52 0.02 0.46 0.12];
        annotation('textbox', legend_right_pos, 'String', ...
            sprintf('电机位移 (mm):\nd1=%.3f  d2=%.3f\n末端 T (mm):\n[%.2f, %.2f, %.2f]', ...
                info.d1, info.d2, pts.T(1), pts.T(2), pts.T(3)), ...
            'FitBoxToText', 'on', 'FontSize', 9, 'BackgroundColor', 'white');
    end
    
    % 捕获帧
    drawnow;
    frame = getframe(fig);
    im{j} = frame2im(frame);
    fprintf('帧 %d/%d 完成\n', j, num_frames);
    close(fig);
end

%% 5. 生成 GIF 文件
filename = 'ik_animation.gif';
fprintf('\n正在生成 GIF: %s\n', filename);

valid_frames = sum(~cellfun(@isempty, im));
if valid_frames == 0
    error('没有成功生成任何帧！');
end

for idx = 1:num_frames
    if isempty(im{idx})
        continue;
    end
    
    [A, map] = rgb2ind(im{idx}, 256);
    if idx == 1
        imwrite(A, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', 0.15);
    else
        imwrite(A, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', 0.15);
    end
end

fprintf('逆运动学动画已保存：%s (共 %d 帧)\n', filename, valid_frames);
