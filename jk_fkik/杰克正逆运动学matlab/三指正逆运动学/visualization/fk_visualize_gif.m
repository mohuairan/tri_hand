%% =========================================================================
%   ILDA 手 MCP 关节正运动学 - 可视化与 GIF 生成
%   根据电机参数 (d1, d2, q3) 的变化生成动画
%   依赖：ik_params.m, fk_generate_points.m
% =========================================================================
%
%   输入：d1 变化量、d2 变化量、q3 角度序列
%   输出：GIF 动画文件
%
% =========================================================================

close all;
clear;

%% 1. 定义输入序列（动画路径）
num_frames = 80;

% q3 从 0 往负值变化（0°到 -30°）
q3_seq = linspace(0, -30, num_frames);

% d1、d2 变化与 q3 同步进行
% 分为四个阶段：
% 阶段 1（帧 1-20）：d2 缩短 1.5mm，d1 不变
% 阶段 2（帧 21-40）：d2 回到初始值，d1 不变
% 阶段 3（帧 41-60）：d1 缩短 1.5mm，d2 不变
% 阶段 4（帧 61-80）：d1 回到初始值，d2 不变

delta_d1_seq = zeros(1, num_frames);
%zeros函数的语法：B = zeros(m, n) 创建一个 m 行 n 列的全零矩阵。
% 这里创建了一个 1 行 num_frames 列的全零向量，用于存储 d1 的变化量序列。
delta_d2_seq = zeros(1, num_frames);

% 阶段 1：d2 缩短 1.5mm（帧 1-20）
delta_d2_seq(1:20) = linspace(0, -1.5, 20);

% 阶段 2：d2 回到初始值（帧 21-40）
delta_d2_seq(21:40) = linspace(-1.5, 0, 20);

% 阶段 3：d1 缩短 1.5mm（帧 41-60）
delta_d1_seq(41:60) = linspace(0, -1.5, 20);

% 阶段 4：d1 回到初始值（帧 61-80）
delta_d1_seq(61:80) = linspace(-1.5, 0, 20);

%% 2. 生成所有点坐标
fprintf('=== 正运动学坐标生成 ===\n');
[all_points, all_info] = fk_generate_points(delta_d1_seq, delta_d2_seq, q3_seq);

%% 3. 加载参数（用于绘图设置）
p = params();

%% 4. 生成动画帧
fprintf('\n=== 生成 GIF 动画 ===\n');
im = cell(1, num_frames);
%cell函数的语法：C = cell(m, n) 创建一个 m 行 n 列的单元格数组。
% 这里创建了一个 1 行 num_frames 列的单元格数组，用于存储每一帧的图像数据。

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
            title(sprintf('正运动学 - 正视图 (帧 %d/%d)', j, num_frames));
            xlabel('X (mm)');
            ylabel('Y (mm)');
            zlabel('Z (mm)');
        else
            view(90, 0);  % 侧视图（XZ 平面）
            title(sprintf('正运动学 - 侧视图 (帧 %d/%d)', j, num_frames));
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
filename = 'fk_animation.gif';
fprintf('\n正在生成 GIF: %s\n', filename);

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

fprintf('正运动学动画已保存：%s\n', filename);

%% 6. 保存末端位姿参考文件
fprintf('\n正在生成末端位姿参考文件...\n');

% 创建表格数据
T_data = zeros(num_frames, 3);
for i = 1:num_frames
    if ~isempty(all_points(i))
        %isempty:MATLAB内置函数，用于检查数组或单元格是否为空。
        % 这里检查 all_points(i) 是否为空，如果不为空则提取 T 点位姿数据。
        T_data(i, :) = all_points(i).T;
    end
end

% 保存为 MAT 文件
save('T_pose_reference.mat', 'T_data', 'num_frames');

% 保存为 CSV 文件（方便查看）
csv_header = 'frame,Tx,Ty,Tz';
csv_data = [(1:num_frames)', T_data];
writematrix(csv_data, 'T_pose_reference.csv', 'WriteMode', 'overwrite');
%writematrix函数的语法：writematrix(A, filename) 将矩阵 A 写入文件 filename 中。
% 这里将 csv_data 矩阵写入 'T_pose_reference.csv' 文件中，'WriteMode', 'overwrite' 参数表示如果文件已存在则覆盖。

fprintf('末端位姿参考文件已保存：T_pose_reference.mat 和 T_pose_reference.csv\n');

% 显示末端位姿摘要
fprintf('\n=== 末端 T 点位姿摘要 ===\n');
fprintf('帧数：%d 帧\n', num_frames);
fprintf('T 点范围:\n');
fprintf('  X: [%.3f, %.3f] mm\n', min(T_data(:,1)), max(T_data(:,1)));
fprintf('  Y: [%.3f, %.3f] mm\n', min(T_data(:,2)), max(T_data(:,2)));
fprintf('  Z: [%.3f, %.3f] mm\n', min(T_data(:,3)), max(T_data(:,3)));
