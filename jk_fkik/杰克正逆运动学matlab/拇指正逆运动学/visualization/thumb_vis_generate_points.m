%% =========================================================================
%   拇指机构可视化坐标生成（基于逆解求得的关节角）
%   不修改任何核心代码，只读调用正解函数
% =========================================================================
%
%   输入：vis_ik_solve_data.mat
%   输出：vis_points_data.mat
%
%   依赖（只读调用）：
%     - thumb_fk_main.m
%     - params.m
% =========================================================================

close all;
clear;
clc;

%% 1. 加载逆解数据
try
    data = load('visualization/vis_ik_solve_data.mat');
    vis_ik_solve_data = data.vis_ik_solve_data;
catch
    error('请先运行 thumb_vis_ik_solve.m 生成逆解数据');
end

% 使用逆解结果（这才是真正的逆解可视化）
q_sequence = vis_ik_solve_data.q_sequence_ik;
success = vis_ik_solve_data.success;
num_frames = vis_ik_solve_data.num_frames;

%% 2. 生成所有点坐标
all_points = struct('O',{}, 'M',{}, 'N',{}, 'T',{}, 'n_c',{}, 'phi',{});
all_points = repmat(all_points, 1, num_frames);

fprintf('=== 生成可视化坐标 (%d 帧，使用逆解结果) ===\n', num_frames);
for i = 1:num_frames
    if ~success(i)
        continue;
    end
    
    q = q_sequence(i, :);
    
    % 调用核心正解函数（只读）
    [T_end, P_end, R_end, fk_info] = thumb_fk_main(q(1), q(2), q(3), q(4));
    
    % 存储关键点坐标
    all_points(i).O = [0, 0, 0];
    all_points(i).M = fk_info.P_M';
    all_points(i).N = fk_info.P_N';
    all_points(i).T = P_end';
    all_points(i).n_c = fk_info.n_c';
    all_points(i).phi = fk_info.phi;
    
    if mod(i, 20) == 0
        fprintf('帧 %d: O-M-N-T 坐标生成完成 ✓\n', i);
    end
end

%% 3. 保存数据
vis_points_data.all_points = all_points;
vis_points_data.q_sequence = q_sequence;
vis_points_data.success = success;
vis_points_data.num_frames = num_frames;

save('visualization/vis_points_data.mat', 'vis_points_data');
fprintf('\n坐标数据已保存：visualization/vis_points_data.mat\n');
