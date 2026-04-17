%% =========================================================================
%   逆运动学轨迹规划：正方形 + 圆形
%   末端T在三维空间中沿指定平面运动，通过IK求解控制量(d1,d2,q3)
%   依赖：params.m, ik_q1q2_new.m, ik_d1_d2.m
% =========================================================================
close all; clear; clc;

%% 1. 参数设置
p = params();
n_sq  = 120;  % 正方形采样点数（每边30个）
n_cir = 100;  % 圆形采样点数

%% 2. 生成正方形轨迹
% 中心(0,50,100), 边长30, 法线(0,1,1)
sq_center = [0, 50, 100];
sq_side   = 30;
sq_normal = [0, 1, 1];

sq_pts = gen_square_3d(sq_center, sq_side, sq_normal, n_sq);

%% 3. 生成圆形轨迹
% 中心(0,70,70), 半径30, 法线(0,2,1)
cir_center = [0, 70, 70];
cir_radius = 30;
cir_normal = [0, 2, 1];

cir_pts = gen_circle_3d(cir_center, cir_radius, cir_normal, n_cir);

%% 4. 对正方形轨迹求解IK
fprintf('=== 正方形轨迹 IK 求解 ===\n');
[sq_d1, sq_d2, sq_q3, sq_ok, sq_info] = solve_ik_trajectory(sq_pts, p);
%对正方形的 120 个点逐个进行逆运动学求解
fprintf('成功: %d/%d\n\n', sum(sq_ok), n_sq);

%% 5. 对圆形轨迹求解IK
fprintf('=== 圆形轨迹 IK 求解 ===\n');
[cir_d1, cir_d2, cir_q3, cir_ok, cir_info] = solve_ik_trajectory(cir_pts, p);
fprintf('成功: %d/%d\n\n', sum(cir_ok), n_cir);

%% 6. 保存结果
% 正方形
square_traj.T_points = sq_pts;
square_traj.delta_d1 = sq_d1;
square_traj.delta_d2 = sq_d2;
square_traj.q3       = sq_q3;
square_traj.success  = sq_ok;
square_traj.info     = sq_info;
square_traj.center   = sq_center;
square_traj.side     = sq_side;
square_traj.normal   = sq_normal;
square_traj.n_points = n_sq;

% 圆形
circle_traj.T_points = cir_pts;
circle_traj.delta_d1 = cir_d1;
circle_traj.delta_d2 = cir_d2;
circle_traj.q3       = cir_q3;
circle_traj.success  = cir_ok;
circle_traj.info     = cir_info;
circle_traj.center   = cir_center;
circle_traj.radius   = cir_radius;
circle_traj.normal   = cir_normal;
circle_traj.n_points = n_cir;

save('ik_trajectory_data.mat', 'square_traj', 'circle_traj');
fprintf('轨迹数据已保存: ik_trajectory_data.mat\n');

% 同时保存为fk_visualize_3d可直接使用的格式
% 正方形
sq_v = find(sq_ok);
sq_delta_d1_seq = sq_d1(sq_v)';
sq_delta_d2_seq = sq_d2(sq_v)';
sq_q3_seq       = sq_q3(sq_v)';

% 圆形
cir_v = find(cir_ok);
cir_delta_d1_seq = cir_d1(cir_v)';
cir_delta_d2_seq = cir_d2(cir_v)';
cir_q3_seq       = cir_q3(cir_v)';

save('ik_traj_square_seq.mat', 'sq_delta_d1_seq', 'sq_delta_d2_seq', 'sq_q3_seq');
save('ik_traj_circle_seq.mat', 'cir_delta_d1_seq', 'cir_delta_d2_seq', 'cir_q3_seq');
fprintf('控制量序列已保存: ik_traj_square_seq.mat, ik_traj_circle_seq.mat\n');

%% 7. 可视化
fig = figure('Color','w','Position',[50,50,1400,600]);

% 7.1 正方形轨迹3D
subplot(1,3,1);
plot3(sq_pts(:,1), sq_pts(:,2), sq_pts(:,3), 'b.-', 'MarkerSize', 8);
hold on;
plot3(sq_center(1), sq_center(2), sq_center(3), 'r*', 'MarkerSize', 15);
% 标记失败点
fail_sq = find(~sq_ok);
if ~isempty(fail_sq)
    plot3(sq_pts(fail_sq,1), sq_pts(fail_sq,2), sq_pts(fail_sq,3), ...
          'rx', 'MarkerSize', 10, 'LineWidth', 2);
end
xlabel('X'); ylabel('Y'); zlabel('Z');
title(sprintf('正方形轨迹 (成功%d/%d)', sum(sq_ok), n_sq));
grid on; axis equal; view(-30, 25);

% 7.2 圆形轨迹3D
subplot(1,3,2);
plot3(cir_pts(:,1), cir_pts(:,2), cir_pts(:,3), 'g.-', 'MarkerSize', 8);
hold on;
plot3(cir_center(1), cir_center(2), cir_center(3), 'r*', 'MarkerSize', 15);
fail_cir = find(~cir_ok);
if ~isempty(fail_cir)
    plot3(cir_pts(fail_cir,1), cir_pts(fail_cir,2), cir_pts(fail_cir,3), ...
          'rx', 'MarkerSize', 10, 'LineWidth', 2);
end
xlabel('X'); ylabel('Y'); zlabel('Z');
title(sprintf('圆形轨迹 (成功%d/%d)', sum(cir_ok), n_cir));
grid on; axis equal; view(-30, 25);

% 7.3 控制量曲线
subplot(1,3,3);
hold on;
% 正方形
yyaxis left;
plot(sq_d1, 'b-', 'DisplayName', '正方形 Δd1');
plot(sq_d2, 'b--', 'DisplayName', '正方形 Δd2');
ylabel('Δd1, Δd2 (mm)');
yyaxis right;
plot(sq_q3, 'r-', 'DisplayName', '正方形 q3');
ylabel('q3 (°)');
xlabel('采样点');
title('正方形控制量');
legend('Location','best'); grid on;

saveas(fig, 'ik_trajectory_plan.png');
fprintf('轨迹图已保存: ik_trajectory_plan.png\n');

% 7.4 单独画圆形控制量
fig2 = figure('Color','w','Position',[100,100,600,400]);
hold on;
yyaxis left;
plot(cir_d1, 'b-', 'DisplayName', 'Δd1');
plot(cir_d2, 'b--', 'DisplayName', 'Δd2');
ylabel('Δd1, Δd2 (mm)');
yyaxis right;
plot(cir_q3, 'r-', 'DisplayName', 'q3');
ylabel('q3 (°)');
xlabel('采样点'); title('圆形轨迹控制量');
legend('Location','best'); grid on;
saveas(fig2, 'ik_trajectory_circle_ctrl.png');

fprintf('\n=== 轨迹规划完成 ===\n');

%% =========================================================================
%   辅助函数
% =========================================================================

function pts = gen_square_3d(center, side, normal, n_pts)
%GEN_SQUARE_3D 在三维空间中生成正方形轨迹点
    % 构建平面局部坐标系
    normal = normal(:)' / norm(normal);
    [u, v] = get_plane_axes(normal);
    
    half = side / 2;
    n_per_side = ceil(n_pts / 4);
    
    % 四个顶点（局部坐标）
    corners_local = [-half, -half;
                      half, -half;
                      half,  half;
                     -half,  half];
    
    pts_local = [];
    for i = 1:4
        j = mod(i, 4) + 1;
        t = linspace(0, 1, n_per_side + 1);
        t = t(1:end-1);  % 去掉终点避免重复
        seg = corners_local(i,:)' + (corners_local(j,:)' - corners_local(i,:)') * t;
        pts_local = [pts_local, seg];
    end
    
    % 截取到n_pts个点
    if size(pts_local, 2) > n_pts
        pts_local = pts_local(:, 1:n_pts);
    end
    
    % 转换到三维空间
    center = center(:)';
    pts = zeros(size(pts_local, 2), 3);
    for i = 1:size(pts_local, 2)
        pts(i,:) = center + pts_local(1,i) * u + pts_local(2,i) * v;
    end
end

function pts = gen_circle_3d(center, radius, normal, n_pts)
%GEN_CIRCLE_3D 在三维空间中生成圆形轨迹点
    normal = normal(:)' / norm(normal);
    [u, v] = get_plane_axes(normal);
    
    theta = linspace(0, 2*pi, n_pts + 1);
    theta = theta(1:end-1);
    
    center = center(:)';
    pts = zeros(n_pts, 3);
    for i = 1:n_pts
        pts(i,:) = center + radius * cos(theta(i)) * u + radius * sin(theta(i)) * v;
    end
end

function [u, v] = get_plane_axes(normal)
%GET_PLANE_AXES 根据法线向量构建平面内的两个正交单位向量
    normal = normal(:)' / norm(normal);
    
    % 选择一个不平行于normal的参考向量
    if abs(dot(normal, [1,0,0])) < 0.9
        ref = [1, 0, 0];
    else
        ref = [0, 1, 0];
    end
    
    u = cross(normal, ref);
    u = u / norm(u);
    v = cross(normal, u);
    v = v / norm(v);
end

function [d1, d2, q3, ok, info_all] = solve_ik_trajectory(T_pts, p)
%SOLVE_IK_TRAJECTORY 对轨迹点批量求解IK
    n = size(T_pts, 1);
    d1 = nan(n, 1);
    d2 = nan(n, 1);
    q3 = nan(n, 1);
    ok = false(n, 1);
    info_all = struct('q1',cell(n,1),'q2',cell(n,1),'q3',cell(n,1),'q4',cell(n,1));
    
    for i = 1:n
        T = T_pts(i, :);
        
        % IK求解: T → (q1, q2, q3, q4)
        [q1_sol, q2_sol, info_ik] = ik_q1q2_new(T, p);
        
        if ~info_ik.success || isempty(q1_sol)
            fprintf('  点%d: IK失败 T=[%.1f,%.1f,%.1f]\n', i, T(1),T(2),T(3));
            continue;
        end
        
        q1_ik = q1_sol(1);
        q2_ik = q2_sol(1);
        q3_ik = info_ik.q3;
        
        % 求d1, d2
        [~, ~, ~, ~, dd1, dd2, info_d] = ik_d1_d2(q1_ik, q2_ik, p);
        
        if ~info_d.success
            fprintf('  点%d: d1d2求解失败\n', i);
            continue;
        end
        
        d1(i) = dd1;
        d2(i) = dd2;
        q3(i) = q3_ik;
        ok(i) = true;
        
        info_all(i).q1 = q1_ik;
        info_all(i).q2 = q2_ik;
        info_all(i).q3 = q3_ik;
        info_all(i).q4 = info_ik.q4;
        
        if mod(i, 20) == 0
            fprintf('  点%d/%d: d1=%.3f d2=%.3f q3=%.2f\n', i, n, dd1, dd2, q3_ik);
        end
    end
end
