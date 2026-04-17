%% =========================================================================
%   ILDA 手 MCP 关节逆运动学 - 坐标生成
%   根据末端位置 T 的变化，计算所有点的坐标
%   依赖：params.m, ik_q1_q2_numerical.m, ik_d1_d2.m
% =========================================================================
%
%   输入：T_sequence - N×3 矩阵，每行是一个目标位置 [Tx, Ty, Tz]
%   输出：所有点坐标结构体数组
%
% =========================================================================

function [all_points, all_info] = ik_generate_points(T_sequence)
% IK_GENERATE_POINTS 逆运动学坐标生成
%   输入：T_sequence - N×3 矩阵，每行是一个目标位置 [Tx, Ty, Tz]
%   输出：all_points - 1×N 结构体数组，包含所有点坐标
%         all_info   - 1×N 结构体数组，包含求解信息

    %% 1. 加载参数
    p = params();
    
    %% 2. 初始化
    num_frames = size(T_sequence, 1);
    
    % 预分配数组
    all_points = struct('O', {[]}, 'P', {[]}, 'A1', {[]}, 'A2', {[]}, ...
                        'B1', {[]}, 'B2', {[]}, 'C1', {[]}, 'C2', {[]}, ...
                        'M', {[]}, 'N', {[]}, 'T', {[]});
    all_points = repmat(all_points, 1, num_frames);
    
    all_info = struct('q1', {[]}, 'q2', {[]}, 'q3', {[]}, 'q4', {[]}, ...
                      'd1', {[]}, 'd2', {[]}, 'success', {false});
    all_info = repmat(all_info, 1, num_frames);
    
    %% 3. 固定点坐标（不随运动变化）
    O = [0, 0, 0];                              % 原点
    P = p.P;                                    % MCP 中心
    A1 = [p.ax, p.ay, 0];                       % 电机 1 基座
    A2 = [-p.ax, p.ay, 0];                      % 电机 2 基座
    
    %% 4. 循环求解每一帧
    success_count = 0;
    for i = 1:num_frames
        T = T_sequence(i, :)';  % 当前目标位置
        
        fprintf('逆解帧 %d/%d: T = [%.2f, %.2f, %.2f]\n', ...
            i, num_frames, T(1), T(2), T(3));
        
        %% 4.1 求解 q1、q2（同时得到 q3、q4）
        [q1_sol, q2_sol, info_q] = ik_q1_q2_numerical(T, [0, 0], p);
        
        if ~info_q.success || length(q1_sol) == 0
            fprintf('  -> 求解失败，跳过\n');
            continue;
        end
        
        % 取第一组解
        q1 = q1_sol(1);
        q2 = q2_sol(1);
        q3 = info_q.q3;
        q4 = info_q.q4;
        
        %% 4.2 求解 d1、d2
        [d1_i, d2_i, d1_n, d2_n, dd1, dd2, info_d] = ik_d1_d2(q1, q2, p);
        
        if ~info_d.success
            fprintf('  -> d1/d2 求解失败，跳过\n');
            continue;
        end
        
        d1_abs = d1_i + dd1;
        d2_abs = d2_i + dd2;
        
        %% 4.3 计算 B1、B2 点（动平台铰链点）
        c1 = cosd(q1); s1 = sind(q1);
        c2 = cosd(q2); s2 = sind(q2);
        
        bx = p.bx; by = p.by; bz = p.bz;
        px = P(1); py = P(2); pz = P(3);
        
        b1x_prime = c2 * bx + s2 * s1 * by + s2 * c1 * bz;
        b1y_prime = c1 * by - s1 * bz;
        b1z_prime = -s2 * bx + c2 * s1 * by + c2 * c1 * bz;
        
        b2x_prime = -c2 * bx + s2 * s1 * by + s2 * c1 * bz;
        b2y_prime = c1 * by - s1 * bz;
        b2z_prime = s2 * bx + c2 * s1 * by + c2 * c1 * bz;
        
        B1 = [px + b1x_prime, py + b1y_prime, pz + b1z_prime];
        B2 = [px + b2x_prime, py + b2y_prime, pz + b2z_prime];
        
        %% 4.4 计算 C1、C2 点（电机端点）
        C1 = [A1(1), A1(2), d1_abs];
        C2 = [A2(1), A2(2), d2_abs];
        
        %% 4.5 计算 M、N 点（三杆机构）
        R_q1q2 = [c2, s2*s1, s2*c1;
                  0,  c1,   -s1;
                 -s2, c2*s1, c2*c1];
        
        theta_init = -5.41;
        Rx_init = [1, 0, 0;
                   0, cosd(theta_init), -sind(theta_init);
                   0, sind(theta_init),  cosd(theta_init)];
        
        R = R_q1q2 * Rx_init;
        
        theta3 = q3 + 5.41;
        theta4 = q4 - 8.15;
        
        Rx3 = [1, 0, 0;
               0, cosd(theta3), -sind(theta3);
               0, sind(theta3),  cosd(theta3)];
        
        Rx4 = [1, 0, 0;
               0, cosd(theta4), -sind(theta4);
               0, sind(theta4),  cosd(theta4)];
        
        PM = p.L_PM;
        MN = p.L_MN;
        
        M_local = [0; 0; PM];
        M = (P' + R * M_local)';
        
        N_local = [0; 0; PM] + Rx3 * [0; 0; MN];
        N = (P' + R * N_local)';
        
        %% 4.6 存储结果
        all_points(i).O = O;
        all_points(i).P = P;
        all_points(i).A1 = A1;
        all_points(i).A2 = A2;
        all_points(i).B1 = B1;
        all_points(i).B2 = B2;
        all_points(i).C1 = C1;
        all_points(i).C2 = C2;
        all_points(i).M = M';
        all_points(i).N = N';
        all_points(i).T = T';
        
        all_info(i).q1 = q1;
        all_info(i).q2 = q2;
        all_info(i).q3 = q3;
        all_info(i).q4 = q4;
        all_info(i).d1 = d1_abs;
        all_info(i).d2 = d2_abs;
        all_info(i).success = true;
        
        success_count = success_count + 1;
        fprintf('  -> 成功 (q1=%.2f°, q2=%.2f°, q3=%.2f°, q4=%.2f°)\n', q1, q2, q3, q4);
    end
    
    fprintf('逆运动学坐标生成完成！成功：%d/%d 帧\n', success_count, num_frames);
end
