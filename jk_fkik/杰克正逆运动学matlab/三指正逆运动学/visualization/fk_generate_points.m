%% =========================================================================
%   ILDA 手 MCP 关节正运动学 - 坐标生成
%   根据电机参数 (d1, d2, q3) 的变化，计算所有点的坐标
%   依赖：ik_params.m, q3_to_q4.m, fk_d1d2_to_q1q2.m, fk_chain_to_T.m
% =========================================================================
%
%   输入：d1 变化量、d2 变化量、q3 角度序列
%   输出：所有点坐标结构体数组 (O, P, A1, A2, B1, B2, C1, C2, M, N, T)
%
% =========================================================================

function [all_points, all_info] = fk_generate_points(delta_d1_seq, delta_d2_seq, q3_seq)
% FK_GENERATE_POINTS 正运动学坐标生成
%   输入：delta_d1_seq - N×1 向量，d1 变化量序列 (mm)
%         delta_d2_seq - N×1 向量，d2 变化量序列 (mm)
%         q3_seq       - N×1 向量，q3 角度序列 (度)
%   输出：all_points - 1×N 结构体数组，包含所有点坐标
%         all_info   - 1×N 结构体数组，包含求解信息

    %% 1. 加载参数
    p = params();
    
    %% 2. 初始化
    num_frames = length(delta_d1_seq);
    all_points = struct();
    all_info = struct();
    %struct：MATLAB内置函数，用于创建结构体数组。这里创建了两个结构体数组 all_points 和 all_info
    % 分别用于存储每一帧的点坐标和求解信息。
    
    % 检查输入长度一致性
    if length(delta_d2_seq) ~= num_frames || length(q3_seq) ~= num_frames
        error('输入序列长度必须一致');
    end
    
    %% 3. 固定点坐标（不随运动变化）
    O = [0, 0, 0];                              % 原点
    P = p.P;                                    % MCP 中心
    A1 = [p.ax, p.ay, 0];                       % 电机 1 基座
    A2 = [-p.ax, p.ay, 0];                      % 电机 2 基座
    
    %% 4. 获取 d1、d2 初始值
    [~, ~,  ~, ~, info_init] = ik_d1_d2(0, 0, p);
    
    if ~info_init.success
        error('无法计算初始状态 d1、d2');
    end
    
    %% 5. 循环求解每一帧
    for i = 1:num_frames
        delta_d1 = delta_d1_seq(i);
        delta_d2 = delta_d2_seq(i);
        q3 = q3_seq(i);
        
        fprintf('正解帧 %d/%d: delta_d1=%.2f, delta_d2=%.2f, q3=%.2f\n', ...
            i, num_frames, delta_d1, delta_d2, q3);
        
        %% 5.1 调用 fk_chain_to_T 计算所有
        [T_pos, T_rot, info_fk] = fk_chain_to_T(delta_d1, delta_d2, q3, p);
        
        if ~info_fk.success
            warning('帧 %d: 正解失败，跳过', i);
            all_points(i) = [];
            all_info(i) = [];
            continue;
        end
        
        %% 5.2 提取关节角
        q1 = info_fk.q1;
        q2 = info_fk.q2;
        q4 = info_fk.q4;
        
        %% 5.3 计算 d1、d2 绝对值
        d1_abs = d1_init + delta_d1;
        d2_abs = d2_init + delta_d2;
        
        %% 5.4 计算所有运动点坐标
        % 旋转矩阵
        c1 = cosd(q1); s1 = sind(q1);
        c2 = cosd(q2); s2 = sind(q2);
        
        R_q1q2 = [c2, s2*s1, s2*c1;
                  0,  c1,   -s1;
                 -s2, c2*s1, c2*c1];
        
        % 初始偏置旋转
        theta_init = -5.41;
        Rx_init = [1, 0, 0;
                   0, cosd(theta_init), -sind(theta_init);
                   0, sind(theta_init),  cosd(theta_init)];
        
        R = R_q1q2 * Rx_init;
        
        % 后续旋转
        theta3 = q3 + 5.41;
        theta4 = q4 - 8.15;
        
        Rx3 = [1, 0, 0;
               0, cosd(theta3), -sind(theta3);
               0, sind(theta3),  cosd(theta3)];
        
        Rx4 = [1, 0, 0;
               0, cosd(theta4), -sind(theta4);
               0, sind(theta4),  cosd(theta4)];
        
        % 计算 M 点 (PM 末端)
        PM = p.L_PM;
        M_local = [0; 0; PM];
        M = (P' + R * M_local)';
        
        % 计算 N 点 (MN 末端)
        MN = p.L_MN;
        N_local = [0; 0; PM] + Rx3 * [0; 0; MN];
        N = (P' + R * N_local)';
        
        %% 5.5 计算 B1、B2、C1、C2 点
        bx = p.bx; by = p.by; bz = p.bz;
        px = P(1); py = P(2); pz = P(3);
        
        % 旋转后的 b1'、b2'
        b1x_prime = c2 * bx + s2 * s1 * by + s2 * c1 * bz;
        b1y_prime = c1 * by - s1 * bz;
        b1z_prime = -s2 * bx + c2 * s1 * by + c2 * c1 * bz;
        
        b2x_prime = -c2 * bx + s2 * s1 * by + s2 * c1 * bz;
        b2y_prime = c1 * by - s1 * bz;
        b2z_prime = s2 * bx + c2 * s1 * by + c2 * c1 * bz;
        
        B1 = [px + b1x_prime, py + b1y_prime, pz + b1z_prime];
        B2 = [px + b2x_prime, py + b2y_prime, pz + b2z_prime];
        
        % C1、C2 是电机端点
        C1 = [A1(1), A1(2), d1_abs];
        C2 = [A2(1), A2(2), d2_abs];
        
        %% 5.6 存储结果
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
        all_points(i).T = T_pos';
        
        all_info(i).q1 = q1;
        all_info(i).q2 = q2;
        all_info(i).q3 = q3;
        all_info(i).q4 = q4;
        all_info(i).d1 = d1_abs;
        all_info(i).d2 = d2_abs;
        all_info(i).T = T_pos';
        all_info(i).success = true;
    end
    
    fprintf('正运动学坐标生成完成！成功：%d/%d 帧\n', ...
        sum([all_info.success]), num_frames);
        %sum([all_info.success])：统计 all_info 结构体数组中 success 字段为 true 的数量，即成功求解的帧数。
end
