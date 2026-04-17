function [q1, q2, info] = thumb_ik_q1q2(P_target, n_c)
    % THUMB_IK_Q1Q2 求解拇指机构的 q1 和 q2（约束平面姿态）
    %
    % 功能：根据目标位置和约束平面法向量，求解基座两个关节的角度
    % 方法：直接从法向量解析求解 q1, q2
    %
    % 输入:
    %   P_target : 3x1 向量，末端目标位置 [x; y; z] (mm)
    %   n_c      : 3x1 向量，约束平面法向量（单位向量）
    %              n_c = [-sin(q2); cos(q1)*cos(q2); sin(q1)*cos(q2)]
    %
    % 输出:
    %   q1       : 关节 1 角度 (度)
    %   q2       : 关节 2 角度 (度)
    %   info     : 结构体，包含求解状态和中间信息
    %              info.status     : 求解状态 (0=成功，1=无解，2=多解/警告)
    %              info.error_msg  : 错误/状态信息
    %              info.n_c        : 输入的法向量（可能已修正）
    %              info.phi        : 约束平面夹角（中间输出，度）
    %              info.q1_all     : 所有可行的 q1 解
    %              info.q2_all     : 所有可行的 q2 解
    %              info.all_candidates : 所有候选解 (包括被过滤的)
    %              info.correction : 法向量修正信息
    %              info.limit_exceeded : 关节限位超限标志
    %              info.exceed_joint   : 超限关节名称列表
    %
    % 示例:
    %   [q1, q2, info] = thumb_ik_q1q2([60; 15; 10], [0.2; 0.98; 0]);
    %
    % 数学原理:
    %   法向量定义：n_c = R_01 * [0; 1; 0]
    %   展开得：n_c = [-sin(q2); cos(q1)*cos(q2); sin(q1)*cos(q2)]
    %   
    %   因此：
    %   - sin(q2) = -n_c(1)
    %   - q2 = asin(-n_c(1))
    %   - cos(q1) = n_c(2) / cos(q2)
    %   - sin(q1) = n_c(3) / cos(q2)
    %   - q1 = atan2(n_c(3), n_c(2))
    %
    %   平面夹角 phi（中间输出）:
    %   - phi = acos(|n_c(2)|) * 180/pi （当 q2≈0 时）
    %   - 或 phi = acos(cos(q1)*cos(q2)) * 180/pi

    %% 1. 参数定义
    % 从参数文件获取结构常数和容差参数
    p = params();
    
    alpha1 = p.alpha1;              % 基座固定偏角 (度)
    joint_limits = [p.q1_limit_min, p.q1_limit_max; 
                    p.q2_limit_min, p.q2_limit_max];  % q1, q2 限位
    limit_tolerance = p.limit_tolerance;  % 限位检查容差 (度)
    
    % 容差参数
    tol_normal = p.tol_normal;      % 法向量分量容差
    deg2rad = pi / 180.0;           % 角度转弧度因子

    %% 2. 初始化输出
    q1 = []; q2 = [];
    info = struct();
    info.status = 0;                    % 0=求解成功
    info.error_msg = '';                % 错误信息
    info.n_c = [];                      % 输入的法向量
    info.phi = [];                      % 约束平面夹角（中间输出）
    info.q1_all = [];                   % 所有可行的 q1 解
    info.q2_all = [];                   % 所有可行的 q2 解
    info.limit_exceeded = false;        % 关节限位超限标志
    info.exceed_joint = {};             % 超限关节名称列表
    info.all_candidates = [];           % 所有候选解
    info.correction = struct('applied', false, ...  % 法向量修正信息
                             'nx', 0, 'ny', 0, 'nz', 0, ...
                             'method', 'none');

    %% 3. 输入验证
    if nargin < 2
        info.status = 1;
        info.error_msg = '输入参数不足：需要 P_target 和 n_c';
        return;
    end

    if length(P_target) ~= 3
        info.status = 1;
        info.error_msg = 'P_target 必须是 3x1 向量 [x; y; z]';
        return;
    end
    
    if length(n_c) ~= 3
        info.status = 1;
        info.error_msg = 'n_c 必须是 3x1 向量 [nx; ny; nz]';
        return;
    end

    %% 4. 法向量预处理和验证
    % 保存原始输入
    nx_orig = n_c(1);
    ny_orig = n_c(2);
    nz_orig = n_c(3);
    
    % 检查法向量是否为单位向量
    norm_nc = norm(n_c);
    if abs(norm_nc - 1.0) > tol_normal * 10
        fprintf('\n[警告] 法向量模长=%.6f，进行归一化处理\n', norm_nc);
        n_c = n_c / norm_nc;
        info.correction.applied = true;
        info.correction.method = 'normalize';
    end
    
    % 检查法向量 X 分量（根据坐标系定义，nx 应为负或接近 0）
    % n_c = [-sin(q2); cos(q1)*cos(q2); sin(q1)*cos(q2)]
    % q2 在 [-55°, 0°] 范围内，sin(q2) 为负，所以 -sin(q2) 为正
    % 因此 nx 应该为正或接近 0
    if n_c(1) < -tol_normal
        fprintf('\n[警告] 法向量 X 分量=%.4f < 0，可能超出工作空间\n', n_c(1));
    end
    
    % 保存归一化后的法向量
    info.n_c = n_c;
    nx = n_c(1);
    ny = n_c(2);
    nz = n_c(3);

    %% 5. 从法向量直接求解 q1, q2
    % 根据关系式：
    % n_c = [-sin(q2); cos(q1)*cos(q2); sin(q1)*cos(q2)]
    
    % 步骤 1: 求解 q2
    % sin(q2) = -n_c(1)
    sin_q2 = -nx;
    
    % 检查 sin(q2) 是否在有效范围内
    if abs(sin_q2) > 1.0 + tol_normal
        info.status = 1;
        info.error_msg = sprintf('法向量 X 分量超出范围 (nx=%.4f)', nx);
        return;
    end
    
    % 截断到 [-1, 1] 避免数值误差
    sin_q2 = max(-1, min(1, sin_q2));
    
    % q2 = asin(-n_c(1))
    q2_rad = asin(sin_q2);
    q2 = q2_rad / deg2rad;
    
    % 步骤 2: 求解 q1
    % cos(q1) = n_c(2) / cos(q2)
    % sin(q1) = n_c(3) / cos(q2)
    % q1 = atan2(n_c(3), n_c(2))
    cos_q2 = cos(q2_rad);
    
    if abs(cos_q2) < tol_normal
        % q2 接近 ±90°，cos(q2) 接近 0，需要特殊处理
        info.status = 1;
        info.error_msg = sprintf('q2 接近±90°，cos(q2)=%.6f 过小', cos_q2);
        return;
    end
    
    % 使用 atan2 直接求解 q1（避免除以 cos(q2) 的数值不稳定）
    q1_rad = atan2(nz, ny);
    q1 = q1_rad / deg2rad;
    
    % 步骤 3: 计算平面夹角 phi（中间输出）
    % phi = acos(|cos(q1)*cos(q2)|) * 180/pi
    % 或 phi = acos(|n_c(2)| / cos(q2)) * 180/pi （当 cos(q2) > 0）
    cos_phi = abs(cos(q1_rad) * cos(q2_rad));
    cos_phi = max(-1, min(1, cos_phi));  % 截断避免数值误差
    info.phi = acos(cos_phi) * 180 / pi;

    %% 6. 关节限位检查
    q1_solutions = [];
    q2_solutions = [];
    all_candidates = [];
    
    % 当前解的限位检查
    in_limit = true;
    skip_reason = '';
    
    if q2 < joint_limits(2,1) - limit_tolerance || ...
       q2 > joint_limits(2,2) + limit_tolerance
        skip_reason = sprintf('q2 超限 (%.2f° 不在 [%.2f°, %.2f°])', ...
            q2, joint_limits(2,1), joint_limits(2,2));
        in_limit = false;
    elseif q1 < joint_limits(1,1) - limit_tolerance || ...
           q1 > joint_limits(1,2) + limit_tolerance
        skip_reason = sprintf('q1 超限 (%.2f° 不在 [%.2f°, %.2f°])', ...
            q1, joint_limits(1,1), joint_limits(1,2));
        in_limit = false;
    end
    
    % 记录候选解
    all_candidates = [q1, q2, in_limit];
    
    if in_limit
        q1_solutions = q1;
        q2_solutions = q2;
    end
    
    info.all_candidates = all_candidates;

    %% 7. 检查是否有可行解
    if isempty(q1_solutions)
        info.status = 1;
        info.error_msg = ['无解：' skip_reason];
        return;
    end

    %% 8. 输出结果
    info.q1_all = q1_solutions;
    info.q2_all = q2_solutions;
    
    % 单解情况直接输出
    q1 = info.q1_all(1);
    q2 = info.q2_all(1);

    % 检查最终解是否超限 (不含容差)
    info.exceed_joint = {};
    if q1 < joint_limits(1,1) - limit_tolerance || q1 > joint_limits(1,2) + limit_tolerance
        info.exceed_joint{end+1} = 'q1';
    end
    if q2 < joint_limits(2,1) - limit_tolerance || q2 > joint_limits(2,2) + limit_tolerance
        info.exceed_joint{end+1} = 'q2';
    end
    if ~isempty(info.exceed_joint)
        info.limit_exceeded = true;
        info.status = 2;
        info.error_msg = sprintf('⚠ 警告：关节 %s 超出限位', strjoin(info.exceed_joint, ', '));
    else
        info.error_msg = '求解成功';
    end
    
    % 记录法向量修正信息
    if info.correction.applied && strcmp(info.correction.method, 'normalize')
        info.correction.nx = nx - nx_orig;
        info.correction.ny = ny - ny_orig;
        info.correction.nz = nz - nz_orig;
    end
end
