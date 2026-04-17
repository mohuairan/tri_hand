function [d1_init, d2_init, d1_new, d2_new, delta_d1, delta_d2, info] = ...
    ik_d1_d2(q1, q2, params)
%IK_D1_D2 根据 q1、q2 计算并联机构 d1、d2 的变化量
%   输入：q1 - 屈伸角 (度)
%         q2 - 外展角 (度)
%         params - 参数结构体 (来自 params())
%   输出：d1_init - 初始状态 d1 值 (mm)
%         d2_init - 初始状态 d2 值 (mm)
%         d1_new - 旋转后 d1 值 (mm)
%         d2_new - 旋转后 d2 值 (mm)
%         delta_d1 - d1 变化量 (mm)
%         delta_d2 - d2 变化量 (mm)
%         info - 详细信息结构体

    %% ========== 加载参数 ==========
    ax = params.ax;
    ay = params.ay;
    bx = params.bx;
    by = params.by;
    bz = params.bz;
    P = params.P;           % [Px, Py, Pz]
    l1 = params.l1;
    l2 = params.l2;
    
    px = P(1); py = P(2); pz = P(3);
    
    %% ========== 第一步：计算初始状态下的 d1、d2 ==========
    % 初始 B 点坐标 (未旋转时)
    X_B1_init = px + bx;
    Y_B1_init = py + by;
    Z_B1_init = pz + bz;
    
    X_B2_init = px - bx;
    Y_B2_init = py + by;
    Z_B2_init = pz + bz;
    
    % 初始水平距离平方
    H1_sq_init = (X_B1_init - ax)^2 + (Y_B1_init - ay)^2;
    H2_sq_init = (X_B2_init + ax)^2 + (Y_B2_init - ay)^2;
    
    % 检查可行性
    if l1^2 < H1_sq_init || l2^2 < H2_sq_init
        info.success = false;
        info.error = '初始状态不可达：杆长不足';
        d1_init = []; d2_init = []; d1_new = []; d2_new = [];
        delta_d1 = []; delta_d2 = [];
        return;
    end
    
    % 求解初始 d1、d2 (取 C 在 B 上方的解，即负号)
    % 根据实际装配模式选择符号，这里假设 C 点在 B 点上方
    d1_init = Z_B1_init - sqrt(l1^2 - H1_sq_init);
    d2_init = Z_B2_init - sqrt(l2^2 - H2_sq_init);
    
    %% ========== 第二步：计算旋转后的 B 点坐标 ==========
    % 角度转弧度
    q1_rad = deg2rad(q1);
    q2_rad = deg2rad(q2);
    
    c1 = cos(q1_rad); s1 = sin(q1_rad);
    c2 = cos(q2_rad); s2 = sin(q2_rad);
    
    % 旋转矩阵 R (与之前推导一致)
    % R = [c2, s2*s1, s2*c1;
    %      0,  c1,   -s1;
    %     -s2, c2*s1, c2*c1];
    
    % 计算旋转后的向量 b1' = R * [bx, by, bz]'
    b1x_prime = c2 * bx + s2 * s1 * by + s2 * c1 * bz;
    b1y_prime = c1 * by - s1 * bz;
    b1z_prime = -s2 * bx + c2 * s1 * by + c2 * c1 * bz;
    
    % 计算旋转后的向量 b2' = R * [-bx, by, bz]'
    b2x_prime = -c2 * bx + s2 * s1 * by + s2 * c1 * bz;
    b2y_prime = c1 * by - s1 * bz;       % 与 b1y_prime 相同
    b2z_prime = s2 * bx + c2 * s1 * by + c2 * c1 * bz;
    
    % 新 B 点坐标
    X_B1_new = px + b1x_prime;
    Y_B1_new = py + b1y_prime;
    Z_B1_new = pz + b1z_prime;
    
    X_B2_new = px + b2x_prime;
    Y_B2_new = py + b2y_prime;
    Z_B2_new = pz + b2z_prime;
    
    %% ========== 第三步：计算新的 d1'、d2' ==========
    % 新水平距离平方
    H1_sq_new = (X_B1_new - ax)^2 + (Y_B1_new - ay)^2;
    H2_sq_new = (X_B2_new + ax)^2 + (Y_B2_new - ay)^2;
    
    % 检查可行性
    if l1^2 < H1_sq_new || l2^2 < H2_sq_new
        info.success = false;
        info.error = '旋转后状态不可达：杆长不足';
        d1_new = []; d2_new = [];
        delta_d1 = []; delta_d2 = [];
        return;
    end
    
    % 求解新 d1'、d2' (符号必须与初始状态一致)
    d1_new = Z_B1_new - sqrt(l1^2 - H1_sq_new);
    d2_new = Z_B2_new - sqrt(l2^2 - H2_sq_new);
    
    %% ========== 第四步：计算变化量 ==========
    delta_d1 = d1_new - d1_init;
    delta_d2 = d2_new - d2_init;
    
    %% ========== 信息输出 ==========
    info.success = true;
    info.q1 = q1;
    info.q2 = q2;
    info.H1_sq_init = H1_sq_init;
    info.H2_sq_init = H2_sq_init;
    info.H1_sq_new = H1_sq_new;
    info.H2_sq_new = H2_sq_new;
end
