function [T_end, P_end, R_end, info] = thumb_fk_main(q1, q2, q3, q4)
    % THUMB_FK_MAIN 计算灵巧手拇指机构的正运动学
    %
    % 功能：根据给定的四个关节角度，计算末端执行器的位置和姿态
    %
    % 输入:
    %   q1, q2, q3, q4 : 关节角度 (单位：度)
    %       q1: 基座关节 1 角度（绕 X 轴旋转）
    %       q2: 基座关节 2 角度（绕 Z 轴旋转）
    %       q3: 中间关节角度
    %       q4: 末端关节角度
    %
    % 输出:
    %   T_end : 4x4 齐次变换矩阵，包含位置和姿态信息
    %   P_end : 3x1 位置向量 [x; y; z] (mm)
    %   R_end : 3x3 旋转矩阵，描述末端姿态
    %   info  : 结构体，包含中间点和平面信息
    %           info.P_M      : M 点位置 (3x1 向量)
    %           info.P_N      : N 点位置 (3x1 向量)
    %           info.n_c      : OMNT 平面法向量 (单位向量)
    %           info.phi      : 平面与基座 X-Z 平面夹角 (度)
    %
    % 示例:
    %   [T, P, R, info] = thumb_fk_main(-1.85, -12.13, -82.04, -63.65);

    %% 1. 参数定义 (结构常数)
    % 从参数文件获取结构常数，确保与逆解保持一致
    p = params();
    
    alpha1 = p.alpha1;       % 基座固定偏角 (度)
    alpha2 = p.alpha2;       % M 系固定偏角 (度)
    alpha3 = p.alpha3;       % N 系固定偏角 (度)
    
    L1 = p.L1;               % OM 长度 (mm)
    L2 = p.L2;               % MN 长度 (mm)
    L3 = p.L3;               % NT 长度 (mm)

    %% 2. 角度预处理
    % 角度转换为弧度
    deg2rad = pi / 180.0;
    
    th1 = q1 * deg2rad;                 % q1 弧度
    th2 = q2 * deg2rad;                 % q2 弧度
    th3 = (alpha2 + q3) * deg2rad;      % 关节 3 总角度 (含固定偏角)
    th4 = (alpha3 + q4) * deg2rad;      % 关节 4 总角度 (含固定偏角)
    th1_base = alpha1 * deg2rad;        % 基座固定偏角弧度

    %% 3. 计算基座旋转矩阵 R_01
    % R_01 描述{1}系相对于{0}系的姿态，由三个旋转组成:
    % 1. Rx1: 绕 X 轴旋转 th1 (关节 1)
    % 2. Rz2: 绕 Z 轴旋转 th2 (关节 2)
    % 3. Ry_alpha1: 绕 Y 轴旋转 alpha1 (固定偏角)
    
    Rx1 = [1, 0, 0; 
           0, cos(th1), -sin(th1); 
           0, sin(th1), cos(th1)];
           
    Rz2 = [cos(th2), -sin(th2), 0; 
           sin(th2), cos(th2), 0; 
           0, 0, 1];
           
    Ry_alpha1 = [cos(th1_base), 0, sin(th1_base); 
                 0, 1, 0; 
                 -sin(th1_base), 0, cos(th1_base)];
    
    % 复合旋转矩阵
    R_01 = Rx1 * Rz2 * Ry_alpha1;

    %% 4. 计算局部平面向量 ({1}系 X-Z 平面内)
    % 在{1}系中，所有点都在 X-Z 平面内 (Y=0)
    % v1: 从 O 到 M 的向量
    % v2_local: 从 M 到 N 的向量
    % v3_local: 从 N 到 T 的向量
    
    v1 = [0; 0; L1];                          % OM 向量
    v2_local = [L2 * sin(th3); 0; L2 * cos(th3)];  % MN 向量
    v3_local = [L3 * sin(th3 + th4); 0; L3 * cos(th3 + th4)];  % NT 向量
    
    % 局部坐标系中的末端位置
    P_local_1 = v1 + v2_local + v3_local;

    %% 5. 计算全局位置
    % 将局部位置通过旋转矩阵变换到基座坐标系
    P_end = R_01 * P_local_1;

    %% 6. 计算中间点位置
    % M 点：OM 向量变换到基座系
    info.P_M = R_01 * v1;
    % N 点：(OM + MN) 向量变换到基座系
    info.P_N = R_01 * (v1 + v2_local);

    %% 7. 计算末端姿态
    % 末端姿态由总旋转角 (th3 + th4) 决定
    Ry_total = [cos(th3 + th4), 0, sin(th3 + th4); 
                0, 1, 0; 
                -sin(th3 + th4), 0, cos(th3 + th4)];
    
    R_end = R_01 * Ry_total;

    %% 8. 构建齐次变换矩阵
    % 4x4 齐次变换矩阵，用于统一的坐标变换
    T_end = eye(4);
    T_end(1:3, 1:3) = R_end;
    T_end(1:3, 4) = P_end;

    %% 9. 计算 OMNT 平面法向量和夹角
    % 法向量 n_c = R_01 * [0; 1; 0]
    % 即{1}系的 Y 轴在{0}系中的方向
    info.n_c = R_01 * [0; 1; 0];
    info.n_c = info.n_c / norm(info.n_c);  % 单位化
    
    % 平面与基座 X-Z 平面夹角 phi
    % phi = acos(n_c · y_axis) = acos(n_c(2))
    info.phi = acos(info.n_c(2)) * 180 / pi;  % 转换为度
end
