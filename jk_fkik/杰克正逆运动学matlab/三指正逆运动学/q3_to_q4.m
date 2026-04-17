function q4 = q3_to_q4(q3, params)
%Q3_TO_Q4 四连杆机构运动学求解
%   输入：q3 - MN 杆转动角度（度），逆时针为正，顺时针为负
%         params - 参数结构体（可选，包含四连杆机构参数）
%   输出：q4 - NT 相对 MN 延长线夹角的变化量（度）
%         初始位置 q3=0 时 q4=0；两者均向负值变化（顺时针）
%
%   调用方式：
%     q4 = q3_to_q4(q3)           % 使用默认参数
%     q4 = q3_to_q4(q3, params)   % 使用 params 中的参数

    % ========== 参数处理 ==========
    if nargin < 2 || isempty(params) 
        %nargin：matlab内置变量 "number of input arguments"（输入参数个数）。
        %nargout：matlab内置变量 "number of output arguments"（输出参数个数）
        % 使用默认参数（向后兼容）
        L_MN = 30.0;
        L_NQ = 4.0;
        L_KQ = 27.5;
        MK_LEN = 5.0;
        MK_ANGLE = -80.0;
        PHI = 141.89;
        ANGLE_INIT = -8.15;
    else
        % 从 params 结构体读取参数
        L_MN = params.L_MN;
        L_NQ = params.L_NQ;
        L_KQ = params.L_KQ;
        MK_LEN = params.MK_LEN;
        MK_ANGLE = params.MK_ANGLE;
        PHI = params.PHI;
        ANGLE_INIT = params.theta4_init;
    end
    
    % ========== 固定点 K 坐标 ==========
    KX = MK_LEN * cosd(MK_ANGLE);
    KY = MK_LEN * sind(MK_ANGLE);
    %cosd和sind：分别用于计算角度的余弦和正弦值，输入参数是以度为单位的角度。
    % ========== 动点 N 坐标 ==========
    NX = L_MN * cosd(q3);
    NY = L_MN * sind(q3);
    
    % ========== 虚拟连杆 NK ==========
    DX = KX - NX;
    DY = KY - NY;
    L_NK = sqrt(DX^2 + DY^2);
    
    % 限位检查：如果 NK 超出可行范围，返回 NaN
    if (L_NK < abs(L_KQ - L_NQ)) || (L_NK > L_KQ + L_NQ)
        %||相比|存在短路特性，如果前面条件为真，后面条件就不再计算了
        q4 = NaN;       %标记此位置无解
        return;         % 提前退出函数
    end
    
    % ========== 三角形内角 γ = ∠QNK ==========
    THETA_NK = atan2d(DY, DX);
    
    COS_GAMMA = (L_NQ^2 + L_NK^2 - L_KQ^2) / (2 * L_NQ * L_NK);
    COS_GAMMA = max(-1.0, min(1.0, COS_GAMMA)); 
     % 数值钳位： 限制在 [-1, 1] 防止数值误差。
     % 即使由于浮点计算（二进制无法精确表示0.1）误差
     % 导致 COS_GAMMA 超出这个范围（如1.000001，也不会引起 acosd 的错误。
    GAMMA = acosd(COS_GAMMA);
    
    % ========== NQ 绝对角度（Q 在 MN 上方）==========
    THETA_NQ = THETA_NK - GAMMA;
    
    % ========== NT 绝对角度 ==========
    % Q 在 NQ 上，T 在 NT 上，∠QNT=PHI，T 在 NQ 顺时针方向
    THETA_NT = THETA_NQ - PHI;
    
    % ========== NT 相对 MN 延长线的绝对夹角 ==========
    angle_abs = THETA_NT - q3;
    
    % ========== 规范化到 [-180°, 180°] ==========
    while angle_abs > 180
        angle_abs = angle_abs - 360;
    end
    while angle_abs < -180
        angle_abs = angle_abs + 360;
    end
    
    % ========== 输出变化量（减去初始值）==========
    q4 = angle_abs - ANGLE_INIT;
end
