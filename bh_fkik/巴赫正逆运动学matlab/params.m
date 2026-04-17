function p = params()
% PARAMS 定义灵巧手运动学参数
%   返回一个结构体 p，包含所有几何参数和关节限制
%   长度单位：mm
%   角度单位：度 (degree)

    p.L1 = 50.0;          % ON 长度 (mm)
    p.L2 = 42.66;         % NT 长度 (mm)
    p.offset_deg = 18.55; % 关节 3 的偏置角度 (度)
    
    % 关节限位 (度)
    p.limit.q1 = [-180, 180];
    p.limit.q2 = [-80, 45];
    p.limit.q3 = [0, 80];
    
    % 计算辅助常数 (用于逆解判断)
    p.D_max = p.L1 + p.L2; % 最大工作半径
    % 最小工作半径取决于关节 3 的最大角度 (此时连杆折叠最厉害)
    theta3_max = deg2rad(p.limit.q3(2) + p.offset_deg);
    p.D_min = sqrt(p.L1^2 + p.L2^2 + 2*p.L1*p.L2*cos(theta3_max));
    
    % 容差
    p.tol = 1e-6;
end
