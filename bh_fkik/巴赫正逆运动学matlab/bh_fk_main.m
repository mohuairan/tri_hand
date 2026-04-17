function T = bh_fk_main(q, p)
% BH_FK_MAIN 三自由度灵巧手正运动学解算

    % 1. 角度转弧度
    q1 = deg2rad(q(1));
    q2 = deg2rad(q(2));
    q3 = deg2rad(q(3));
    
    % 2. 计算实际关节 3 角度 (含偏置)
    theta3 = q3 + deg2rad(p.offset_deg);
    
    % 3. 三角函数预计算
    s1 = sin(q1); c1 = cos(q1);
    s2 = sin(q2); c2 = cos(q2);
    s3 = sin(theta3); c3 = cos(theta3);
    
    L1 = p.L1;
    L2 = p.L2;
    
    % 4. 根据机构学推导的公式
    % term_K = L1*sin(q2) + L2*sin(q2+theta3)
    %        = L1*s2 + L2*(s2*c3 + c2*s3)
    %        = s2*(L1 + L2*c3) + L2*s3*c2
    term_K = s2 * (L1 + L2 * c3) + L2 * s3 * c2;
    
    x = s1 * term_K;
    y = -c1 * term_K;      % 负号来自 Rx 旋转矩阵
    z = L1 * c2 + L2 * (c2 * c3 - s2 * s3);  % cos(q2+theta3) = c2*c3 - s2*s3
    
    % 5. 构建齐次变换矩阵
    R_z1 = [c1, -s1, 0; s1, c1, 0; 0, 0, 1];
    R_x2 = [1, 0, 0; 0, c2, -s2; 0, s2, c2];
    R_x3 = [1, 0, 0; 0, c3, -s3; 0, s3, c3];
    
    R = R_z1 * R_x2 * R_x3;
    
    T = eye(4);
    T(1:3, 1:3) = R;
    T(1:3, 4) = [x; y; z];
end
