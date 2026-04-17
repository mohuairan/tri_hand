% BH_IK_SOLVE 逆运动学独立测试脚本
clear; clc; close all;

% ================= 用户输入区域 =================
% 使用您的实测数据
P_input = [-21.3643,16.3106,78.1069];  
% ===============================================

% 1. 加载参数
p = params();

fprintf('=== 逆运动学独立测试（实测校准版）===\n');
fprintf('输入目标位置 (mm): [%.4f, %.4f, %.4f]\n', P_input(1), P_input(2), P_input(3));
fprintf('目标点距离原点：%.4f mm (工作空间：%.2f ~ %.2f mm)\n\n', ...
    norm(P_input), p.D_min, p.D_max);

% 2. 调用逆解函数
[q_all, status, info] = bh_ik_main(P_input, p);

% 3. 输出所有解
fprintf('--- 所有可能的逆解 ---\n');
for i = 1:info.num_solutions
    q = info.all_solutions{i};
    fprintf('解 %d: q = [%7.2f°, %7.2f°, %7.2f°]', i, q(1), q(2), q(3));
    
    in_limit = (q(1) >= p.limit.q1(1) && q(1) <= p.limit.q1(2)) && ...
               (q(2) >= p.limit.q2(1) && q(2) <= p.limit.q2(2)) && ...
               (q(3) >= p.limit.q3(1) && q(3) <= p.limit.q3(2));
    
    if in_limit
        fprintf(' ✓ (在限位内)\n');
    else
        fprintf(' ✗ (超限)\n');
    end
    
    % 验证位置
    T_test = bh_fk_main(q, p);
    P_test = T_test(1:3, 4)';
    err_test = norm(P_test - P_input);
    fprintf('       位置：[%.4f, %.4f, %.4f], 误差：%.6f mm\n', ...
        P_test(1), P_test(2), P_test(3), err_test);
end

fprintf('\n');

% 4. 输出有效解
if status == 1
    fprintf('=== 有效解（在限位内）===\n');
    if size(q_all, 1) == 1 && size(q_all, 2) == 3
        num_valid = 1;
    else
        num_valid = size(q_all, 1);
    end
    
    for i = 1:num_valid
        if num_valid == 1
            q = q_all;
        else
            q = q_all(i, :);
        end
        
        fprintf('有效解 %d: q = [%.4f°, %.4f°, %.4f°]\n', i, q(1), q(2), q(3));
        
        T_verify = bh_fk_main(q, p);
        P_verify = T_verify(1:3, 4)';
        err_pos = norm(P_verify - P_input);
        fprintf('           位置复现误差：%.6f mm\n\n', err_pos);
    end
else
    fprintf('✗ 无有效解\n');
end
