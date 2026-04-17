%% =========================================================================
%   ILDA 手 MCP 关节逆运动学求解 - 应用实例
%   依赖：params.m, q3_to_q4.m, ik_q3_from_PT.m, 
%         ik_q1_q2_numerical.m, ik_d1_d2.m
% =========================================================================
%
%   工作流程：
%   1. params - 获取机构参数
%   2. q3_to_q4 - 求 q3 和 q4 的耦合关系
%   3. ik_q3_from_PT - 求 q3、q4 具体值
%   4. ik_q1_q2_numerical - 求 q1、q2 的值
%   5. ik_d1_d2 - 求 d1 和 d2 的变化量
%
% =========================================================================

clear; clc;

%% 1. 参数与目标点设置
p = params();
T = [-3.69, 93.91, 46.59];          % 修改为目标点坐标
% q_init = [0, 0];                % 迭代初值 [q1, q2]

fprintf('=== ILDA 手 MCP 关节逆运动学求解 ===\n\n');
fprintf('目标点 T = [%.2f, %.2f, %.2f] mm\n\n', T(1), T(2), T(3));

%% 2. 调用 ik_solve 求解末端位姿
fprintf('正在求解 q1, q2...\n');
% [q1_sol, q2_sol, info_q] = ik_q1_q2_numerical(T, q_init, p);
[q1_sol, q2_sol, info_q] = ik_q1q2_new(T, p);

if ~info_q.success
    error('q1/q2 求解失败：%s', info_q.error);
end

fprintf('求解成功！找到 %d 组解\n\n', info_q.num_solutions);

%% 3. 显示中间变量 q3, q4
fprintf('=== 中间变量 ===\n');
fprintf('q3 = %.4f°, q4 = %.4f°\n\n', info_q.q3, info_q.q4);

%% 4. 计算每组解对应的 d1, d2 变化量
fprintf('=== 计算结果 ===\n');
fprintf('%-4s %-8s %-8s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
    '编号', 'q1(°)', 'q2(°)', 'd1 初始', 'd1 新', 'Δd1', 'd2 初始', 'd2 新', 'Δd2');
fprintf('%s\n', repmat('-', 1, 85));
%repmat函数的语法：B = repmat(A, m, n) 将数组 A 重复 m 行 n 列，生成一个新的数组 B。

for i = 1:length(q1_sol)
    q1 = q1_sol(i); 
    q2 = q2_sol(i);
    
    [d1_i, d2_i, d1_n, d2_n, dd1, dd2, info] = ik_d1_d2(q1, q2, p);
    
    if info.success
        fprintf('%-4d %-8.3f %-8.3f %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f %-10.3f\n', ...
            i, q1, q2, d1_i, d1_n, dd1, d2_i, d2_n, dd2);
    else
        fprintf('%-4d %-8.3f %-8.3f %-10s\n', i, q1, q2, '不可达');
    end
end

fprintf('\n=== 求解完成 ===\n');
