% BH_FK_SOLVE 正运动学独立测试脚本
clear; clc; close all;

% ================= 用户输入区域 =================
q_input = [-127.3600, -43.71, 35.52];  % 您的实测角度
% ===============================================

p = params();

fprintf('=== 正运动学测试===\n');
fprintf('输入关节角 (度): [%.2f, %.2f, %.2f]\n', q_input(1), q_input(2), q_input(3));

T = bh_fk_main(q_input, p);
pos = T(1:3, 4);

fprintf('计算结果:\n');
fprintf('   X = %.4f mm\n', pos(1));
fprintf('   Y = %.4f mm  (实测：31.94 mm)\n', pos(2));
fprintf('   Z = %.4f mm  (实测：59.61 mm)\n', pos(3));

% 验证公式正确性
q2 = deg2rad(q_input(2));
theta3 = deg2rad(q_input(3) + p.offset_deg);
term_K_manual = sin(q2)*(p.L1 + p.L2*cos(theta3)) + p.L2*sin(theta3)*cos(q2);
y_manual = -cos(0) * term_K_manual;
fprintf('\n公式验证：term_K = %.4f, Y = %.4f\n', term_K_manual, y_manual);
