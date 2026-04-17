% ================================================================================
%                    拇指机构正运动学求解与验证脚本
% ================================================================================
% 功能：用于调试和验证正运动学算法
% 说明：本文件仅包含输入、打印和验证部分，不含运算核心代码
%       核心算法在 thumb_fk_main.m 中实现
% ================================================================================

clear; clc; close all;

%% 1. 设置关节角度 (输入部分)
% 在此处修改关节角度进行测试
q1 = -24.6;        % 度 - 基座关节 1
q2 = -0.32;     % 度 - 基座关节 2
q3 = -17.43;     % 度 - 中间关节
q4 = -20.17;     % 度 - 末端关节

%% 2. 调用正解函数
% 核心运算在 thumb_fk_main.m 中实现
[T, P, R, info] = thumb_fk_main(q1, q2, q3, q4);

%% 3. 输出结果 (打印部分)
fprintf('==================== 正解计算结果 ====================\n');
fprintf('关节角度：q1=%.2f°, q2=%.2f°, q3=%.2f°, q4=%.2f°\n', q1, q2, q3, q4);
fprintf('\n末端位置 (mm): x=%.4f, y=%.4f, z=%.4f\n', P(1), P(2), P(3));
fprintf('末端姿态矩阵:\n');
disp(R);

fprintf('==================== 约束平面信息 ====================\n');
fprintf('OMNT 平面法向量：[%.6f, %.6f, %.6f]\n', info.n_c(1), info.n_c(2), info.n_c(3));
fprintf('平面与基座 X-Z 平面夹角：phi = %.4f°\n', info.phi);

fprintf('==================== 中间点位置 ====================\n');
fprintf('M 点位置：[%.4f, %.4f, %.4f] (mm)\n', info.P_M(1), info.P_M(2), info.P_M(3));
fprintf('N 点位置：[%.4f, %.4f, %.4f] (mm)\n', info.P_N(1), info.P_N(2), info.P_N(3));

%% 4. 共面性验证
% 验证 O、M、N、T 四点是否共面
fprintf('\n==================== 共面性验证 ====================\n');

O = [0; 0; 0];          % 原点
M = info.P_M;           % M 点位置
N = info.P_N;           % N 点位置
T_point = P;            % T 点位置 (末端)

% 向量计算
OM = M - O;
ON = N - O;
OT = T_point - O;

% 混合积 (OM × ON) · OT 应接近 0 (表示四点共面)
coplanar_check = dot(cross(OM, ON), OT);
fprintf('共面性检验 (混合积): %.6f (应接近 0)\n', coplanar_check);

% 获取容差参数
p = params();
if abs(coplanar_check) < p.coplanar_tolerance
    fprintf('✅ OMNT 四点共面验证通过\n');
else
    fprintf('⚠ 共面性检验警告：误差超出容差范围 (%.2e)\n', p.coplanar_tolerance);
end

%% 5. 可视化 (可选)
% 取消注释以启用 3D 可视化
%{
figure('Name', '拇指机构正解可视化');
hold on; grid on; axis equal;

% 绘制各点
plot3(0, 0, 0, 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
plot3(info.P_M(1), info.P_M(2), info.P_M(3), 'bo', 'MarkerSize', 8, 'MarkerFaceColor', 'b');
plot3(info.P_N(1), info.P_N(2), info.P_N(3), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
plot3(P(1), P(2), P(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');

% 绘制连杆
plot3([0, info.P_M(1)], [0, info.P_M(2)], [0, info.P_M(3)], 'b-', 'LineWidth', 2);
plot3([info.P_M(1), info.P_N(1)], [info.P_M(2), info.P_N(2)], [info.P_M(3), info.P_N(3)], 'g-', 'LineWidth', 2);
plot3([info.P_N(1), P(1)], [info.P_N(2), P(2)], [info.P_N(3), P(3)], 'r-', 'LineWidth', 2);

% 标注
text(0, 0, 0, ' O', 'VerticalAlignment', 'bottom');
text(info.P_M(1), info.P_M(2), info.P_M(3), ' M', 'VerticalAlignment', 'bottom');
text(info.P_N(1), info.P_N(2), info.P_N(3), ' N', 'VerticalAlignment', 'bottom');
text(P(1), P(2), P(3), ' T', 'VerticalAlignment', 'bottom');

xlabel('X (mm)'); ylabel('Y (mm)'); zlabel('Z (mm)');
title('拇指机构正解可视化 (O-M-N-T)');
view(3);
%}
