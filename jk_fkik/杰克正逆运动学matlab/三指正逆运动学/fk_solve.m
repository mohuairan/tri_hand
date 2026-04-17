%% =========================================================================
%   ILDA 手 MCP 关节正运动学求解 - 应用实例
%   依赖：ik_params.m, q3_to_q4.m, fk_d1d2_to_q1q2.m, fk_chain_to_T.m
% =========================================================================
%
%   工作流程：
%   1. ik_params - 获取机构参数
%   2. fk_d1d2_to_q1q2 - 根据 d1、d2 变化量求 q1、q2
%   3. q3_to_q4 - 根据 q3 求 q4（耦合关系）
%   4. fk_chain_to_T - 计算末端 T 的位姿
%
%   输入：d1、d2 的变化量 (delta_d1, delta_d2) 和 q3
%   输出：T 点的位置和姿态
%
% =========================================================================

clear; clc;

%% 1. 参数设置
p = params();

% 输入：d1、d2 变化量和 q3
delta_d1 = 19.75-20.854;         % d1 变化量 (mm)，修改为实际值
delta_d2 = 18.26-20.854;         % d2 变化量 (mm)，修改为实际值
q3 = -30.35;               % q3 关节角 (度)，修改为实际值

% delta_d1 = 0.0;
% delta_d2 = 0.0;
% q3 = -30.0;

fprintf('=== ILDA 手 MCP 关节正运动学求解 ===\n\n');
fprintf('输入参数：\n');
fprintf('  delta_d1 = %.3f mm\n', delta_d1);
%fprintf 函数的格式化输出控制符：
% %f：表示以浮点数格式输出。3 表示小数点后保留 3 位。
fprintf('  delta_d2 = %.3f mm\n', delta_d2);
fprintf('  q3 = %.4f°\n\n', q3);

%% 2. 调用 fk_chain_to_T 求解末端位姿
fprintf('正在计算末端位姿...\n');
[T_pos, T_rot, info] = fk_chain_to_T(delta_d1, delta_d2, q3, p);

if ~info.success
    error('正运动学求解失败：%s', info.error);
    %%s：字符串 (string) 格式化占位符，用于在输出中插入文本字符串
    %error有一个输入参数，即错误消息字符串。当调用 error 函数时，它会显示该错误消息并终止程序的执行。
end

fprintf('求解成功！\n\n');

%% 3. 显示中间变量 q1, q2, q4
fprintf('=== 中间变量 ===\n');
fprintf('q1 = %.4f°\n', info.q1);
fprintf('q2 = %.4f°\n', info.q2);
fprintf('q3 = %.4f°\n', info.q3);
fprintf('q4 = %.4f°\n\n', info.q4);

%% 4. 显示末端 T 点位姿
fprintf('=== 末端 T 点位姿 ===\n');
fprintf('位置：T = [%.3f, %.3f, %.3f] mm\n\n', T_pos(1), T_pos(2), T_pos(3));

fprintf('姿态矩阵 R =\n');
fprintf('  [%.4f  %.4f  %.4f\n', T_rot(1,1), T_rot(1,2), T_rot(1,3));
fprintf('   %.4f  %.4f  %.4f\n', T_rot(2,1), T_rot(2,2), T_rot(2,3));
fprintf('   %.4f  %.4f  %.4f]\n\n', T_rot(3,1), T_rot(3,2), T_rot(3,3));

%% 5. 欧拉角表示（可选）
% 将姿态矩阵转换为欧拉角（ZYX 顺序）
sy = sqrt(T_rot(1,1)^2 + T_rot(2,1)^2);
if sy > 1e-10
    %判断万向节锁死情况，避免数值不稳定
    roll  = atan2(T_rot(3,2), T_rot(3,3));
    pitch = atan2(-T_rot(3,1), sy);
    yaw   = atan2(T_rot(2,1), T_rot(1,1));
else
    roll  = 0;
    pitch = atan2(-T_rot(3,1), sy);
    yaw   = atan2(-T_rot(1,2), T_rot(2,2));
end

fprintf('欧拉角 (ZYX 顺序，度)：\n');
fprintf('  Roll  (绕 Z) = %.4f°\n', rad2deg(yaw));
fprintf('  Pitch (绕 Y) = %.4f°\n', rad2deg(pitch));
fprintf('  Yaw   (绕 X) = %.4f°\n', rad2deg(roll));

fprintf('\n=== 求解完成 ===\n');
