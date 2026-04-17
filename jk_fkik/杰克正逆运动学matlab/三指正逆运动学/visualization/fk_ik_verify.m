%% =========================================================================
%   正逆运动学精度验证程序
%   原理：输入控制量(d1,d2,q3) → FK求解末端T → IK从T反求(d1,d2,q3) → 比较误差
%   依赖：params.m, fk_chain_to_T.m, ik_q1q2_new.m, ik_d1_d2.m
% =========================================================================
close all; clear; clc;

%% 1. 定义测试输入序列
n_half = 20;

% 阶段1: 仅d2变化
s1_d1 = zeros(1,2*n_half);
 %zeros函数的语法：B = zeros(m, n) 创建一个 m 行 n 列的全零矩阵。
s1_d2 = [linspace(0,-10,n_half), linspace(-10,0,n_half)];
%函数的语法：B = linspace(a, b, n) 生成一个行向量 B，包含 n 个从 a 到 b 的等间距点。
s1_q3 = zeros(1,2*n_half);

% 阶段2: 仅d1变化
s2_d1 = [linspace(0,-10,n_half), linspace(-10,0,n_half)];
s2_d2 = zeros(1,2*n_half);
s2_q3 = zeros(1,2*n_half);

% 阶段3: 仅q3变化
s3_d1 = zeros(1,2*n_half);
s3_d2 = zeros(1,2*n_half);
s3_q3 = [linspace(0,-60,n_half), linspace(-60,0,n_half)];

% 阶段4: d1+d2同步
s4_d1 = [linspace(0,-10,n_half), linspace(-10,0,n_half)];
s4_d2 = [linspace(0,-10,n_half), linspace(-10,0,n_half)];
s4_q3 = zeros(1,2*n_half);

% 阶段5: 协同变化(60帧)
n5 = 3*n_half;
t5 = linspace(0,2*pi,n5);
s5_d1 = -4*(1-cos(t5));
s5_d2 = -5*(1-cos(1.5*t5));
s5_d2 = max(s5_d2,-10); s5_d2 = min(s5_d2,0);
s5_q3 = -25*(1-cos(t5));

% 拼接
dd1_seq = [s1_d1, s2_d1, s3_d1, s4_d1, s5_d1];
dd2_seq = [s1_d2, s2_d2, s3_d2, s4_d2, s5_d2];
q3_seq  = [s1_q3, s2_q3, s3_q3, s4_q3, s5_q3];
%长序列表示整个测试过程中 d1 的完整变化轨迹
N = length(dd1_seq);

stage_bounds = [1,2*n_half; 2*n_half+1,4*n_half; 4*n_half+1,6*n_half;
                6*n_half+1,8*n_half; 8*n_half+1,8*n_half+n5];
stage_names = {'仅d2','仅d1','仅q3','d1+d2同步','协同'};
%`stage_bounds` 是一个 5×2 矩阵，每行定义一个阶段的起始帧和结束帧
%`stage_names` 是一个包含阶段名称的单元格数组，用于后续输出报告时标识不同阶段的误差统计结果。

fprintf('=== 正逆运动学精度验证 ===\n');
fprintf('总测试点: %d\n\n', N);

%% 2. 加载参数
p = params();

%% 3. 逐帧验证
% 预分配结果
err_d1  = nan(N,1);
err_d2  = nan(N,1);
err_q3  = nan(N,1);
err_T   = nan(N,1);
fk_T    = nan(N,3);
ik_dd1  = nan(N,1);
ik_dd2  = nan(N,1);
ik_q3   = nan(N,1);
success = false(N,1);
%用 NaN 初始化是为了方便后续识别哪些帧没有被成功处理

for i = 1:N
    %% 3.1 正运动学：输入(dd1,dd2,q3) → 输出T
    [T_pos, ~, info_fk] = fk_chain_to_T(dd1_seq(i), dd2_seq(i), q3_seq(i), p);
    
    if ~info_fk.success
        fprintf('帧%d: FK失败，跳过\n', i);
        %fprintf：MATLAB内置函数，用于格式化输出文本到命令窗口。
        % 这里输出当前帧号和 FK 失败的提示信息。
        continue;
    end
    fk_T(i,:) = T_pos(:)';
    
    %% 3.2 逆运动学：输入T → 输出(q1,q2,q3,q4) → (dd1,dd2)
    [q1_sol, q2_sol, info_ik] = ik_q1q2_new(T_pos(:)', p);
    
    if ~info_ik.success || isempty(q1_sol)
        fprintf('帧%d: IK q1q2失败，跳过\n', i);
        continue;
    end
    
    q1_ik = q1_sol(1);
    q2_ik = q2_sol(1);
    q3_ik = info_ik.q3;
    
    [~, ~, ~, ~, dd1_ik, dd2_ik, info_d] = ik_d1_d2(q1_ik, q2_ik, p);
    
    if ~info_d.success
        fprintf('帧%d: IK d1d2失败，跳过\n', i);
        continue;
    end
    
    %% 3.3 计算误差
    ik_dd1(i) = dd1_ik;
    ik_dd2(i) = dd2_ik;
    ik_q3(i)  = q3_ik;
    
    err_d1(i) = abs(dd1_ik - dd1_seq(i));
    err_d2(i) = abs(dd2_ik - dd2_seq(i));
    err_q3(i) = abs(q3_ik  - q3_seq(i));
    
    % 用IK恢复的控制量再做一次FK，验证T的一致性
    [T_verify, ~, info_v] = fk_chain_to_T(dd1_ik, dd2_ik, q3_ik, p);
    if info_v.success
        err_T(i) = norm(T_verify(:)' - T_pos(:)');
    end
    
    success(i) = true;
    
    if mod(i,20)==0
        fprintf('帧%d/%d: err_d1=%.4f err_d2=%.4f err_q3=%.4f err_T=%.6f\n', ...
            i, N, err_d1(i), err_d2(i), err_q3(i), err_T(i));
    end
end

%% 4. 统计分析
valid = success;
n_valid = sum(valid);
n_fail  = N - n_valid;

fprintf('\n');
fprintf('========================================\n');
fprintf('       正逆运动学精度验证报告\n');
fprintf('========================================\n');
fprintf('总测试点: %d\n', N);
fprintf('成功: %d  失败: %d  成功率: %.1f%%\n\n', n_valid, n_fail, 100*n_valid/N);

% 总体误差统计
fprintf('--- 总体误差统计 ---\n');
fprintf('%-12s %12s %12s %12s %12s\n', '指标', '平均值', '最大值', '中位数', '标准差');
fprintf('%s\n', repmat('-',1,60));
print_stat('Δd1 (mm)', err_d1(valid));
print_stat('Δd2 (mm)', err_d2(valid));
print_stat('q3 (°)',   err_q3(valid));
print_stat('T距离(mm)',err_T(valid));

% 分阶段统计
fprintf('\n--- 分阶段误差统计 ---\n');
for s = 1:5
    idx = stage_bounds(s,1):stage_bounds(s,2);
    v = valid(idx);
    if sum(v) == 0
        fprintf('\n[%s] 无有效数据\n', stage_names{s});
        continue;
    end
    fprintf('\n[%s] (有效%d/%d帧)\n', stage_names{s}, sum(v), length(idx));
    fprintf('%-12s %12s %12s %12s %12s\n', '指标', '平均值', '最大值', '中位数', '标准差');
    fprintf('%s\n', repmat('-',1,60));
    print_stat('Δd1 (mm)', err_d1(idx(v)));
    print_stat('Δd2 (mm)', err_d2(idx(v)));
    print_stat('q3 (°)',   err_q3(idx(v)));
    print_stat('T距离(mm)',err_T(idx(v)));
end

%% 5. 可视化
fig = figure('Color','w','Position',[100,100,1200,800]);
sgtitle('正逆运动学精度验证', 'FontSize', 16, 'FontWeight', 'bold');

% 5.1 Δd1 误差
subplot(2,3,1);
plot(find(valid), err_d1(valid), 'b.-'); hold on;
for s=1:5, xline(stage_bounds(s,1),'--r'); end
xlabel('帧号'); ylabel('|误差| (mm)');
title('Δd1 误差'); grid on;

% 5.2 Δd2 误差
subplot(2,3,2);
plot(find(valid), err_d2(valid), 'r.-'); hold on;
for s=1:5, xline(stage_bounds(s,1),'--r'); end
xlabel('帧号'); ylabel('|误差| (mm)');
title('Δd2 误差'); grid on;

% 5.3 q3 误差
subplot(2,3,3);
plot(find(valid), err_q3(valid), 'g.-'); hold on;
for s=1:5, xline(stage_bounds(s,1),'--r'); end
xlabel('帧号'); ylabel('|误差| (°)');
title('q3 误差'); grid on;

% 5.4 T点距离误差
subplot(2,3,4);
semilogy(find(valid), err_T(valid), 'm.-'); hold on;
for s=1:5, xline(stage_bounds(s,1),'--r'); end
xlabel('帧号'); ylabel('距离误差 (mm)');
title('末端T距离误差 (对数)'); grid on;

% 5.5 输入vs恢复 d1
subplot(2,3,5);
plot(1:N, dd1_seq, 'b-', 'LineWidth', 1.5); hold on;
plot(find(valid), ik_dd1(valid), 'r--', 'LineWidth', 1);
for s=1:5, xline(stage_bounds(s,1),'--k','Alpha',0.3); end
legend('输入Δd1','IK恢复Δd1'); xlabel('帧号'); ylabel('mm');
title('Δd1: 输入 vs IK恢复'); grid on;

% 5.6 输入vs恢复 q3
subplot(2,3,6);
plot(1:N, q3_seq, 'b-', 'LineWidth', 1.5); hold on;
plot(find(valid), ik_q3(valid), 'r--', 'LineWidth', 1);
for s=1:5, xline(stage_bounds(s,1),'--k','Alpha',0.3); end
legend('输入q3','IK恢复q3'); xlabel('帧号'); ylabel('°');
title('q3: 输入 vs IK恢复'); grid on;

% 保存图片
saveas(fig, 'fk_ik_verify_result.png');
fprintf('\n误差分析图已保存: fk_ik_verify_result.png\n');

%% 6. 保存结果到MAT
verify_result.dd1_seq = dd1_seq;
verify_result.dd2_seq = dd2_seq;
verify_result.q3_seq  = q3_seq;
verify_result.ik_dd1  = ik_dd1;
verify_result.ik_dd2  = ik_dd2;
verify_result.ik_q3   = ik_q3;
verify_result.err_d1  = err_d1;
verify_result.err_d2  = err_d2;
verify_result.err_q3  = err_q3;
verify_result.err_T   = err_T;
verify_result.fk_T    = fk_T;
verify_result.success = success;
verify_result.stage_bounds = stage_bounds;
verify_result.stage_names  = stage_names;
save('fk_ik_verify_result.mat', '-struct', 'verify_result');
fprintf('验证结果已保存: fk_ik_verify_result.mat\n');
fprintf('\n=== 验证完成 ===\n');

%% =========================================================================
function print_stat(name, data)
    if isempty(data)
        fprintf('%-12s %12s\n', name, '无数据');
        return;
    end
    fprintf('%-12s %12.6f %12.6f %12.6f %12.6f\n', ...
        name, mean(data), max(data), median(data), std(data));
end
