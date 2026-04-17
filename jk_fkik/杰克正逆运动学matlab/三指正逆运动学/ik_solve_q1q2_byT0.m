function [q1_deg, q2_deg, error] = ik_solve_q1q2_byT0(P, T0, T_current)
% SOLVE_ROTATION  由旋转后T点坐标反推旋转角q1, q2
%
% 旋转定义：以P点为旋转中心，先绕x轴旋转q1，再绕y轴旋转q2
% 坐标系：x轴指向屏幕外，y轴指向右，z轴竖直向上
%
% ── 输入（已知量）────────────────────────────────────────────
%   P         : [3x1] 或 [1x3]，旋转中心坐标（P点固定不动）
%   T0        : [3x1] 或 [1x3]，q1=q2=0时 T 的绝对坐标
%   T_current : [3x1] 或 [1x3]，旋转后 T 的绝对坐标
%
% ── 输出（待求量）────────────────────────────────────────────
%   q1_deg    : 绕x轴旋转角（度），范围约束 [-90°, +15°]
%   q2_deg    : 绕y轴旋转角（度），范围约束 [-15°, +15°]
%   error     : 重建误差（坐标单位），理想情况下接近0
%
% ── 角度范围约束 ─────────────────────────────────────────────
%   q1 ∈ [-90°, +15°]
%   q2 ∈ [-15°, +15°]
%
% ── 用法示例 ─────────────────────────────────────────────────
%   P         = [0, 0, 27.98];
%   T0        = [0, 7.432, 124.097];
%   T_current = [...];
%   [q1, q2, err] = solve_rotation(P, T0, T_current);

    % ── 角度范围定义（度） ────────────────────────────────────
    Q1_MIN = -90.0;
    Q1_MAX =  15.0;
    Q2_MIN = -45.0;
    Q2_MAX =  45.0;

    % ── 统一转为列向量 ────────────────────────────────────────
    P         = P(:);
    T0        = T0(:);
    T_current = T_current(:);

    % ════════════════════════════════════════════════════════
    % 第1步：构造相对P点的PT向量（消去P点偏置）
    % ════════════════════════════════════════════════════════
    v0 = T0        - P;    % 初始PT向量，分量为 (a, b, c)
    vp = T_current - P;    % 旋转后PT向量，分量为 (a', b', c')

    a  = v0(1);  b  = v0(2);  c  = v0(3);
    ap = vp(1);  bp = vp(2);  cp = vp(3);

    % ════════════════════════════════════════════════════════
    % 第2步：校验刚体约束（模长应相等）
    % ════════════════════════════════════════════════════════
    norm0 = norm(v0);
    normp = norm(vp);
    if abs(norm0 - normp) / (norm0 + 1e-12) > 1e-4
        error('solve_rotation:rigidBodyViolation', ...
              '向量模长不一致: |v0|=%.6f, |v''|=%.6f\n请检查输入坐标是否满足刚体约束', ...
              norm0, normp);
    end

    % ════════════════════════════════════════════════════════
    % 第3步：求 q1 的两个候选值
    %
    % 由旋转方程第2行：bp = b*cos(q1) - c*sin(q1)
    % 令 rho = sqrt(b^2+c^2)，phi = atan2(c, b)
    % 则：rho * cos(q1 + phi) = bp
    % 解：q1 = -phi ± arccos(bp / rho)
    % ════════════════════════════════════════════════════════
    rho = sqrt(b^2 + c^2);

    if rho < 1e-12
        % b=c=0，q1不可观测
        warning('solve_rotation:q1Unobservable', ...
                'v0的y/z分量均为0，q1不可观测，设q1=0');
        q1_candidates = [0.0];
    else
        cos_val = bp / rho;
        cos_val = max(-1.0, min(1.0, cos_val));   % 防止浮点误差越界
        phi     = atan2(c, b);
        delta   = acos(cos_val);
        % 两个候选解
        q1_candidates = [-phi + delta, ...        % 候选解1
                         -phi - delta];           % 候选解2info.success = (exitflag > 0) && (abs(fval) < 1e-5);
    end

    % ════════════════════════════════════════════════════════
    % 第4步：对每个q1候选值，解析求q2，并筛选范围
    %
    % 将Rx(q1)作用于v0，得中间向量u：
    %   ux = a                          （x分量不受Rx影响）
    %   uy = b*cos(q1) - c*sin(q1)     （= bp，由q1保证）
    %   uz = b*sin(q1) + c*cos(q1)
    %
    % 再由Ry(q2)*u = vp的第1、3行：
    %   ap =  ux*cos(q2) + uz*sin(q2)
    %   cp = -ux*sin(q2) + uz*cos(q2)
    % 解：q2 = atan2(uz, ux) - atan2(cp, ap)
    % ════════════════════════════════════════════════════════

    % 预分配结果存储
    valid_q1  = [];
    valid_q2  = [];
    valid_err = [];

    for k = 1 : length(q1_candidates)

        q1_raw = q1_candidates(k);

        % 归一化到 (-180°, 180°] 对应的弧度范围
        q1_rad = wrap_to_pi(q1_raw);
        q1_d   = rad2deg(q1_rad);

        % ── 检查q1是否在允许范围内 ───────────────────────────
        if q1_d < Q1_MIN || q1_d > Q1_MAX
            fprintf('  候选解%d: q1=%.4f° 超出范围[%g°, %g°]，跳过\n', ...
                    k, q1_d, Q1_MIN, Q1_MAX);
            continue;
        end

        % ── 用原始q1（未截断）计算中间量S，保证数值一致性 ────
        ux = a;
        uz = b * sin(q1_rad) + c * cos(q1_rad);   % = S

        % ── 解析求q2 ──────────────────────────────────────────
        q2_rad = atan2(uz, ux) - atan2(cp, ap);
        q2_rad = wrap_to_pi(q2_rad);
        q2_d   = rad2deg(q2_rad);

        % ── 检查q2是否在允许范围内 ───────────────────────────
        if q2_d < Q2_MIN || q2_d > Q2_MAX
            fprintf('  候选解%d: q1=%.4f°, q2=%.4f° 超出范围[%g°, %g°]，跳过\n', ...
                    k, q1_d, q2_d, Q2_MIN, Q2_MAX);
            continue;
        end

        % ── 计算重建误差 ───────────────────────────────────────
        v_recon = apply_rotation(v0, q1_rad, q2_rad);
        err     = norm(v_recon - vp);

        valid_q1  = [valid_q1,  q1_d];
        valid_q2  = [valid_q2,  q2_d];
        valid_err = [valid_err, err];

        fprintf('  候选解%d: q1=%.6f°, q2=%.6f°, 重建误差=%.2e ✓\n', ...
                k, q1_d, q2_d, err);
    end

    % ════════════════════════════════════════════════════════
    % 第5步：从有效解中选误差最小的作为输出
    % ════════════════════════════════════════════════════════
    if isempty(valid_err)
        %isempty是MATLAB内置函数，用于检查数组是否为空。
        error('solve_rotation:noValidSolution', ...
              '没有找到满足角度范围约束的有效解！\n请检查输入数据或放宽角度范围');
    end

    [error, best_idx] = min(valid_err);
    q1_deg = valid_q1(best_idx);
    q2_deg = valid_q2(best_idx);

end


% ════════════════════════════════════════════════════════════
% 子函数1：施加旋转（先绕x轴q1，再绕y轴q2）
% ════════════════════════════════════════════════════════════
function v_out = apply_rotation(v, q1, q2)
% APPLY_ROTATION  对向量v施加旋转：先Rx(q1)，再Ry(q2)
%   v    : [3x1] 输入向量
%   q1   : 绕x轴角度（弧度）
%   q2   : 绕y轴角度（弧度）
%   v_out: [3x1] 旋转后向量

    Rx = [1,        0,         0;
          0,  cos(q1), -sin(q1);
          0,  sin(q1),  cos(q1)];

    Ry = [ cos(q2), 0, sin(q2);
                 0, 1,       0;
          -sin(q2), 0, cos(q2)];

    v_out = Ry * Rx * v;
end


% ════════════════════════════════════════════════════════════
% 子函数2：角度归一化到 (-π, π]
% ════════════════════════════════════════════════════════════
function angle = wrap_to_pi(angle)
% WRAP_TO_PI  将弧度值归一化到 (-π, π]
    angle = mod(angle + pi, 2*pi) - pi;
end
