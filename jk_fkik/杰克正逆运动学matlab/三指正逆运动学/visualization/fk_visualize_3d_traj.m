%% =========================================================================
%   正运动学三维可视化 - 轨迹跟踪（正方形 + 圆形）
%   加载IK轨迹规划的控制量，用FK生成机构运动视频
%   依赖：params.m, fk_generate_points.m
%   输入：ik_traj_square_seq.mat, ik_traj_circle_seq.mat
%   输出：fk_traj_square.mp4, fk_traj_circle.mp4
% =========================================================================
close all; clear; clc;

%% 1. 加载轨迹数据
fprintf('=== 加载轨迹控制量 ===\n');
sq = load('ik_traj_square_seq.mat');
ci = load('ik_traj_circle_seq.mat');

% 同时加载完整轨迹数据（用于绘制目标轨迹）
td = load('ik_trajectory_data.mat');

%% 2. 生成正方形视频
fprintf('\n=== 正方形轨迹视频 ===\n');
gen_traj_video(sq.sq_delta_d1_seq, sq.sq_delta_d2_seq, sq.sq_q3_seq, ...
    td.square_traj.T_points, '正方形轨迹', 'fk_traj_square.mp4');

%% 3. 生成圆形视频
fprintf('\n=== 圆形轨迹视频 ===\n');
gen_traj_video(ci.cir_delta_d1_seq, ci.cir_delta_d2_seq, ci.cir_q3_seq, ...
    td.circle_traj.T_points, '圆形轨迹', 'fk_traj_circle.mp4');

fprintf('\n=== 全部完成 ===\n');

%% =========================================================================
function gen_traj_video(d1_seq, d2_seq, q3_seq, target_pts, traj_name, video_file)
    p = params();
    num_frames = length(d1_seq);
    
    % FK求解
    fprintf('FK求解 %d 帧...\n', num_frames);
    [all_pts, all_info] = fk_generate_points(d1_seq, d2_seq, q3_seq);
    
    % 颜色定义
    cb = [0.4,0.4,0.4]; cp = [0,0.45,0.74]; cm1 = [0.85,0.33,0.10];
    cm2 = [0.47,0.67,0.19]; cc = [0.64,0.08,0.18]; ct = [0.8,0,0.8];
    
    % 视角
    az0 = 135; el0 = 25;
    az_step = 360 / num_frames;
    
    % 视频
    v = VideoWriter(video_file, 'MPEG-4');
    v.FrameRate = 10; v.Quality = 95; open(v);
    
    T_trail = nan(num_frames, 3);
    
    for j = 1:num_frames
        if isempty(all_pts(j).O)
            fprintf('跳过帧%d\n', j); continue;
        end
        pt = all_pts(j); inf = all_info(j);
        T_trail(j,:) = pt.T(:)';
        
        fig = figure('Visible','off','Color','w');
        set(fig,'Units','pixels','Position',[100,50,1000,800]);
        ax = axes('Parent',fig); hold(ax,'on');
        
        % 绘制目标轨迹（半透明灰色虚线）
        plot3(ax, target_pts(:,1), target_pts(:,2), target_pts(:,3), ...
              '--', 'Color', [0.6,0.6,0.6], 'LineWidth', 1.5);
        
        % 基座
        dl(ax,pt.O,pt.P,cb,2); dl(ax,pt.O,pt.A1,cb,2);
        dl(ax,pt.O,pt.A2,cb,2); dl(ax,pt.P,pt.A1,cb,2); dl(ax,pt.P,pt.A2,cb,2);
        % 并联
        dl(ax,pt.P,pt.B1,cp,2.5); dl(ax,pt.P,pt.B2,cp,2.5); dl(ax,pt.B1,pt.B2,cp,2.5);
        % 电机
        dl(ax,pt.A1,pt.C1,cm1,2,'--'); dl(ax,pt.C1,pt.B1,cm1,2);
        dl(ax,pt.A2,pt.C2,cm2,2,'--'); dl(ax,pt.C2,pt.B2,cm2,2);
        % 三杆
        dl(ax,pt.P,pt.M,cc,3); dl(ax,pt.M,pt.N,cc,3); dl(ax,pt.N,pt.T,cc,3);
        
        % 关键点
        dp(ax,pt.O,[0,0,0],7,'o'); dp(ax,pt.P,[1,0,0],10,'o');
        dp(ax,pt.T,ct,10,'p'); dp(ax,pt.M,[0,0.8,0],7,'o'); dp(ax,pt.N,[0.8,0.8,0],7,'o');
        dp(ax,pt.A1,[0,0.75,0.75],7,'s'); dp(ax,pt.A2,[0,0.75,0.75],7,'s');
        dp(ax,pt.C1,[0.93,0.69,0.13],7,'^'); dp(ax,pt.C2,[0.93,0.69,0.13],7,'^');
        
        % 标签
        lb(ax,pt.O,'O'); lb(ax,pt.P,'P'); lb(ax,pt.T,'T');
        lb(ax,pt.M,'M'); lb(ax,pt.N,'N');
        
        % 末端轨迹
        valid = T_trail(1:j,:); valid = valid(~any(isnan(valid),2),:);
        if size(valid,1)>1
            plot3(ax,valid(:,1),valid(:,2),valid(:,3),'-','Color',[ct,0.6],'LineWidth',2);
        end
        
        % 坐标轴参考
        al = 8;
        quiver3(ax,0,0,0,al,0,0,0,'r','LineWidth',1.5);
        quiver3(ax,0,0,0,0,al,0,0,'g','LineWidth',1.5);
        quiver3(ax,0,0,0,0,0,al,0,'b','LineWidth',1.5);
        
        % 基座平面
        bs = 20;
        fill3(ax,[-bs,bs,bs,-bs],[-bs,-bs,bs,bs],[0,0,0,0],...
              [0.9,0.9,0.9],'FaceAlpha',0.3,'EdgeColor',[0.7,0.7,0.7]);
        
        % 坐标轴设置
        axis(ax,'equal');
        xlim(ax,[-60,60]); ylim(ax,[-40,120]); zlim(ax,[-10,160]);
        grid(ax,'on'); ax.GridAlpha=0.3;
        xlabel(ax,'X (mm)'); ylabel(ax,'Y (mm)'); zlabel(ax,'Z (mm)');
        view(ax, az0, el0);
        lighting(ax,'gouraud'); camlight(ax,'headlight');
        
        % 标题
        title(ax, sprintf('三指手 %s (帧%d/%d)', traj_name, j, num_frames), ...
              'FontSize',13,'FontWeight','bold');
        
        % 信息文本
        str = sprintf(['\\bf控制量: \\color[rgb]{0.85,0.33,0.10}\\Deltad1=%+.2f ' ...
              '\\color[rgb]{0.47,0.67,0.19}\\Deltad2=%+.2f ' ...
              '\\color[rgb]{0.64,0.08,0.18}q3=%+.1f°\n' ...
              '\\color{black}末端T: [%.2f, %.2f, %.2f] mm'], ...
              d1_seq(j), d2_seq(j), q3_seq(j), pt.T(1), pt.T(2), pt.T(3));
        annotation(fig,'textbox',[0.05,0.83,0.90,0.10],'String',str,...
            'Interpreter','tex','FitBoxToText','on','FontSize',11,...
            'HorizontalAlignment','center',...
            'BackgroundColor',[1,1,0.85],'EdgeColor',[0.8,0.6,0],'LineWidth',2);
        
        % 左下角：关节角和电机位移信息窗口
        str_info = sprintf(['关节角 (°):\n' ...
                            'q1=%+.2f  q2=%+.2f\n' ...
                            'q3=%+.2f  q4=%+.2f\n' ...
                            '电机位移 (mm):\n' ...
                            'd1=%.3f  d2=%.3f'], ...
                            inf.q1, inf.q2, inf.q3, inf.q4, ...
                            inf.d1, inf.d2);
        annotation(fig,'textbox',[0.01,0.01,0.22,0.15],'String',str_info,...
            'FitBoxToText','on','FontSize',9,...
            'BackgroundColor','w','EdgeColor',[0.5,0.5,0.5]);
        
        % 图例
        h = [];
        h(end+1) = plot3(ax,nan,nan,nan,'--','Color',[0.6,0.6,0.6],'LineWidth',1.5);
        h(end+1) = plot3(ax,nan,nan,nan,'-','Color',ct,'LineWidth',2);
        h(end+1) = plot3(ax,nan,nan,nan,'-','Color',cc,'LineWidth',2);
        legend(ax,h,{'目标轨迹','实际轨迹','三杆机构'},...
               'Location','northeast','FontSize',8);
        
        drawnow;
        frame = getframe(fig);
        writeVideo(v, frame);
        if mod(j,20)==0, fprintf('帧%d/%d\n',j,num_frames); end
        close(fig);
    end
    close(v);
    fprintf('%s 视频已保存: %s\n', traj_name, video_file);
end

%% 辅助函数
function dl(ax,p1,p2,c,lw,ls)
    if nargin<6, ls='-'; end
    p1=p1(:)'; p2=p2(:)';
    plot3(ax,[p1(1),p2(1)],[p1(2),p2(2)],[p1(3),p2(3)],ls,'Color',c,'LineWidth',lw);
end

function dp(ax,pt,c,ms,mk)
    pt=pt(:)';
    plot3(ax,pt(1),pt(2),pt(3),mk,'MarkerSize',ms,'MarkerFaceColor',c,...
          'MarkerEdgeColor',c*0.6,'LineWidth',1.2);
end

function lb(ax,pt,s)
    pt=pt(:)';
    text(ax,pt(1)+2,pt(2)+2,pt(3)+2,s,'FontSize',8,'FontWeight','bold','Color',[0.2,0.2,0.2]);
end
