function p = vis_params()
    % VIS_PARAMS 返回拇指机构可视化参数结构体
    %
    % 功能：统一管理所有可视化相关的配置参数
    % 说明：修改此文件中的参数将影响所有可视化脚本
    
    %% ==================== 视频参数 ====================
    % 视频输出设置
    
    % 文件名
    p.video_filename = 'visualization/thumb_ik_animation_3d.avi';
    p.mat_filename = 'visualization/thumb_ik_vis_params.mat';
    
    % 视频格式
    p.video_format = 'Motion JPEG AVI';  % AVI 格式
    p.frame_rate = 10;                   % 帧率 (fps)
    
    %% ==================== 图形窗口设置 ====================
    % MATLAB 图形窗口属性
    
    % 窗口大小 (像素)
    p.fig_width = 1200;
    p.fig_height = 800;
    p.fig_position = [100, 50, p.fig_width, p.fig_height];
    
    % 窗口背景色
    p.fig_color = 'w';  % 白色
    
    % 可见性 (生成视频时设为 'off')
    p.fig_visible = 'off';
    
    %% ==================== 颜色定义 ====================
    % RGB 颜色值 [R, G, B]，范围 0-1
    
    % 连杆颜色
    p.color_OM   = [0.0, 0.45, 0.74];   % OM 连杆 - 蓝色
    p.color_MN   = [0.0, 0.8, 0.0];     % MN 连杆 - 绿色
    p.color_NT   = [0.64, 0.08, 0.18];  % NT 连杆 - 深红
    p.color_traj = [0.8, 0, 0.8];       % 末端轨迹 - 品红
    
    % 关键点颜色
    p.color_O  = [0, 0, 0];              % 原点 - 黑色
    p.color_M  = [0, 0.75, 0.75];        % M 点 - 青色
    p.color_N  = [0.93, 0.69, 0.13];     % N 点 - 金色
    p.color_T  = [0.8, 0, 0.8];          % T 点 - 品红
    p.color_nc = [1, 0, 0];              % 法向量 - 红色
    
    % 坐标轴颜色
    p.color_X  = [1, 0, 0];              % X 轴 - 红色
    p.color_Y  = [0, 0.6, 0];            % Y 轴 - 绿色
    p.color_Z  = [0, 0, 1];              % Z 轴 - 蓝色
    
    % 背景颜色
    p.color_base = [0.9, 0.9, 0.9];      % 基座平面 - 浅灰
    p.color_grid = [0.7, 0.7, 0.7];      % 网格线 - 灰色
    
    %% ==================== 线宽和点大小 ====================
    
    % 连杆线宽
    p.lw_OM   = 3.0;
    p.lw_MN   = 2.5;
    p.lw_NT   = 2.0;
    p.lw_traj = 1.5;
    
    % 点标记大小
    p.ms_small  = 7;   % O、N 点
    p.ms_large  = 10;  % M、T 点
    
    % 法向量线宽
    p.lw_nc = 2;
    
    %% ==================== 三维视角设置 ====================
    
    % 初始视角 (度)
    p.az_init = -45;   % 方位角
    p.el_init = 25;    % 仰角
    
    % 旋转设置
    p.rotate_full_circle = true;           % 是否全程旋转 360 度
    p.az_rotate_per_frame = [];            % 每帧旋转角度 (动态计算)
    
    %% ==================== 坐标轴范围 ====================
    % 3D 空间显示范围 (mm)
    
    p.xlim = [-80, 80];
    p.ylim = [-60, 80];
    p.zlim = [-10, 120];
    
    % 基座平面大小
    p.base_size = 30;
    
    % 坐标轴箭头长度
    p.arrow_len = 10;
    
    %% ==================== 网格设置 ====================
    
    p.grid_on = true;
    p.grid_alpha = 0.3;
    p.grid_linestyle = ':';
    p.axis_equal = true;
    
    %% ==================== 字体设置 ====================
    
    p.font_size_title = 13;
    p.font_size_label = 11;
    p.font_size_text = 9;
    p.font_size_legend = 8;
    p.font_weight_title = 'bold';
    p.font_weight_label = 'bold';
    p.font_name = 'Microsoft YaHei';  % 微软雅黑
    
    %% ==================== 信息框设置 ====================
    
    % 输入信息框 (顶部)
    p.input_box_position = [0.05, 0.85, 0.90, 0.08];
    p.input_box_bg_color = [1, 1, 0.85];
    p.input_box_edge_color = [0.8, 0.6, 0.0];
    p.input_box_font_size = 12;
    
    % 关节角信息框 (左下角)
    p.joint_box_position = [0.01, 0.01, 0.28, 0.12];
    p.joint_box_bg_color = 'w';
    p.joint_box_edge_color = [0.5, 0.5, 0.5];
    p.joint_box_font_size = 9;
    
    % 阶段信息框 (中下方)
    p.stage_box_position = [0.35, 0.01, 0.35, 0.10];
    p.stage_box_bg_color = [1, 1, 0.9];
    p.stage_box_edge_color = [0.5, 0.5, 0.5];
    p.stage_box_font_size = 9;
    
    % 法向量信息框 (右下角)
    p.nc_box_position = [0.75, 0.01, 0.24, 0.10];
    p.nc_box_bg_color = 'w';
    p.nc_box_edge_color = [0.5, 0.5, 0.5];
    p.nc_box_font_size = 9;
    
    %% ==================== 轨迹设置 ====================
    
    % 轨迹透明度
    p.trail_alpha = 0.5;
    
    %% ==================== 运动阶段 ====================
    % 五阶段运动规划定义
    
    % 每个半阶段帧数
    p.n_half = 50;
    
    % 阶段 5 帧数
    p.n5 = 4 * p.n_half;  % 120 帧
    
    % 阶段 1: q2 变化范围
    p.stage1_q2_min = 0;
    p.stage1_q2_max = -55;
    
    % 阶段 2: q1 变化范围
    p.stage2_q1_min = 0;
    p.stage2_q1_max = -90;
    
    % 阶段 3: q3, q4 变化范围
    p.stage3_q3_min = 0;
    p.stage3_q3_max = -85;
    p.stage3_q4_min = 0;
    p.stage3_q4_max = -70;
    
    % 阶段 4: q1, q2 同步变化范围
    p.stage4_q1_min = 0;
    p.stage4_q1_max = -90;
    p.stage4_q2_min = 0;
    p.stage4_q2_max = -55;
    
    % 阶段 5: 四关节协同变化幅度
    p.stage5_q1_amp = -40;
    p.stage5_q2_amp = -25;
    p.stage5_q3_amp = -40;
    p.stage5_q4_amp = -35;
    
    % 阶段 5 频率因子
    p.stage5_q1_freq = 1.0;
    p.stage5_q2_freq = 1.2;
    p.stage5_q3_freq = 1.0;
    p.stage5_q4_freq = 1.1;
    
    %% ==================== 图例设置 ====================
    
    p.legend_location = 'northeast';
    p.legend_items = {'OM 连杆', 'MN 连杆', 'NT 连杆', '平面法向量', '末端轨迹'};
    
    %% ==================== 光照设置 ====================
    
    p.lighting_method = 'gouraud';
    p.camlight_type = 'headlight';
end
