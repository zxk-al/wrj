import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time

# ===================== 全局常量 =====================
MAX_DRONES = 10000
DEFAULT_DRONES = 150
SIM_STEP = 0.04
GROUND_Z = 0
SKY_Z = 80
TRAIL_MAX_LEN = 80

# ===================== 会话状态初始化 =====================
if "drone_pos" not in st.session_state:
    st.session_state.drone_pos = np.zeros((MAX_DRONES, 3))
if "drone_target" not in st.session_state:
    st.session_state.drone_target = np.zeros((MAX_DRONES, 3))
if "drone_trails" not in st.session_state:
    st.session_state.drone_trails = [[] for _ in range(MAX_DRONES)]
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "drone_num" not in st.session_state:
    st.session_state.drone_num = DEFAULT_DRONES
# 自定义点位（改用手动输入坐标）
if "custom_points" not in st.session_state:
    st.session_state.custom_points = []
# 场景障碍物（城市建筑群）
if "obstacles" not in st.session_state:
    st.session_state.obstacles = [
        {"pos": (25, 25), "r": 7, "h": 38},
        {"pos": (-18, 12), "r": 5, "h": 32},
        {"pos": (5, -28), "r": 9, "h": 42},
        {"pos": (-30, -20), "r": 6, "h": 30}
    ]

# ===================== 1. 预设队形生成 =====================
def gen_preset_fleet(shape, count):
    count = min(count, MAX_DRONES)
    pos = np.zeros((MAX_DRONES, 3))
    target = np.zeros((MAX_DRONES, 3))

    if shape == "矩形方阵":
        row = int(np.ceil(np.sqrt(count)))
        idx = 0
        for y in range(row):
            for x in range(row):
                if idx >= count:
                    break
                pos[idx] = [x * 6, y * 6, GROUND_Z]
                target[idx] = [x * 6, y * 6, 32]
                idx += 1

    elif shape == "环形编队":
        r = max(8, count / 4)
        for i in range(count):
            ang = 2 * np.pi * i / count
            x = r * np.cos(ang)
            y = r * np.sin(ang)
            pos[i] = [x, y, GROUND_Z]
            target[i] = [x, y, 35]

    elif shape == "一字横队":
        for i in range(count):
            pos[i] = [i * 5, 0, GROUND_Z]
            target[i] = [i * 5, 0, 28]

    elif shape == "三角编队":
        idx = 0
        layer = 1
        while idx < count:
            offset = (layer - 1) * 3
            for i in range(layer):
                if idx >= count:
                    break
                x = i * 5 - offset
                pos[idx] = [x, layer * 5, GROUND_Z]
                target[idx] = [x, layer * 5, 25]
                idx += 1
            layer += 1

    elif shape == "随机散点":
        pos[:count] = np.random.rand(count, 3) * 40 - 20
        pos[:count, 2] = GROUND_Z
        target[:count] = pos[:count] + [0, 0, 22]

    st.session_state.drone_trails = [[] for _ in range(MAX_DRONES)]
    return pos, target

# ===================== 2. 自定义队形生成 =====================
def gen_custom_fleet(point_list, count):
    count = min(count, MAX_DRONES)
    pos = np.zeros((MAX_DRONES, 3))
    target = np.zeros((MAX_DRONES, 3))
    point_num = len(point_list)

    if point_num == 0:
        for i in range(count):
            pos[i] = [0, 0, GROUND_Z]
            target[i] = [0, 0, 30]
    else:
        for i in range(count):
            px, py = point_list[i % point_num]
            pos[i] = [px, py, GROUND_Z]
            target[i] = [px, py, 30]

    st.session_state.drone_trails = [[] for _ in range(MAX_DRONES)]
    return pos, target

# ===================== 3. 飞行物理 + 避障更新 =====================
def update_flight():
    num = st.session_state.drone_num
    obs_list = st.session_state.obstacles

    for i in range(num):
        p = st.session_state.drone_pos[i]
        t = st.session_state.drone_target[i]

        delta = t - p
        dist = np.linalg.norm(delta)
        if dist > 0.1:
            p += delta * SIM_STEP

        for obs in obs_list:
            ox, oy = obs["pos"]
            or_ = obs["r"]
            oh = obs["h"]
            dx = p[0] - ox
            dy = p[1] - oy
            d_xy = np.hypot(dx, dy)

            if d_xy < or_ + 4 and p[2] < oh:
                push = (or_ + 4 - d_xy) / (or_ + 4) * 0.4
                p[0] += dx / (d_xy + 1e-6) * push
                p[1] += dy / (d_xy + 1e-6) * push
                p[2] += 0.15

        p[2] = np.clip(p[2], GROUND_Z, SKY_Z)

        st.session_state.drone_trails[i].append(p.copy())
        if len(st.session_state.drone_trails[i]) > TRAIL_MAX_LEN:
            st.session_state.drone_trails[i].pop(0)

# ===================== 4. 高写实3D场景渲染 =====================
def render_real_scene():
    drone_num = st.session_state.drone_num
    drone_pos = st.session_state.drone_pos
    trails = st.session_state.drone_trails
    obs = st.session_state.obstacles
    custom_pts = st.session_state.custom_points

    fig = go.Figure()

    # 1. 写实地面
    gx = np.linspace(-65, 65, 25)
    gy = np.linspace(-65, 65, 25)
    gxx, gyy = np.meshgrid(gx, gy)
    gzz = np.zeros_like(gxx)
    fig.add_trace(go.Surface(
        x=gxx, y=gyy, z=gzz,
        colorscale=[[0, "#2c3e50"], [1, "#4a5568"]],
        opacity=0.8, showscale=False, name="城市地面"
    ))

    # 2. 建筑群
    for building in obs:
        bx, by = building["pos"]
        br = building["r"]
        bh = building["h"]
        theta = np.linspace(0, 2 * np.pi, 24)
        circ_x = bx + br * np.cos(theta)
        circ_y = by + br * np.sin(theta)
        h_range = np.linspace(0, bh, 2)
        cx_mesh, cz_mesh = np.meshgrid(circ_x, h_range)
        cy_mesh, _ = np.meshgrid(circ_y, h_range)

        fig.add_trace(go.Surface(
            x=cx_mesh, y=cy_mesh, z=cz_mesh,
            colorscale=[[0, "#34495e"], [1, "#57606f"]],
            opacity=0.9, showscale=False, name="楼房"
        ))

    # 3. 自定义点位标记
    if len(custom_pts) > 0:
        pts_arr = np.array(custom_pts)
        fig.add_trace(go.Scatter3d(
            x=pts_arr[:, 0], y=pts_arr[:, 1], z=np.zeros(len(pts_arr)),
            mode="markers", marker=dict(size=6, color="red", symbol="diamond"),
            name="自定义点位"
        ))

    # 4. 无人机飞行轨迹
    for i in range(min(drone_num, 20)):
        tr = np.array(trails[i])
        if len(tr) > 2:
            fig.add_trace(go.Scatter3d(
                x=tr[:, 0], y=tr[:, 1], z=tr[:, 2],
                mode="lines", line=dict(color="rgba(100,200,255,0.4)", width=1.5),
                showlegend=False
            ))

    # 5. 无人机集群
    drones = drone_pos[:drone_num]
    fig.add_trace(go.Scatter3d(
        x=drones[:, 0], y=drones[:, 1], z=drones[:, 2],
        mode="markers",
        marker=dict(
            size=4.5,
            color=drones[:, 2],
            colorscale="Plasma",
            opacity=0.92,
            line=dict(color="white", width=0.3)
        ),
        name="无人机"
    ))

    # 6. 场景设置
    fig.update_layout(
        title="🏙️ 全真城市环境 - 无人机编队仿真",
        title_font=dict(size=16, color="#f0f0f0"),
        scene=dict(
            xaxis=dict(range=[-65, 65], title="X 轴(米)", color="#ddd"),
            yaxis=dict(range=[-65, 65], title="Y 轴(米)", color="#ddd"),
            zaxis=dict(range=[0, SKY_Z], title="高度 Z(米)", color="#ddd"),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.4))
        ),
        height=720,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        scene_bgcolor="#1e293b"
    )
    return fig

# ===================== 页面主体 UI =====================
st.set_page_config(page_title="全真无人机编队系统", layout="wide")
st.title("🏙️ 全真城市环境 · 无人机编队仿真平台")
st.markdown("✅ 预设队形 + ✅ 坐标自定义编队 + ✅ 建筑避障 + ✅ 飞行轨迹 | 零AI训练，简单易上手")

# 左侧控制面板
with st.sidebar:
    st.header("⚙️ 操作面板")
    st.divider()

    # 1. 无人机数量
    drone_cnt = st.number_input("无人机数量", min_value=1, max_value=MAX_DRONES, value=DEFAULT_DRONES, step=10)
    st.session_state.drone_num = drone_cnt

    # 2. 编队模式选择
    mode = st.radio("编队模式", ["预设队形", "坐标自定义队形"])

    # 3. 预设队形选择
    preset_list = ["矩形方阵", "环形编队", "一字横队", "三角编队", "随机散点"]
    if mode == "预设队形":
        select_shape = st.selectbox("选择队形", preset_list)
    else:
        # 坐标输入
        st.subheader("添加自定义点位")
        col1, col2 = st.columns(2)
        x_coord = col1.number_input("X坐标", min_value=-60.0, max_value=60.0, value=0.0, step=1.0)
        y_coord = col2.number_input("Y坐标", min_value=-60.0, max_value=60.0, value=0.0, step=1.0)
        if st.button("➕ 添加点位", use_container_width=True):
            st.session_state.custom_points.append((x_coord, y_coord))
            st.success(f"已添加点位 ({x_coord}, {y_coord})")
        if st.button("🧹 清空所有点位", use_container_width=True):
            st.session_state.custom_points = []
            st.success("已清空所有点位")
        st.info(f"当前已添加点位数量：{len(st.session_state.custom_points)}")

    st.divider()

    # 4. 功能按钮
    if mode == "预设队形":
        btn_init = st.button("🔧 初始化预设编队", use_container_width=True)
    else:
        btn_init = st.button("🔧 生成自定义编队", use_container_width=True)

    btn_start = st.button("▶️ 开始飞行仿真", use_container_width=True)
    btn_pause = st.button("⏸️ 暂停飞行", use_container_width=True)

    st.divider()
    st.info("""
    💡 使用教程：
    1. 预设队形：选样式 → 初始化 → 开始飞行
    2. 自定义队形：在侧边栏输入X/Y坐标 → 添加点位 → 生成编队
    3. 无人机遇楼房会自动避让
    """)

# 主画面容器
view_holder = st.empty()

# 底部状态栏
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.info(f"运行状态：{'飞行中 🟢' if st.session_state.is_running else '已暂停 🔴'}")
c2.info(f"无人机总数：{st.session_state.drone_num} 架")
c3.info(f"建筑物数量：{len(st.session_state.obstacles)} 栋")
c4.info(f"自定义点位：{len(st.session_state.custom_points)} 个")

# ===================== 功能逻辑执行 =====================
# 初始化编队
if btn_init:
    if mode == "预设队形":
        pos, target = gen_preset_fleet(select_shape, drone_cnt)
    else:
        pos, target = gen_custom_fleet(st.session_state.custom_points, drone_cnt)
    st.session_state.drone_pos = pos
    st.session_state.drone_target = target
    st.session_state.is_running = False
    st.success("编队初始化完成！")

# 飞行启停
if btn_start:
    st.session_state.is_running = True
if btn_pause:
    st.session_state.is_running = False

# 渲染画面（常驻渲染，无空白）
try:
    fig = render_real_scene()
    view_holder.plotly_chart(fig, use_container_width=True)
except Exception as e:
    view_holder.error(f"渲染异常：{str(e)}")

# 飞行循环（安全刷新）
if st.session_state.is_running:
    update_flight()
    time.sleep(0.08)
    st.rerun()
