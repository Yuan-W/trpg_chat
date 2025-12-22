import streamlit as st
import json
import random
import os
import copy
import time
from openai import OpenAI
from datetime import datetime

# ================= 1. 基础配置与工具函数 =================
st.set_page_config(page_title="暗夜刀锋 GM", page_icon="🗡️", layout="wide")

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* 骰子结果样式 */
    .dice-result {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin-bottom: 10px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# 默认配置
DEFAULT_CONFIG = {
    "model": "gemini-3-flash-preview",
    "temperature": 1.0,
    "top_p": 1.0,
    "max_tokens": 4000,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "historyMessageCount": 20,
    # 如果没有 JSON，默认只有一条 System
    "initial_messages": [{"role": "system", "content": "你是一个冷酷的暗夜刀锋GM。"}],
}

def get_config(key, default=None):
    """
    优先从系统环境变量获取 (Zeabur/Docker)，
    获取不到则尝试从 st.secrets 获取 (Local)，
    最后返回默认值。
    """
    # 1. 尝试系统环境变量 (Zeabur)
    value = os.environ.get(key)
    if value:
        return value
    
    # 2. 尝试 st.secrets (Local)
    # 注意：st.secrets 可能会报错如果key不存在，所以用 .get()
    try:
        if key in st.secrets:
            return st.secrets[key]
    except FileNotFoundError:
        pass # 本地没有 secrets.toml 文件
        
    return default


def get_api_client():
    """获取 OpenAI 客户端，优先从 Secrets 读取，否则从 Sidebar 读取"""
    api_key = get_config("API_KEY")
    base_url = get_config("BASE_URL")

    # 如果 Session 中有（用户在侧边栏输入的）
    if not api_key and "user_api_key" in st.session_state:
        api_key = st.session_state["user_api_key"]
        base_url = st.session_state["user_base_url"]

    if not api_key:
        return None

    return OpenAI(api_key=api_key, base_url=base_url)


# ================= 2. 存档系统 =================
def export_save_data():
    save_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": st.session_state.messages,
        "long_term_memory": st.session_state.get("long_term_memory", ""),
        "mask_config": st.session_state.get("mask_config", DEFAULT_CONFIG),
    }
    return json.dumps(save_data, ensure_ascii=False, indent=2)


def load_save_data(uploaded_file):
    try:
        data = json.load(uploaded_file)
        if "messages" not in data:
            raise ValueError("缺少消息记录")

        st.session_state.messages = data["messages"]
        st.session_state["long_term_memory"] = data.get("long_term_memory", "")
        # 兼容旧存档，如果没有 config 则使用默认
        st.session_state["mask_config"] = data.get("mask_config", DEFAULT_CONFIG)
        st.toast(f"✅ 存档已加载！时间: {data['timestamp']}")
        time.sleep(1)
        st.rerun()
    except Exception as e:
        st.error(f"坏档或格式错误: {e}")


# ================= 3. 记忆总结引擎 =================
def summarize_memory(client, model, messages_to_summarize, current_summary):
    if not client:
        return current_summary

    summary_prompt = "请简要总结以下跑团剧情的发生经过、关键决策和当前状态。保留NPC名字和重要的物品/后果。不要遗漏关键信息。"
    if current_summary:
        summary_prompt += f"\n\n已知前情提要：{current_summary}"

    # 清洗消息，去除 'is_dice' 等自定义字段，否则 API 会报错
    dialogue_content = []
    for m in messages_to_summarize:
        if m["role"] in ["user", "assistant"]:
            dialogue_content.append({"role": m["role"], "content": str(m["content"])})

    msgs = [{"role": "system", "content": "你是一个专业的跑团记录员。"}]
    msgs.extend(dialogue_content)
    msgs.append({"role": "user", "content": summary_prompt})

    try:
        response = client.chat.completions.create(
            model=model, messages=msgs, max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Summary Error: {e}")  # 打印后台日志
        return current_summary


# ================= 4. Mask 解析器 =================
def parse_nextchat_mask(file_path):
    """解析 NextChat 格式的 JSON"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容 NextChat 导出格式 (可能是个 list 或者是 dict)
        mask_data = (data["masks"][0] if "masks" in data and isinstance(data["masks"], list) else data)

        raw_context = mask_data.get("context", [])
        initial_messages = []
        for msg in raw_context:
            if msg.get("role") and msg.get("content"):
                initial_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        # 如果 Mask 里没写 Context，就用默认的
        if not initial_messages:
            initial_messages = copy.deepcopy(DEFAULT_CONFIG["initial_messages"])

        mc = mask_data.get("modelConfig", {})
        config = {
            "name": mask_data.get("name", "未命名剧本"),
            "model": mc.get("model", DEFAULT_CONFIG["model"]),
            "temperature": mc.get("temperature", DEFAULT_CONFIG["temperature"]),
            "top_p": mc.get("top_p", DEFAULT_CONFIG["top_p"]),
            "max_tokens": mc.get("max_tokens", DEFAULT_CONFIG["max_tokens"]),
            "presence_penalty": mc.get(
                "presence_penalty", DEFAULT_CONFIG["presence_penalty"]
            ),
            "frequency_penalty": mc.get(
                "frequency_penalty", DEFAULT_CONFIG["frequency_penalty"]
            ),
            "historyMessageCount": mc.get("historyMessageCount", 10),
            "initial_messages": initial_messages,
        }
        return config
    except Exception as e:
        st.error(f"JSON 解析错误: {e}")
        return None


def get_mask_files():
    folder = "masks"
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
            # 创建一个示例文本
            with open(os.path.join(folder, "readme.txt"), "w") as f:
                f.write("请将 NextChat 导出的 JSON 文件放入此文件夹")
        except:
            return []  # 权限不足等情况，回退

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    return files

# ================= 5. 侧边栏与初始化 =================

# 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = copy.deepcopy(DEFAULT_CONFIG["initial_messages"])
if "long_term_memory" not in st.session_state:
    st.session_state["long_term_memory"] = ""
if "mask_config" not in st.session_state:
    st.session_state["mask_config"] = copy.deepcopy(DEFAULT_CONFIG)

with st.sidebar:
    st.title("控制台")

    client = get_api_client()
    if not client:
        st.warning("⚠️ 未检测到 API 配置")
        with st.expander("配置 API Key", expanded=True):
            st.text_input("API Key", key="user_api_key", type="password")
            st.text_input(
                "Base URL", key="user_base_url", value="https://api.openai.com/v1"
            )
            if st.button("保存配置"):
                st.rerun()
        st.stop()  # 停止渲染主界面

    # --- 🎭 剧本管理 ---

    mask_files = get_mask_files()
    selected_file = (
        st.selectbox("📚 选择剧本文件:", mask_files, index=0, format_func=lambda x: os.path.basename(x)) if mask_files else None
    )

    if selected_file:
        # 如果当前没有配置，或者切换了文件，则重新加载
        if (
            "current_script" not in st.session_state
            or st.session_state["current_script"] != selected_file
        ):
            config_data = parse_nextchat_mask(selected_file)
            if config_data:
                st.session_state["mask_config"] = config_data
                st.session_state["current_script"] = selected_file
                st.session_state.messages = copy.deepcopy(
                    config_data["initial_messages"]
                )
                st.session_state["long_term_memory"] = ""
                st.success(f"已装载: {config_data['name']}")
                time.sleep(0.5)
                st.rerun()

     # --- 🎲 骰子系统 ---
    st.divider()

    action_dots = st.slider("骰子数量", 1, 6, 2)
    if st.button("🎲 投掷!", use_container_width=True):
        with st.spinner("🎲 命运流转中..."):
            rolls = [random.randint(1, 6) for _ in range(action_dots)]
            result = max(rolls)

            # 结果判定与颜色渲染
            if result == 6 and rolls.count(6) > 1:
                outcome = "🔴 **暴击 (CRIT)**"
            elif result == 6:
                outcome = "🟢 **完全成功 (6)**"
            elif result >= 4:
                outcome = "🟡 **代价成功 (4/5)**"
            else:
                outcome = "⚫ **失败 (1-3)**"

            msg_content = f"(系统广播: 玩家投掷了 {action_dots} 个骰子，结果: {rolls} -> {outcome.replace('*','').replace('<br>','')})"

            # 添加系统消息到历史
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": msg_content,
                    "is_dice": True,
                }
            )
        st.rerun()

    # --- 💾 存档管理 ---
    st.divider()
    with st.expander("💾 记忆与存档", expanded=False):
        st.caption("🧠 长期记忆摘要：")
        st.text_area(
            "Memory",
            value=st.session_state.get("long_term_memory", ""),
            height=100,
            disabled=True,
        )

        uploaded_save = st.file_uploader("读取存档 (.json)", type=["json"])
        if uploaded_save:
            if st.button("⚠️ 确认覆盖当前进度", type="primary"):
                load_save_data(uploaded_save)

        st.download_button(
            label="⬇️ 下载当前存档",
            data=export_save_data(),
            file_name=f"Save_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
        )

# ================= 6. 主聊天界面 =================
mask_cfg = st.session_state.get("mask_config", {})
st.title(f"{mask_cfg.get('name', '暗夜刀锋 GM')}")

# 🌟 修改后的渲染逻辑：包含 System Prompt 🌟
for msg in st.session_state.messages:
    if msg["role"] == "system":
        # 排除掉后期自动生成的"前情提要" (通常以【前情提要】开头)，只显示原始设定
        if "【前情提要" in msg["content"]:
            continue
        with st.chat_message("system", avatar="📜"):
            with st.expander(
                f"查看剧本设定: {mask_cfg.get('name', '系统')}", expanded=False
            ):
                st.markdown(msg["content"])
        continue  # 处理完 System 后跳过，不走下面的通用渲染

    # --- 2. 正常处理 User / Assistant ---
    avatar = "👤" if msg["role"] == "user" else '🤖'
    if msg.get("is_dice"): avatar = "🎲"

    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# 处理用户输入
if prompt := st.chat_input("描述你的行动..."):
    # 1. 显示用户输入
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 2. 准备上下文
    mask_cfg = st.session_state["mask_config"]

    # --- 记忆压缩逻辑 ---
    threshold = mask_cfg.get("historyMessageCount", 10)
    keep_count = int(threshold / 3)

    all_messages = st.session_state.messages
    system_msgs = [m for m in all_messages if m["role"] == "system"]
    chat_msgs = [m for m in all_messages if m["role"] != "system"]

    if len(chat_msgs) > (threshold + 5):
        with st.status("🧠 正在整理记忆...", expanded=True) as status:
            msgs_to_compress = chat_msgs[:-keep_count]  # 保留最后 N 条，压缩前面的
            msgs_to_keep = chat_msgs[-keep_count:]

            current_ltm = st.session_state.get("long_term_memory", "")

            new_summary = summarize_memory(
                client,
                mask_cfg["model"],
                msgs_to_compress,
                current_ltm,
            )

            st.session_state["long_term_memory"] = new_summary

            # 重构消息列表：System + Remaining
            st.session_state.messages = system_msgs + msgs_to_keep

            chat_msgs = msgs_to_keep

            status.update(label="记忆已更新", state="complete", expanded=False)

    # --- 构建最终 Prompt ---
    final_messages = []

    # (1) 先放入所有的原始 System Prompt
    final_messages.extend(system_msgs)

    # 插入长期记忆
    if st.session_state["long_term_memory"]:
        final_messages.append(
            {
                "role": "system",
                "content": f"【前情提要 / Long Term Memory】\n{st.session_state['long_term_memory']}",
            }
        )

    # 插入最近对话 (过滤掉骰子的 HTML 标记，只保留 content 用于推理)
    for m in st.session_state.messages[1:]:
        clean_msg = {"role": m["role"], "content": m["content"]}
        final_messages.append(clean_msg)

    # 3. AI 生成回复
    try:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("⏳ GM 正在构思..."):
                stream = client.chat.completions.create(
                    model=mask_cfg["model"],
                    messages=final_messages,
                    stream=True,
                    temperature=mask_cfg["temperature"],
                    top_p=mask_cfg["top_p"],
                    max_tokens=mask_cfg["max_tokens"],
                    presence_penalty=mask_cfg["presence_penalty"],
                    frequency_penalty=mask_cfg["frequency_penalty"],
                )
                response = st.write_stream(stream)

        st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"API 请求失败: {e}")
