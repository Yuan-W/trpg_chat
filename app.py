import streamlit as st
import json
import random
import os
import copy
import time
import uuid
from openai import OpenAI
from datetime import datetime
from streamlit_local_storage import LocalStorage

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
    # 优先导出整个 Local Storage 中的数据
    if "storage_data" in st.session_state:
        return json.dumps(st.session_state["storage_data"], ensure_ascii=False, indent=2)

    # Fallback 到当前单次会话
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
        
        # 情况 1: 全量备份 (包含 "sessions")
        if "sessions" in data:
            st.session_state["storage_data"] = data
            st.session_state["data_loaded"] = True  # 标记为已加载，允许保存

            # 尝试恢复当前会话
            current_id = data.get("current_session_id")
            sessions = data.get("sessions", {})

            if current_id and current_id in sessions:
                st.session_state["current_session_id"] = current_id
                sess = sessions[current_id]

                # 恢复 current_script
                script_path = sess.get("current_script")
                
                # 准备消息 (System + History)
                system_msgs = copy.deepcopy(DEFAULT_CONFIG["initial_messages"])
                
                if script_path:
                    st.session_state["current_script"] = script_path
                    fresh_mask = parse_nextchat_mask(script_path)
                    if fresh_mask:
                        st.session_state["mask_config"] = fresh_mask
                        system_msgs = fresh_mask.get("initial_messages", [])

                saved_msgs = sess.get("messages", [])
                st.session_state.messages = system_msgs + saved_msgs
                st.session_state["long_term_memory"] = sess.get("long_term_memory", "")
                
                st.toast(f"✅ 全局存档已加载！恢复会话: {sess.get('name', 'Unknown')}")
            else:
                st.toast("✅ 全局存档已加载！(未找到活跃会话)")

            save_to_local_storage() # 同步到浏览器
            time.sleep(1)
            st.rerun()
            return

        # 情况 2: 单次会话备份 (包含 "messages")
        if "messages" not in data:
            raise ValueError("缺少消息记录")

        st.session_state.messages = data["messages"]
        st.session_state["long_term_memory"] = data.get("long_term_memory", "")
        # 兼容旧存档，如果没有 config 则使用默认
        st.session_state["mask_config"] = data.get("mask_config", DEFAULT_CONFIG)
        
        st.toast(f"✅ 存档已加载！时间: {data.get('timestamp', 'Unknown')}")
        save_to_local_storage() # 保存为当前会话
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
    """解析 NextChat 格式的 JSON，支持扩展字段"""
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
            "historyMessageCount": mc.get("historyMessageCount", 20),
            "initial_messages": initial_messages,
            # 新增扩展字段
            "tailPrompt": mask_data.get("tailPrompt", ""),
            "negativeConstraints": mask_data.get("negativeConstraints", []),
            "glossary": mask_data.get("glossary", {}),
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
            with open(os.path.join(folder, "readme.txt"), "w") as f:
                f.write("请将 NextChat 导出的 JSON 文件放入此文件夹")
        except:
            return []

    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".json")]
    return files


# ================= 5. LocalStorage Manager =================
KEY_LOCAL_STORAGE = "trpg_chat_data_v1"

# 初始化 LocalStorage 实例
localS = LocalStorage()

def load_from_local_storage():
    """从浏览器读取数据 (仅在初始化时调用)"""
    # 如果已经加载过，直接返回
    if st.session_state.get("data_loaded", False):
        return

    # 使用 streamlit-local-storage 的 getItem
    data_str = localS.getItem(KEY_LOCAL_STORAGE)

    # 逻辑优化：处理异步加载
    if data_str is not None:
        # 情况 A: 成功读取到数据
        st.session_state["data_loaded"] = True
        st.session_state["load_retries"] = 0 # reset
        if data_str and isinstance(data_str, str):
            try:
                data = json.loads(data_str)
                st.session_state["storage_data"] = data
                # 恢复当前会话
                current_id = data.get("current_session_id")
                sessions = data.get("sessions", {})

                if current_id and current_id in sessions:
                    st.session_state["current_session_id"] = current_id
                    sess = sessions[current_id]

                    # 恢复 current_script (用于加载 mask)
                    script_path = sess.get("current_script")
                    if script_path:
                        st.session_state["current_script"] = script_path
                        # 从文件加载最新的 mask_config (包括 system prompts)
                        fresh_mask = parse_nextchat_mask(script_path)
                        if fresh_mask:
                            st.session_state["mask_config"] = fresh_mask
                            # 合并: 新 system prompts + 保存的 user/assistant 对话
                            system_msgs = fresh_mask.get("initial_messages", [])
                            saved_msgs = sess.get("messages", [])
                            st.session_state.messages = system_msgs + saved_msgs
                        else:
                            st.session_state.messages = sess.get("messages", copy.deepcopy(DEFAULT_CONFIG["initial_messages"]))
                    else:
                        st.session_state.messages = sess.get("messages", copy.deepcopy(DEFAULT_CONFIG["initial_messages"]))

                    st.session_state["long_term_memory"] = sess.get("long_term_memory", "")

                    st.toast(f"已恢复会话: {sess.get('name', 'Unknown')}")
            except Exception as e:
                st.error(f"读取存档失败: {e}")
    else:
        # 情况 B: 读取为 None (可能是加载中，也可能是 Key 不存在)
        # 增加重试计数，防止无限等待导致新用户无法 Save
        retries = st.session_state.get("load_retries", 0) + 1
        st.session_state["load_retries"] = retries
        print(f"DEBUG: Load returned None. Retry count: {retries}")

        # 认为超过 2 次就是真的没有数据 (新用户)
        if retries > 2:
            print("DEBUG: Assumed New User (Empty Storage). Enabling Save.")
            st.session_state["data_loaded"] = True

def save_to_local_storage():
    """将当前状态保存到 storage_data 并写入浏览器"""
    # 关键修复：如果必须等待加载完成才能保存，否则会覆盖掉旧数据
    if not st.session_state.get("data_loaded", False):
        print("DEBUG: Skipping save because data not loaded yet.")
        return

    if "current_session_id" not in st.session_state:
        create_new_session()

    session_id = st.session_state["current_session_id"]

    # 1. 更新内存中的 storage_data
    if "storage_data" not in st.session_state:
        st.session_state["storage_data"] = {"sessions": {}, "current_session_id": session_id}

    sessions = st.session_state["storage_data"]["sessions"]

    # 提取对话摘要作为标题
    name = "新会话"
    if len(st.session_state.messages) > 1:
        # 取第一条 User 消息的前 15 个字
        for m in st.session_state.messages:
            if m["role"] == "user":
                name = m["content"][:15]
                break

    # 只保存用户生成的数据，不保存 mask_config (会变旧) 和 initial_messages (从文件加载)
    # 过滤掉 system messages，只保存 user/assistant 对话
    user_messages = [m for m in st.session_state.messages if m["role"] != "system"]

    sessions[session_id] = {
        "id": session_id,
        "name": name,
        "timestamp": time.time(),
        "messages": user_messages,  # 只保存对话，不包含 system prompt
        "long_term_memory": st.session_state.get("long_term_memory", ""),
        "current_script": st.session_state.get("current_script")
    }
    st.session_state["storage_data"]["current_session_id"] = session_id

    # 2. 使用 streamlit-local-storage 的 setItem 保存
    json_str = json.dumps(st.session_state["storage_data"], ensure_ascii=False)
    # 使用唯一 key 避免 Streamlit 的 duplicate key 错误
    save_key = f"save_{int(time.time()*1000)}"
    localS.setItem(KEY_LOCAL_STORAGE, json_str, key=save_key)

def create_new_session():
    new_id = str(uuid.uuid4())
    st.session_state["current_session_id"] = new_id

    # 逻辑优化: 确定使用哪套配置
    # 1. 如果当前已经加载了某个剧本 (current_script exists), 则继承之 (Mask config & persistence)
    # 2. 如果当前是 Default (current_script None), 但 masks 文件夹里有文件, 则默认加载第一个文件 (Selection 0)
    # 3. 否则才使用纯净的 DEFAULT_CONFIG

    config_to_use = DEFAULT_CONFIG

    if st.session_state.get("current_script"):
        config_to_use = st.session_state.get("mask_config", DEFAULT_CONFIG)
    else:
        files = get_mask_files()
        if files:
            # 尝试加载第一个文件
            first_file = files[0]
            parsed = parse_nextchat_mask(first_file)
            if parsed:
                config_to_use = parsed
                st.session_state["current_script"] = first_file

    st.session_state.messages = copy.deepcopy(config_to_use.get("initial_messages", DEFAULT_CONFIG["initial_messages"]))
    st.session_state["long_term_memory"] = ""
    st.session_state["mask_config"] = copy.deepcopy(config_to_use)

    return new_id

def delete_session(session_id):
    if "storage_data" in st.session_state:
        sessions = st.session_state["storage_data"].get("sessions", {})
        if session_id in sessions:
            del sessions[session_id]
            # 如果删除了当前会话，新建一个
            if st.session_state.get("current_session_id") == session_id:
                create_new_session()
            save_to_local_storage()
            st.rerun()

def switch_session(session_id):
    if "storage_data" in st.session_state:
        sessions = st.session_state["storage_data"].get("sessions", {})
        if session_id in sessions:
            sess = sessions[session_id]
            st.session_state["current_session_id"] = session_id
            st.session_state.messages = sess.get("messages", [])
            st.session_state["long_term_memory"] = sess.get("long_term_memory", "")
            st.session_state["mask_config"] = sess.get("mask_config", DEFAULT_CONFIG)
            save_to_local_storage() # 更新 timestamp
            st.rerun()

# ================= 6. 初始化与侧边栏 =================

# 0. 加载本地存储 (最优先)
load_from_local_storage()

# 1. 初始化 Session State
if "messages" not in st.session_state:
    st.session_state.messages = copy.deepcopy(DEFAULT_CONFIG["initial_messages"])
if "long_term_memory" not in st.session_state:
    st.session_state["long_term_memory"] = ""
if "mask_config" not in st.session_state:
    st.session_state["mask_config"] = copy.deepcopy(DEFAULT_CONFIG)
if "current_session_id" not in st.session_state:
    create_new_session()

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

    # --- 📚 会话管理 (NextChat style) ---
    st.subheader("💬 会话历史")

    if st.button("➕ 新建对话", use_container_width=True):
        create_new_session()
        st.rerun()

    sessions = st.session_state.get("storage_data", {}).get("sessions", {})
    # 按时间倒序
    sorted_sessions = sorted(sessions.values(), key=lambda x: x.get("timestamp", 0), reverse=True)

    # 显示最近 10 条
    for s in sorted_sessions[:10]:
        col1, col2 = st.columns([4, 1])
        with col1:
             # 当前会话高亮
            label = s.get("name", "未命名")
            if s["id"] == st.session_state.get("current_session_id"):
                st.info(f"📌 {label}")
            else:
                if st.button(label, key=f"btn_{s['id']}"):
                    switch_session(s["id"])
        with col2:
            if st.button("x", key=f"del_{s['id']}", help="删除"):
                delete_session(s["id"])

    st.divider()

    # --- 🎭 剧本管理 ---
    st.write("📖 **剧本导入**")
    mask_files = get_mask_files()
    selected_file = (
        st.selectbox("选择剧本文件:", mask_files, index=0, format_func=lambda x: os.path.basename(x)) if mask_files else None
    )

    if selected_file:
        # 如果当前没有配置，或者切换了文件，则重新加载
        # 但如果刚刚从 LocalStorage 恢复了会话，不要覆盖
        already_loaded = st.session_state.get("data_loaded") and len(st.session_state.get("messages", [])) > 1

        if (
            "current_script" not in st.session_state
            or st.session_state["current_script"] != selected_file
        ) and not already_loaded:
            config_data = parse_nextchat_mask(selected_file)
            if config_data:
                st.session_state["mask_config"] = config_data
                st.session_state["current_script"] = selected_file
                st.session_state.messages = copy.deepcopy(
                    config_data["initial_messages"]
                )
                st.session_state["long_term_memory"] = ""
                st.success(f"已装载: {config_data['name']}")
                save_to_local_storage() # 加载剧本也自动保存
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

    # Auto-save dice roll
    save_to_local_storage()

    # --- 💾 存档管理 ---
    st.divider()
    with st.expander("💾 记忆与存档", expanded=False):
        ltm = st.session_state.get("long_term_memory", "")
        st.caption(f"🧠 长期记忆摘要 ({len(ltm)} 字)：")
        if ltm:
            st.text_area(
                "Memory",
                value=ltm,
                height=200,  # 增加高度
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            st.info("暂无压缩记忆。对话超过 ~25 条时会自动生成摘要。")

        uploaded_save = st.file_uploader("读取存档 (.json)", type=["json"])
        if uploaded_save:
            if st.button("⚠️ 确认覆盖当前进度", type="primary"):
                load_save_data(uploaded_save)

        st.download_button(
            label="⬇️ 导出所有数据",
            data=export_save_data(),
            file_name=f"Backup_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json",
        )
        st.caption("注：这会导出当前所有会话历史")

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

    # 立即保存用户消息
    save_to_local_storage()

    # 2. 准备上下文
    mask_cfg = st.session_state["mask_config"]

    # --- 记忆压缩逻辑 (TRPG 优化版) ---
    # historyMessageCount: 发送给 AI 的最大消息数
    # 当消息超过该阈值+缓冲区时，压缩旧消息为 long_term_memory
    threshold = mask_cfg.get("historyMessageCount", 20)  # 提高默认值，适合 TRPG 长对话
    keep_count = max(int(threshold / 2), 5)  # 保留至少一半或 5 条，确保上下文连贯

    all_messages = st.session_state.messages
    system_msgs = [m for m in all_messages if m["role"] == "system"]
    chat_msgs = [m for m in all_messages if m["role"] != "system"]

    if len(chat_msgs) > (threshold + 3):  # 更早触发压缩 (+3 而非 +5)
        with st.status("🧠 正在整理记忆...", expanded=True) as status:
            msgs_to_compress = chat_msgs[:-keep_count]  # 保留最后 N 条，压缩前面的
            msgs_to_keep = chat_msgs[-keep_count:]

            current_ltm = st.session_state.get("long_term_memory", "")

            print(f"DEBUG: Compressing {len(msgs_to_compress)} messages, keeping {len(msgs_to_keep)}")

            new_summary = summarize_memory(
                client,
                mask_cfg["model"],
                msgs_to_compress,
                current_ltm,
            )

            print(f"DEBUG: summarize_memory returned: {type(new_summary)} - '{str(new_summary)[:100] if new_summary else 'EMPTY/NONE'}'...")

            st.session_state["long_term_memory"] = new_summary if new_summary else ""

            # 重构消息列表：System + Remaining
            st.session_state.messages = system_msgs + msgs_to_keep

            chat_msgs = msgs_to_keep

            # 显示压缩结果摘要
            st.write(f"**已压缩 {len(msgs_to_compress)} 条消息**")
            if new_summary:
                st.text_area("新摘要预览", value=new_summary[:500] + "...", height=150, disabled=True)

            status.update(label="✅ 记忆已更新", state="complete", expanded=False)

            # 压缩后立即保存，防止刷新丢失
            save_to_local_storage()

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

    # --- 注入扩展字段 (最后注入以增强效果) ---

    # 检查 mask_cfg 是否包含新字段，如果没有则尝试从文件重新加载
    if not mask_cfg.get("glossary") and st.session_state.get("current_script"):
        # 尝试从文件重新读取
        refreshed = parse_nextchat_mask(st.session_state["current_script"])
        if refreshed and refreshed.get("glossary"):
            # 合并新字段到现有 config
            mask_cfg["glossary"] = refreshed.get("glossary", {})
            mask_cfg["negativeConstraints"] = refreshed.get("negativeConstraints", [])
            mask_cfg["tailPrompt"] = refreshed.get("tailPrompt", "")
            st.session_state["mask_config"] = mask_cfg
            print("DEBUG: Refreshed mask_config with new fields from file")

    # (A) 术语对照表 (Glossary)
    glossary = mask_cfg.get("glossary", {})
    print(f"DEBUG: Glossary has {len(glossary)} entries")
    if glossary:
        glossary_text = "【术语对照 / Glossary】\n" + "\n".join([f"- {en}: {zh}" for en, zh in glossary.items()])
        final_messages.append({"role": "system", "content": glossary_text})

    # (B) 负面约束 (Negative Constraints)
    neg_constraints = mask_cfg.get("negativeConstraints", [])
    print(f"DEBUG: negativeConstraints has {len(neg_constraints)} entries")
    if neg_constraints:
        constraints_text = "【禁止事项 / Negative Constraints】\n" + "\n".join([f"❌ {c}" for c in neg_constraints])
        final_messages.append({"role": "system", "content": constraints_text})

    # (C) 尾部指令 (Tail Prompt) - 最后注入
    tail_prompt = mask_cfg.get("tailPrompt", "")
    print(f"DEBUG: tailPrompt = '{tail_prompt[:50]}...' " if tail_prompt else "DEBUG: tailPrompt is empty")
    if tail_prompt:
        final_messages.append({"role": "system", "content": tail_prompt})

    print(f"DEBUG: Total messages to send: {len(final_messages)}")

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
        # 保存 AI 回复
        save_to_local_storage()

    except Exception as e:
        st.error(f"API 请求失败: {e}")
