import os
from typing import TypedDict, List
from datetime import datetime

from dotenv import load_dotenv

# LangGraph
from langgraph.graph import StateGraph, END

# 模型和工具
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage   # ✅ 修正这行

# WxPusher SDK
from wxpusher import WxPusher
# 加载环境变量（本地调试用，GitHub Actions 中由 Secrets 注入）
load_dotenv()

# ==================== 配置 ====================
# 自定义你关心的关键词（可以随时修改）
KEYWORDS = [
    "人工智能 最新新闻",
    "大模型 最新进展",
    "AI Agent 应用",
]

# WxPusher 配置
WXPUSHER_APP_TOKEN = os.getenv("WXPUSHER_APP_TOKEN")
WXPUSHER_UID = os.getenv("WXPUSHER_UID")

# ==================== 初始化模型 ====================
model = ChatDeepSeek(
    model="deepseek-v4-flash",  # 或 deepseek-chat
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.3,
    max_tokens=2048,
)

# ==================== 定义工作流状态 ====================
class NewsState(TypedDict):
    keywords: List[str]
    raw_articles: str
    report: str

# ==================== 节点 1：搜索新闻 ====================
def fetch_news_node(state: NewsState) -> dict:
    """使用 Tavily 搜索每个关键词的最新新闻（含来源链接）"""
    print(f"[fetch_news] 开始搜索 {len(state['keywords'])} 个关键词...")

    all_results = []
    for keyword in state["keywords"]:
        try:
            tool = TavilySearch(
                max_results=3,
                tavily_api_key=os.getenv("TAVILY_API_KEY")
            )
            result = tool.invoke(f"{keyword} {datetime.now().strftime('%Y-%m-%d')}")
            # Tavily 返回 dict，含 results 列表（title/url/content）
            articles = result.get("results", []) if isinstance(result, dict) else []
            if not articles:
                all_results.append(f"### 关键词：{keyword}\n（未找到相关结果）\n")
                print(f"[fetch_news] ⚠️ 无结果：{keyword}")
                continue

            lines = [f"### 关键词：{keyword}"]
            for i, article in enumerate(articles, 1):
                title = article.get("title", "无标题") or "无标题"
                url = article.get("url", "") or ""
                content = (article.get("content", "") or "")[:300]
                lines.append(f'{i}. **[{title}]({url})**')
                if content:
                    lines.append(f"   > {content}")
            all_results.append("\n".join(lines))
            print(f"[fetch_news] ✅ 完成：{keyword}（{len(articles)} 条）")
        except Exception as e:
            print(f"[fetch_news] ❌ 搜索失败 [{keyword}]：{e}")
            all_results.append(f"### 关键词：{keyword}\n搜索失败：{e}\n")

    return {"raw_articles": "\n\n".join(all_results)}

# ==================== 节点 2：生成简报 ====================
def summarize_node(state: NewsState) -> dict:
    """使用 DeepSeek 将搜索结果总结为每日简报"""
    print("[summarize] 正在生成新闻简报...")

    prompt = f"""
你是一个专业的新闻编辑。请根据以下搜索到的新闻片段，生成一份今日新闻简报。

要求：
1. 按关键词分类，每个关键词下列出 2-3 条最相关的新闻
2. 每条新闻用一句话概括
3. **每条新闻必须包含对应的来源链接**，用 [标题](URL) 的 Markdown 链接格式附在新闻标题上，方便读者点击查看原文
4. 整体语言简洁清晰，适合手机阅读
5. 在简报末尾添加一句简短的今日总结
6. 不要添加无关内容，不要编造新闻

搜索内容：
{state["raw_articles"]}
"""
    try:
        response = model.invoke([HumanMessage(content=prompt)])
        report = response.content
        print(f"[summarize] ✅ 简报生成完成，字数：{len(report)}")
        return {"report": report}
    except Exception as e:
        error_msg = f"简报生成失败：{e}"
        print(f"[summarize] ❌ {error_msg}")
        return {"report": error_msg}

# ==================== 节点 3：推送到微信 (WxPusher) ====================
import requests
import json


# ... 在 send_notification_node 函数中 ...

def send_notification_node(state: NewsState) -> dict:
    """通过 WxPusher 将简报推送到微信"""
    print("[send_notification] 正在推送到微信...")

    title = f"📰 每日 AI 新闻 ({datetime.now().strftime('%m月%d日')})"
    content = state["report"]
    full_message = f"## {title}\n\n{content}"

    # 准备请求数据
    url = "https://wxpusher.zjiecode.com/api/send/message"
    headers = {"Content-Type": "application/json"}
    body = {
        "appToken": os.getenv("WXPUSHER_APP_TOKEN"),
        "content": full_message,
        "contentType": 3,  # 1=文字, 2=HTML, 3=Markdown
        "uids": [os.getenv("WXPUSHER_UID")],
        "verifyPayType": 0
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(body), timeout=30)
        result = resp.json()

        if result.get("code") == 1000:
            print("[send_notification] ✅ 推送成功")
        else:
            print(f"[send_notification] ❌ 推送失败，返回信息：{result}")

    except Exception as e:
        print(f"[send_notification] ❌ 推送请求异常：{e}")

    return state

# ==================== 组装 LangGraph 工作流 ====================
def build_news_agent():
    """构建并编译新闻 Agent 工作流"""
    builder = StateGraph(NewsState)

    # 添加节点
    builder.add_node("fetch_news", fetch_news_node)
    builder.add_node("summarize", summarize_node)
    builder.add_node("send_notification", send_notification_node)

    # 连接边：顺序执行
    builder.set_entry_point("fetch_news")
    builder.add_edge("fetch_news", "summarize")
    builder.add_edge("summarize", "send_notification")
    builder.add_edge("send_notification", END)

    return builder.compile()

# ==================== 主函数 ====================
def run_daily_news():
    """执行每日新闻任务"""
    print(f"\n{'='*50}")
    print(f"🚀 每日新闻 Agent 启动")
    print(f"时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"关键词：{', '.join(KEYWORDS)}")
    print(f"{'='*50}\n")

    required_env = ["DEEPSEEK_API_KEY", "TAVILY_API_KEY", "WXPUSHER_APP_TOKEN", "WXPUSHER_UID"]
    missing = [e for e in required_env if not os.getenv(e)]
    if missing:
        print(f"❌ 缺少环境变量：{', '.join(missing)}")
        return

    try:
        agent = build_news_agent()
        final_state = agent.invoke({"keywords": KEYWORDS})
        print(f"\n{'='*50}")
        print("📰 今日简报预览：")
        print(final_state["report"])
        print(f"{'='*50}")
        print("✅ 每日新闻 Agent 执行完成")
    except Exception as e:
        print(f"❌ 任务执行失败：{e}")
        raise

if __name__ == "__main__":
    run_daily_news()
