# utils.py — 공통 유틸리티
import os
import csv
import time
from datetime import datetime
from pathlib import Path
from functools import wraps

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from tavily import TavilyClient

load_dotenv()

# ─────────────────────────────────────────
# LLM 설정 — 모델 폴백 체인
# ─────────────────────────────────────────

MODELS = [
    "gemini-3.1-flash-lite",   # 무료 토큰 가장 많음
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

def _build_llm(model: str) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0,
    )

def call_with_retry(prompt: str, max_retries: int = 3) -> str:
    """모델 폴백 체인으로 Gemini를 호출한다. 429/503 에러 시 다음 모델로 전환."""
    for model in MODELS:
        llm = _build_llm(model)
        for attempt in range(max_retries):
            try:
                result = llm.invoke(prompt)
                # content가 list로 올 때 처리 (Gemini 멀티파트 응답)
                content = result.content
                if isinstance(content, list):
                    content = "".join(
                        part if isinstance(part, str) else part.get("text", "")
                        for part in content
                    )
                return content.strip()
            except Exception as e:
                err = str(e)
                if "429" in err or "503" in err:
                    wait = 2 ** attempt
                    print(f"[{model}] {err} — {wait}s 대기 후 재시도")
                    time.sleep(wait)
                else:
                    print(f"[{model}] 오류: {err}")
                    break  # 다음 모델로
    raise RuntimeError("모든 모델에서 호출 실패")

# ─────────────────────────────────────────
# Tavily 설정
# ─────────────────────────────────────────

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ─────────────────────────────────────────
# 로깅 — log_node 데코레이터
# ─────────────────────────────────────────

LOG_FILE = "subagent_log.csv"

def _write_log(name: str, event: str, duration: str) -> None:
    file_exists = Path(LOG_FILE).exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "event", "subagent_name", "duration_sec"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            event,
            name,
            duration,
        ])

def log_node(func):
    """노드 함수를 감싸서 시작/종료를 subagent_log.csv에 자동 기록하고 터미널에 출력한다."""
    @wraps(func)
    def wrapper(state):
        name = func.__name__

        # 터미널 출력 — 노드명 + 간단한 컨텍스트
        context = ""
        if name == "research_node":
            context = f" → '{state.get('current_subtopic', '')}'"
        elif name == "write_node":
            mode = state.get("human_decision", "")
            context = " (재작업)" if mode == "rework" else " (최초 작성)"
        elif name == "review_final_node":
            mode = state.get("human_decision", "")
            context = " (재검토)" if mode == "rework" else " (최초 종합)"

        print(f"[{datetime.now().strftime('%H:%M:%S')}] [▶] {name}{context}")
        start = time.time()
        _write_log(name, "START", "")
        try:
            result = func(state)
            duration = time.time() - start
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [✓] {name}{context} ({duration:.1f}s)")
            return result
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] [✗] {name} 실패: {e}")
            raise
        finally:
            _write_log(name, "END", f"{time.time() - start:.1f}s")
    return wrapper