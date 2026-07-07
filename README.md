# LangGraph_HITL_lecture_planner
> Multi-Agent AI Workflow for Lecture Plan Generation — LangGraph · FastAPI · Gemini · Tavily · HITL

---

## Why

강의계획서 작성은 강사에게 반복적이고 판단이 많이 필요한 작업이다. 주제 조사, 커리큘럼 설계, 시간 배분 검토, 난이도 검증이 정해진 순서로 이어지는 이 과정을 멀티에이전트 파이프라인으로 자동화했다.

단순 자동화가 아니라 **품질 미달 시 사람이 개입하는 HITL(Human-in-the-Loop) 구조**를 추가했다. AI 점수가 기준 미달이면 자동 재작업 대신 `interrupt()`로 그래프를 정지시키고, 사람이 판단한 후 `/resume`으로 이어서 실행한다.

> 같은 시나리오의 n8n 구현과 비교하면 "n8n에서 순차로 처리하던 researcher가 LangGraph에서 Send API로 진짜 병렬이 된다", "n8n Wait 노드가 interrupt()로 코드화된다"는 차이가 명확하게 보인다.

---

## Architecture

```
입력 (6개 필드: 주제/대상자/일수/방식/도구/제약조건)
      ↓
validate → plan
      ↓
research_node × N (Send API, 동적 병렬)     ← n8n은 순차
      ↓
write
      ↓
review_content ┐
review_time    ├ add_edge 다중 분기 (3개 동시)
review_difficulty┘
      ↓
review_final → score >= 80 → save
           └→ score <  80 → interrupt()
                                ↓
                         사람 결정 (rework / save)
                                ↓
                         write → review_final → save
```

---

## Key Design Decisions

**Send API — 동적 N개 진짜 병렬**

서브토픽 수만큼 `research_node`를 동시에 실행한다. n8n에서는 Split Out + 단일 노드 순차 처리만 가능했다.

```python
def fan_out_research(state):
    return [
        Send("research_node", {**state, "current_subtopic": subtopic})
        for subtopic in state["subtopics"]
    ]
```

**Annotated reducer — 병렬 노드의 리스트 업데이트**

병렬 실행되는 노드에서 `{**state, ...}` 전체를 반환하면 `InvalidUpdateError`가 발생한다. 같은 키에 여러 노드가 동시에 쓰려 하기 때문이다.

```python
# state.py — reducer 선언
research_results: Annotated[list[dict], operator.add]
reviews:          Annotated[list[dict], operator.add]

# 병렬 노드 — 변경분만 반환
return {"research_results": [result]}
```

**interrupt() — HITL을 코드로 선언**

```python
def hitl_node(state):
    decision = interrupt({
        "score":    state["review_score"],
        "feedback": state["review_feedback"],
    })
    return {**state, "human_decision": decision}
```

`MemorySaver`가 `thread_id`별 상태를 저장해서 재개(Resume)가 가능하다. HITL 발생 시 서버 터미널에 바로 복사할 수 있는 curl 명령어를 자동 출력한다.

**@log_node 데코레이터 — 타임스탬프 포함 노드 로깅**

```python
def log_node(func):
    @wraps(func)
    def wrapper(state):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] [▶] {func.__name__}")
        ...
```

병렬 노드의 시작 시각이 같게 찍히면서 진짜 병렬임을 터미널에서 바로 확인할 수 있다.

---

## Tech Stack

| 항목 | 내용 |
|---|---|
| LLM | Gemini 3.1 Flash Lite (무료 API, 폴백: gemini-2.5-flash-lite → gemini-2.5-flash) |
| 웹 검색 | Tavily Search API |
| Orchestration | LangGraph + FastAPI |
| HITL | `interrupt()` + `MemorySaver` |
| 로깅 | `@log_node` 데코레이터 → `subagent_log.csv` |

---

## Repository Structure

```
multi-agent-lecture-planner-langgraph/
├── state.py          # LecturePlanState TypedDict + Annotated reducer
├── utils.py          # LLM/Tavily 설정, @log_node, call_with_retry
├── nodes.py          # 모든 노드 함수 + 라우팅 함수
├── graph.py          # StateGraph 구성 + pipeline export
├── main.py           # FastAPI /generate + /resume
```

---

## How to Run

```bash
pip install langchain langchain-google-genai langgraph fastapi uvicorn \
            python-dotenv tavily-python
cp .env.example .env  # GEMINI_API_KEY, TAVILY_API_KEY 입력

# 단독 실행 (터미널)
python3 analyze.py \
  --topic "업무에 바로 쓰는 AI 활용법" \
  --audience_level "비개발자 직장인 초급" \
  --duration "1일 8시간" \
  --delivery_method "강의+실습 병행" \
  --platform_tools "ChatGPT, Notion" \
  --constraints "프로젝터 없음, 노트북 1인 1대"

# FastAPI 서버
uvicorn main:app --port 3000 --reload

curl -X POST http://localhost:3000/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "...", "audience_level": "...", "duration": "...",
       "delivery_method": "...", "platform_tools": "...", "constraints": "..."}'
# HITL 발생 시 터미널에 curl 명령어 자동 출력
```

---

## Known Issues & Lessons

**`InvalidUpdateError: At key 'topic'`**

병렬 노드가 `{**state, ...}` 전체를 반환하면 공통 키에 동시 쓰기 충돌이 발생한다. `Annotated[list, operator.add]` + 변경분만 반환으로 해결. `research_results`와 `reviews` 두 필드 모두 적용 필요.

**`'list' object has no attribute 'strip'`**

`ChatGoogleGenerativeAI`가 멀티파트 응답을 보낼 때 `.content`가 리스트로 온다. `isinstance(content, list)` 분기 처리 필요.

**`/resume`에서 validate_node 재실행**

잘못된 `thread_id`를 사용하면 MemorySaver에 상태가 없어서 그래프를 처음부터 실행하려 한다. `/resume`에 `pipeline.get_state(config).next` 확인 로직을 추가해 명확한 에러를 반환한다.