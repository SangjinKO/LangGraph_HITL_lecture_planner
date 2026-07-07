# nodes.py — 노드 함수 + 라우팅 함수
import json
import re
from pathlib import Path
from datetime import datetime

from langgraph.types import Send, interrupt

from state import LecturePlanState
from utils import call_with_retry, tavily, log_node

TEMPLATE_FILE = "lecture_plan_template.md"
RESULTS_DIR   = Path("lecture_results")
RESULTS_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────

def _parse_json(text: str) -> dict:
    """LLM 응답에서 JSON 블록을 추출해 파싱한다."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    json_str = match.group(1) if match else text
    return json.loads(json_str.strip())

# ─────────────────────────────────────────
# Round 0 — validate
# ─────────────────────────────────────────

@log_node
def validate_node(state: LecturePlanState) -> LecturePlanState:
    """6개 필드 검증. 누락 시 즉시 예외 발생."""
    required = ["topic", "audience_level", "duration",
                "delivery_method", "platform_tools", "constraints"]
    missing = [k for k in required if not state.get(k, "").strip()]
    if missing:
        raise ValueError(f"누락된 필드: {', '.join(missing)}")
    return state

# ─────────────────────────────────────────
# Round 1 — plan + research (동적 병렬)
# ─────────────────────────────────────────

@log_node
def plan_node(state: LecturePlanState) -> LecturePlanState:
    """서브토픽 3~5개 분해."""
    prompt = f"""
너는 강의 기획 전문가다. 아래 정보를 바탕으로 강의를 구성하는 핵심 서브토픽을
3개에서 5개 사이로 뽑아라. 강의 내용이 단순하면 3개, 복잡하면 5개까지 늘린다.
각 서브토픽은 인터넷 검색 키워드로 바로 쓸 수 있는 한 문장으로 작성한다.
platform_tools와 constraints를 반드시 고려해 현실적인 서브토픽을 선정한다.

주제: {state['topic']}
대상자/수준: {state['audience_level']}
일수/시간: {state['duration']}
강의방식: {state['delivery_method']}
플랫폼/도구: {state['platform_tools']}
제약조건: {state['constraints']}

출력은 다음 JSON 형식으로만 응답한다 (다른 텍스트 포함 금지):
{{
  "subtopics": ["서브토픽1", "서브토픽2", "서브토픽3", "(필요 시 서브토픽4)", "(필요 시 서브토픽5)"]
}}
"""
    result = _parse_json(call_with_retry(prompt))
    return {**state, "subtopics": result["subtopics"], "research_results": []}


def fan_out_research(state: LecturePlanState) -> list[Send]:
    """서브토픽 개수만큼 research_node를 동적으로 병렬 생성."""
    return [
        Send("research_node", {**state, "current_subtopic": subtopic})
        for subtopic in state["subtopics"]
    ]


@log_node
def research_node(state: LecturePlanState) -> dict:
    """서브토픽 1개 검색 + 요약. fan_out_research에 의해 병렬 실행된다.
    반환값: 변경된 필드만 반환 (병렬 실행 시 {**state} 전체 반환 금지).
    research_results는 Annotated[list, operator.add]라서 각 노드가 [result]를 반환하면
    LangGraph가 자동으로 합산한다."""
    subtopic = state["current_subtopic"]

    queries = [
        f"{subtopic} 강의",
        f"{subtopic} 교육 커리큘럼",
        f"{subtopic} 부트캠프",
    ]

    search_results = tavily.search(
        query=queries[0],
        max_results=5,
        exclude_domains=["wikipedia.org", "namu.wiki"],
    )
    contents = "\n\n".join([
        f"제목: {r['title']}\n내용: {r['content']}"
        for r in search_results.get("results", [])
    ])

    prompt = f"""
아래는 "{subtopic}"에 대한 검색 결과다.
강의계획서 작성에 직접 활용할 수 있는 내용만 5~8문장으로 요약해라.
- 어떤 개념/실습을 어떤 순서로 가르치는가
- 어떤 툴/환경을 사용하는가
- 수강 후 할 수 있는 것(학습 결과)은 무엇인가

검색 결과:
{contents}

출력은 다음 JSON 형식으로만 응답한다 (다른 텍스트 포함 금지):
{{
  "subtopic": "{subtopic}",
  "search_queries": {json.dumps(queries[:2], ensure_ascii=False)},
  "summary": "요약 내용"
}}
"""
    result = _parse_json(call_with_retry(prompt))
    # 변경분만 반환 — research_results 하나만
    return {"research_results": [result]}

# ─────────────────────────────────────────
# Round 2 — write
# ─────────────────────────────────────────

@log_node
def write_node(state: LecturePlanState) -> LecturePlanState:
    """초안 작성 (최초) 또는 피드백 반영 재작업."""
    template = ""
    if Path(TEMPLATE_FILE).exists():
        template = Path(TEMPLATE_FILE).read_text(encoding="utf-8")

    if state.get("human_decision") == "rework":
        # 재작업: 직전 피드백 항목만 수정
        prompt = f"""
너는 강의계획서 작성 전문가다.
아래 피드백에서 지적된 항목만 정확히 수정하고 나머지는 그대로 유지해라.

직전 피드백:
{state['review_feedback']}

현재 초안:
{state['draft']}

출력은 완성된 마크다운 본문 전체만 출력한다 (설명 문구 없이).
"""
    else:
        # 최초 작성
        research_summaries = "\n\n".join([
            f"[{r['subtopic']}]\n{r['summary']}"
            for r in state.get("research_results", [])
        ])
        prompt = f"""
너는 강의계획서 작성 전문가다.
아래 서식과 정보를 바탕으로 완성된 강의계획서를 작성해라.

[서식]
{template}

[입력 정보]
주제: {state['topic']}
대상자/수준: {state['audience_level']}
일수/시간: {state['duration']}
강의방식: {state['delivery_method']}
플랫폼/도구: {state['platform_tools']}
제약조건: {state['constraints']}

[리서치 요약]
{research_summaries}

시간 배치 규칙 (반드시 준수):
- 1일 = 8시간(480분) 기준
- 점심시간 60분 포함
- 오전/오후 각 10분 휴식 최소 1회씩 (하루 최소 2회)
- 각 세션은 90분 이하
- 영어 약어는 최초 등장 시 전체 단어 병기 (예: AI(Artificial Intelligence))
- platform_tools에 명시된 도구를 세션 내용에 반영
- constraints를 반드시 반영

출력은 완성된 마크다운 본문 전체만 출력한다 (설명 문구 없이).
"""
    draft = call_with_retry(prompt)
    return {**state, "draft": draft, "reviews": []}

# ─────────────────────────────────────────
# Round 3 — review (고정 병렬, 3개 동시)
# ─────────────────────────────────────────

@log_node
def review_content_node(state: LecturePlanState) -> LecturePlanState:
    """내용 충실도 검토."""
    prompt = f"""
너는 강의 내용 검토 전문가다. 아래 강의계획서 초안을 검토해라.
학습 목표와 커리큘럼/세션 내용이 논리적으로 연결되는지,
내용 누락이나 모순이 없는지, 제약조건이 반영됐는지만 평가한다
(시간 배치나 난이도는 평가하지 않는다).

제약조건: {state.get('constraints', '없음')}

초안:
{state['draft']}

출력은 다음 JSON 형식으로만 응답한다 (다른 텍스트 포함 금지):
{{
  "category": "content",
  "score": 0에서 100 사이 정수,
  "feedback": "구체적인 문제점. 문제 없으면 문제 없음"
}}
"""
    result = _parse_json(call_with_retry(prompt))
    return {"reviews": [result]}


@log_node
def review_time_node(state: LecturePlanState) -> dict:
    """시간 배치 규칙 검토."""
    prompt = f"""
너는 시간 배분 검토 전문가다. 아래 강의계획서 초안에서 시간 배치 규칙만 확인한다.

체크리스트:
- 일자별 세션 합계 + 점심 60분 + 휴식 20분 이상 = 480분인가?
- 모든 세션이 90분 이하인가?
- 오전/오후 각 최소 1회, 하루 최소 2회 휴식이 있는가?
- 점심시간이 60분으로 명시되어 있는가?

초안:
{state['draft']}

출력은 다음 JSON 형식으로만 응답한다 (다른 텍스트 포함 금지):
{{
  "category": "time_allocation",
  "score": 0에서 100 사이 정수,
  "feedback": "위반 항목을 구체적으로. 문제 없으면 문제 없음"
}}
"""
    result = _parse_json(call_with_retry(prompt))
    return {"reviews": [result]}


@log_node
def review_difficulty_node(state: LecturePlanState) -> dict:
    """난이도 적합성 검토."""
    prompt = f"""
너는 난이도 적합성 검토 전문가다.
아래 강의계획서 초안이 대상자 수준에 적합한지만 평가한다.

대상자/수준: {state.get('audience_level', '미지정')}

초안:
{state['draft']}

출력은 다음 JSON 형식으로만 응답한다 (다른 텍스트 포함 금지):
{{
  "category": "difficulty",
  "score": 0에서 100 사이 정수,
  "feedback": "구체적인 문제점. 문제 없으면 문제 없음"
}}
"""
    result = _parse_json(call_with_retry(prompt))
    return {"reviews": [result]}

# ─────────────────────────────────────────
# Round 4 — review_final + HITL
# ─────────────────────────────────────────

@log_node
def review_final_node(state: LecturePlanState) -> LecturePlanState:
    """reviewer 3개 결과 종합 (모드1) 또는 재작업 반영 확인 (모드2)."""
    if state.get("human_decision") != "rework":
        # 모드 1: 최초 검토 종합
        reviews = state["reviews"]
        scores  = [r["score"] for r in reviews]
        avg     = int(sum(scores) / len(scores))
        passed  = avg >= 80

        failed_feedbacks = [
            f"[{r['category']}] {r['feedback']}"
            for r in reviews
            if r["score"] < 80 and r["feedback"] != "문제 없음"
        ]
        feedback = "\n".join(failed_feedbacks) if failed_feedbacks else "문제 없음"

        return {**state, "review_score": avg, "review_pass": passed,
                "review_feedback": feedback}
    else:
        # 모드 2: 재검토 — 피드백 반영 여부만 확인
        prompt = f"""
아래 피드백의 각 지적 사항이 새 초안에 실제로 반영됐는지만 확인해라.
모두 반영됐으면 pass=true, 하나라도 안 됐으면 pass=false.

직전 피드백:
{state['review_feedback']}

새 초안:
{state['draft']}

출력은 다음 JSON 형식으로만 응답한다 (다른 텍스트 포함 금지):
{{
  "pass": true 또는 false,
  "feedback": "반영 안 된 항목. 모두 반영됐으면 모두 반영됨"
}}
"""
        result = _parse_json(call_with_retry(prompt))
        return {**state, "review_pass": result["pass"],
                "review_feedback": result["feedback"]}


@log_node
def hitl_node(state: LecturePlanState) -> LecturePlanState:
    """HITL 게이트 — interrupt()로 실행 정지, 사람 결정을 기다린다."""
    decision = interrupt({
        "message": "검토 점수가 기준 미달입니다. 어떻게 할까요?",
        "score":    state["review_score"],
        "feedback": state["review_feedback"],
        "options":  ["rework", "save"],
    })
    return {**state, "human_decision": decision}


@log_node
def save_node(state: LecturePlanState) -> LecturePlanState:
    """최종 강의계획서 + pipeline_log 저장."""
    now = datetime.now()

    # 파일 1: 최종 강의계획서
    final_path = RESULTS_DIR / "lecture_plan_final.md"
    final_path.write_text(state["draft"], encoding="utf-8")

    # 파일 2: pipeline_log
    log_ts   = now.strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"pipeline_log_{log_ts}.md"
    log_content = f"""=== 강의계획서 생성 실행 로그 ===
실행일시: {now.strftime("%Y-%m-%d %H:%M:%S")}

[입력값]
- topic:           {state['topic']}
- audience_level:  {state['audience_level']}
- duration:        {state['duration']}
- delivery_method: {state['delivery_method']}
- platform_tools:  {state['platform_tools']}
- constraints:     {state['constraints']}

[최종 결과]
- 최종 점수:     {state['review_score']}점
- pass 여부:     {state['review_pass']}
- rework 횟수:   {state['rework_count']}회
- HITL 발생:     {state['human_decision'] != ''}
- 사람 결정:     {state['human_decision'] or 'none'}

[생성 파일]
- {final_path}
- {log_path}
================================
"""
    log_path.write_text(log_content, encoding="utf-8")

    return {**state, "save_path": str(final_path), "log_path": str(log_path)}

# ─────────────────────────────────────────
# 라우팅 함수
# ─────────────────────────────────────────

def route_after_review(state: LecturePlanState) -> str:
    """review_final 이후 — 자동 승인 or HITL 분기."""
    return "save" if state["review_pass"] else "hitl"


def route_after_hitl(state: LecturePlanState) -> str:
    """hitl_node 이후 — 재작업 or 저장 분기."""
    return "write" if state["human_decision"] == "rework" else "save"