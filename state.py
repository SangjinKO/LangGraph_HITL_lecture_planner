# state.py — LecturePlanState 정의
from typing import TypedDict, Annotated
import operator


class LecturePlanState(TypedDict):
    # 입력 (6개 필드)
    topic: str
    audience_level: str
    duration: str
    delivery_method: str
    platform_tools: str
    constraints: str

    # planner 결과
    subtopics: list[str]
    current_subtopic: str          # Send API에서 개별 researcher에 전달

    # researcher 결과
    # Annotated[list, operator.add]: 병렬 실행되는 research_node들이 각자 결과를 추가할 때
    # LangGraph가 자동으로 합산해준다. {**state, ...} 패턴 대신 이 방식을 써야 충돌이 없다.
    research_results: Annotated[list[dict], operator.add]

    # writer 결과
    draft: str

    # reviewer 결과
    # research_results와 동일하게 Annotated 처리 필요
    # review_content/time/difficulty 3개가 병렬 실행되면서 각자 reviews에 추가하기 때문
    reviews: Annotated[list[dict], operator.add]
    review_score: int
    review_feedback: str
    review_pass: bool

    # HITL
    human_decision: str            # "rework" | "save" | "" (초기값)
    rework_count: int

    # 산출물
    save_path: str
    log_path: str