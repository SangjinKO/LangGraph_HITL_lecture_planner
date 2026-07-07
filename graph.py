# graph.py — LangGraph 그래프 구성 + compile
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from state import LecturePlanState
from nodes import (
    validate_node, plan_node, fan_out_research, research_node,
    write_node,
    review_content_node, review_time_node, review_difficulty_node,
    review_final_node, hitl_node, save_node,
    route_after_review, route_after_hitl,
)

workflow = StateGraph(LecturePlanState)

# 노드 등록
workflow.add_node("validate",           validate_node)
workflow.add_node("plan",               plan_node)
workflow.add_node("research_node",      research_node)
workflow.add_node("write",              write_node)
workflow.add_node("review_content",     review_content_node)
workflow.add_node("review_time",        review_time_node)
workflow.add_node("review_difficulty",  review_difficulty_node)
workflow.add_node("review_final",       review_final_node)
workflow.add_node("hitl",               hitl_node)
workflow.add_node("save",               save_node)

# 엣지 연결
workflow.set_entry_point("validate")
workflow.add_edge("validate", "plan")

# Round 1 병렬: Send API (n8n의 Split Out + 순차 Research와 달리 진짜 동시 실행)
workflow.add_conditional_edges("plan", fan_out_research, ["research_node"])
workflow.add_edge("research_node", "write")

# Round 3 병렬: add_edge 다중 분기 (n8n의 Review-Content/Time/Difficulty 3개 분기 대응)
workflow.add_edge("write", "review_content")
workflow.add_edge("write", "review_time")
workflow.add_edge("write", "review_difficulty")
workflow.add_edge("review_content",    "review_final")
workflow.add_edge("review_time",       "review_final")
workflow.add_edge("review_difficulty", "review_final")

# 자동 승인 vs HITL 분기
workflow.add_conditional_edges(
    "review_final",
    route_after_review,
    {"save": "save", "hitl": "hitl"},
)

# HITL 이후 분기
workflow.add_conditional_edges(
    "hitl",
    route_after_hitl,
    {"write": "write", "save": "save"},
)

workflow.add_edge("save", END)

# MemorySaver: thread_id별 상태 저장 (HITL Resume에 필수)
checkpointer = MemorySaver()
pipeline = workflow.compile(checkpointer=checkpointer)