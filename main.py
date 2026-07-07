# main.py — FastAPI 서버 + /generate + /resume
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.types import Command

from graph import pipeline

app = FastAPI(title="Multi-Agent Lecture Planner")

# ─────────────────────────────────────────
# 요청/응답 모델
# ─────────────────────────────────────────

class LectureRequest(BaseModel):
    topic:           str
    audience_level:  str
    duration:        str
    delivery_method: str
    platform_tools:  str
    constraints:     str

class ResumeRequest(BaseModel):
    thread_id: str
    decision:  str   # "rework" | "save"

# ─────────────────────────────────────────
# /generate — 강의계획서 생성 시작
# ─────────────────────────────────────────

@app.post("/generate")
async def generate(req: LectureRequest):
    thread_id = str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        **req.model_dump(),
        "subtopics": [], "current_subtopic": "",
        "research_results": [],
        "draft": "", "reviews": [],
        "review_score": 0, "review_feedback": "", "review_pass": False,
        "human_decision": "", "rework_count": 0,
        "save_path": "", "log_path": "",
    }

    pipeline.invoke(initial_state, config)

    # interrupt() 발생 여부 확인
    snapshot = pipeline.get_state(config)
    if snapshot.next:   # 다음 실행 노드가 있으면 = interrupt()에서 멈춘 상태
        interrupt_val = snapshot.tasks[0].interrupts[0].value

        # 터미널에 HITL 안내 + 복사용 curl 명령어 출력
        print(f"\n{'='*60}")
        print(f"[HITL] 검토 점수: {interrupt_val['score']}점 | 통과: False")
        print(f"\n피드백:\n{interrupt_val['feedback']}")
        print(f"\n{'─'*60}")
        print(f"바로 복사해서 사용하세요:\n")
        print(f"재작업:")
        print(f"curl -X POST http://localhost:3000/resume \\")
        print(f'  -H "Content-Type: application/json" \\')
        print(f'  -d \'{{"thread_id": "{thread_id}", "decision": "rework"}}\'')
        print(f"\n현재 버전 저장:")
        print(f"curl -X POST http://localhost:3000/resume \\")
        print(f'  -H "Content-Type: application/json" \\')
        print(f'  -d \'{{"thread_id": "{thread_id}", "decision": "save"}}\'')
        print(f"{'='*60}\n")

        return {
            "status":    "waiting_for_human",
            "thread_id": thread_id,
            "score":     interrupt_val["score"],
            "feedback":  interrupt_val["feedback"],
            "message":   interrupt_val["message"],
        }

    # 자동 완료 (score >= 80)
    final_state = snapshot.values
    return {
        "status":    "completed",
        "save_path": final_state.get("save_path", ""),
        "log_path":  final_state.get("log_path", ""),
    }

# ─────────────────────────────────────────
# /resume — 사람 결정 전달 + 재개
# ─────────────────────────────────────────

@app.post("/resume")
async def resume(req: ResumeRequest):
    if req.decision not in ("rework", "save"):
        raise HTTPException(status_code=400, detail="decision은 'rework' 또는 'save'여야 합니다.")

    config = {"configurable": {"thread_id": req.thread_id}}

    # thread_id 유효성 확인 — 대기 중인 상태가 없으면 에러 반환
    snapshot = pipeline.get_state(config)
    if not snapshot.next:
        raise HTTPException(
            status_code=404,
            detail=f"thread_id '{req.thread_id}'에 대기 중인 실행이 없습니다. "
                   f"/generate 응답에서 받은 thread_id를 사용하세요."
        )

    pipeline.invoke(Command(resume=req.decision), config)

    snapshot    = pipeline.get_state(config)
    final_state = snapshot.values
    return {
        "status":    "completed",
        "save_path": final_state.get("save_path", ""),
        "log_path":  final_state.get("log_path", ""),
    }