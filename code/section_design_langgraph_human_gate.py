import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import section_design_agent as section_agent
from section_design_agent_prompt import (
    STEP8_FRAMEWORK,
    build_aggregation_prompt,
    build_evidence_model_prompt,
    build_field_planning_prompt,
    build_figure_classification_prompt,
    build_locating_prompt,
    build_mechanism_requirement_prompt,
    build_query_semantics_prompt,
    build_schema_design_prompt,
    build_section_partition_prompt,
    build_shared_context_block,
    build_specialization_critic_prompt,
    build_supervisor_prompt,
    build_subjective_supervisor_prompt,
    build_topic_adaptation_prompt,
)


class SectionDesignGraphState(TypedDict, total=False):
    args: dict[str, Any]
    shared_context: dict[str, Any]
    module_outputs: dict[str, Any]
    module_attempts: dict[str, Any]
    module_errors: dict[str, Any]
    validation_errors: list[str]
    human_advice: str
    result: dict[str, Any]
    status: str
    output: str
    current_node: str
    next_node: str


def namespace_to_dict(args):
    return vars(args).copy()


def dict_to_namespace(values):
    return argparse.Namespace(**values)


GRAPH_SEQUENCE = [
    "prepare",
    "locating",
    "mechanism",
    "query_semantics",
    "evidence_model",
    "subjective_supervisor",
    "topic_adaptation",
    "section_partition",
    "field_planning",
    "human_advice_gate",
    "supervisor",
    "figure_classification",
    "schema_design",
    "specialization_critic",
    "aggregation",
    "write_output",
]


def state_path_from_args(args):
    output_path = Path(args.output)
    return output_path.with_suffix(output_path.suffix + ".state.json")


def human_gate_context_path_from_args(args):
    output_path = Path(args.output)
    return output_path.with_suffix(output_path.suffix + ".human_gate_context.json")


def json_safe_state(state):
    return json.loads(json.dumps(state, ensure_ascii=False, default=str))


def write_state_snapshot(state, current_node, next_node=None):
    args = dict_to_namespace(state["args"])
    snapshot = deepcopy(state)
    snapshot["current_node"] = current_node
    if next_node:
        snapshot["next_node"] = next_node
    snapshot_path = state_path_from_args(args)
    snapshot_path.write_text(
        json.dumps(json_safe_state(snapshot), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return snapshot


def load_state_snapshot(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_human_gate_context(state):
    args = dict_to_namespace(state["args"])
    outputs = state.get("module_outputs", {})
    shared = state.get("shared_context", {})
    context = {
        "status": "waiting_for_human_advice",
        "gate": "before_supervisor_module",
        "next_node_after_advice": "supervisor",
        "state_path": str(state_path_from_args(args)),
        "reference_papers": args.reference_papers,
        "reference_paper_context_preview": shared.get("reference_paper_context", "")[:4000],
        "available_context": {
            "section_partition_module": outputs.get("section_partition_module"),
            "field_planning_module": outputs.get("field_planning_module"),
        },
        "required_advice": (
            "Provide expert suggestions about figure ownership risk, sections "
            "that must be reviewed by the supervisor, and fields that should or "
            "should not use figure evidence."
        ),
    }
    context_path = human_gate_context_path_from_args(args)
    context_path.write_text(
        json.dumps(json_safe_state(context), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(context_path), context


def merge_update(state, update):
    merged = deepcopy(state)
    merged.update(update or {})
    return merged


def interactive_resume_node(state):
    next_node = state.get("next_node") or "prepare"
    if next_node in {"write_output", "__end__"}:
        return "write_output"
    return next_node


def resume_router_node(state):
    return state


def prepare_state(state):
    if state.get("shared_context"):
        return write_state_snapshot(state, "prepare", "locating")

    args = dict_to_namespace(state["args"])
    query_requirements = section_agent.load_query_requirements(args.query_requirements)
    key_description_text = section_agent.load_key_description_text(args.key_description_path)
    reference_paper_context = section_agent.load_reference_paper_context(args.reference_papers)
    human_advice = args.human_advice or section_agent.load_optional_text(args.human_advice_path)
    shared_context = {
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "key_description_text": key_description_text,
        "reference_paper_context": reference_paper_context,
        "human_advice": human_advice,
        "shared_context_block": build_shared_context_block(
            args.database_goal,
            args.discipline,
            query_requirements,
            key_description_text,
            reference_paper_context,
        ),
    }
    update = {
        "shared_context": shared_context,
        "module_outputs": {},
        "module_attempts": {},
        "module_errors": {},
        "validation_errors": [],
        "human_advice": human_advice,
        "status": "running",
    }
    next_state = merge_update(state, update)
    return write_state_snapshot(next_state, "prepare", "locating")


def call_module_node(state, module_name, prompt, current_node, next_node):
    args = dict_to_namespace(state["args"])
    client = section_agent.get_client(
        base_url=args.base_url,
        api_key=args.api_key,
        backend=args.llm_backend,
    )
    result, errors, attempts = section_agent.call_module(
        client,
        args.model,
        module_name,
        prompt,
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    outputs = deepcopy(state.get("module_outputs", {}))
    module_attempts = deepcopy(state.get("module_attempts", {}))
    module_errors = deepcopy(state.get("module_errors", {}))
    outputs[module_name] = result
    module_attempts[module_name] = attempts
    module_errors[module_name] = errors
    update = {
        "module_outputs": outputs,
        "module_attempts": module_attempts,
        "module_errors": module_errors,
    }
    if errors:
        update["validation_errors"] = errors
        update["status"] = "needs_review"
        next_node = "write_output"
    return write_state_snapshot(merge_update(state, update), current_node, next_node)


def stop_if_errors(state, next_node):
    if state.get("validation_errors"):
        return "write_output"
    return next_node


def locating_node(state):
    args = dict_to_namespace(state["args"])
    shared = state["shared_context"]
    return call_module_node(
        state,
        "locating_module",
        build_locating_prompt(
            args.database_goal,
            args.discipline,
            shared["query_requirements"],
            shared["key_description_text"],
            shared.get("reference_paper_context", ""),
        ),
        "locating",
        "mechanism",
    )


def mechanism_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "mechanism_requirement_module",
        build_mechanism_requirement_prompt(state["shared_context"], outputs["locating_module"]),
        "mechanism",
        "query_semantics",
    )


def query_semantics_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "query_semantics_module",
        build_query_semantics_prompt(state["shared_context"], outputs["locating_module"]),
        "query_semantics",
        "evidence_model",
    )


def evidence_model_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "evidence_model_module",
        build_evidence_model_prompt(state["shared_context"], outputs["locating_module"]),
        "evidence_model",
        "subjective_supervisor",
    )


def subjective_supervisor_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "subjective_supervisor_module",
        build_subjective_supervisor_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
        ),
        "subjective_supervisor",
        "topic_adaptation",
    )


def topic_adaptation_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "topic_adaptation_module",
        build_topic_adaptation_prompt(state["shared_context"], outputs["locating_module"]),
        "topic_adaptation",
        "section_partition",
    )


def section_partition_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "section_partition_module",
        build_section_partition_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["topic_adaptation_module"],
        ),
        "section_partition",
        "field_planning",
    )


def field_planning_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "field_planning_module",
        build_field_planning_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["topic_adaptation_module"],
            outputs["section_partition_module"],
        ),
        "field_planning",
        "human_advice_gate",
    )


def human_advice_gate_node(state):
    human_advice = (state.get("human_advice") or "").strip()
    if not human_advice:
        context_path, context = write_human_gate_context(state)
        waiting_state = merge_update(
            state,
            {
                "status": "waiting_for_human_advice",
                "next_node": "human_advice_gate",
                "module_outputs": {
                    **state.get("module_outputs", {}),
                    "human_advice_gate": {
                        "status": "waiting_for_human_advice",
                        "context_path": context_path,
                    },
                },
            },
        )
        write_state_snapshot(waiting_state, "human_advice_gate", "human_advice_gate")
        outputs = state["module_outputs"]
        human_advice = interrupt(
            {
                "status": "waiting_for_human_advice",
                "gate": "before_supervisor_module",
                "context_path": context_path,
                "available_context": {
                    "section_partition_module": outputs.get("section_partition_module"),
                    "field_planning_module": outputs.get("field_planning_module"),
                },
                "required_advice": context["required_advice"],
            }
        )

    shared = deepcopy(state["shared_context"])
    shared["human_advice"] = str(human_advice).strip()
    outputs = deepcopy(state.get("module_outputs", {}))
    outputs["human_advice_gate"] = {
        "status": "human_advice_received",
        "position": "before_supervisor_module",
        "human_advice": shared["human_advice"],
    }
    update = {
        "human_advice": shared["human_advice"],
        "shared_context": shared,
        "module_outputs": outputs,
    }
    return write_state_snapshot(merge_update(state, update), "human_advice_gate", "supervisor")


def supervisor_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "supervisor_module",
        build_supervisor_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["topic_adaptation_module"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            human_advice=state.get("human_advice", ""),
        ),
        "supervisor",
        "figure_classification",
    )


def figure_classification_node(state):
    outputs = state["module_outputs"]
    update = call_module_node(
        state,
        "figure_classification_module",
        build_figure_classification_prompt(
            state["shared_context"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            outputs["supervisor_module"],
        ),
        "figure_classification",
        "schema_design",
    )
    figure_result = update["module_outputs"]["figure_classification_module"]
    supervisor_result = state["module_outputs"]["supervisor_module"]
    if not update.get("validation_errors") and (
        supervisor_result.get("enable_figure_classification")
        != figure_result.get("enable_figure_classification")
    ):
        update["validation_errors"] = [
            "Supervisor decision and figure classification branch are inconsistent"
        ]
        update["status"] = "needs_review"
        update["next_node"] = "write_output"
    return write_state_snapshot(update, "figure_classification", update.get("next_node", "schema_design"))


def schema_design_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "schema_design_module",
        build_schema_design_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["topic_adaptation_module"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            outputs["supervisor_module"],
            outputs["figure_classification_module"],
        ),
        "schema_design",
        "specialization_critic",
    )


def specialization_critic_node(state):
    outputs = state["module_outputs"]
    return call_module_node(
        state,
        "specialization_critic_module",
        build_specialization_critic_prompt(
            state["shared_context"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            outputs["schema_design_module"],
        ),
        "specialization_critic",
        "aggregation",
    )


def aggregation_node(state):
    outputs = state["module_outputs"]
    update = call_module_node(
        state,
        "aggregation",
        build_aggregation_prompt(
            state["shared_context"],
            outputs["locating_module"],
            outputs["mechanism_requirement_module"],
            outputs["query_semantics_module"],
            outputs["evidence_model_module"],
            outputs["subjective_supervisor_module"],
            outputs["topic_adaptation_module"],
            outputs["section_partition_module"],
            outputs["field_planning_module"],
            outputs["supervisor_module"],
            outputs["figure_classification_module"],
            outputs["schema_design_module"],
            outputs["specialization_critic_module"],
        ),
        "aggregation",
        "write_output",
    )
    if update.get("validation_errors"):
        return update

    final_result = section_agent.finalize_result(
        state["shared_context"],
        update["module_outputs"]["aggregation"],
    )
    validation_errors = section_agent.validate_result(final_result)
    validation_errors.extend(
        section_agent.validate_specialization(
            state["shared_context"],
            update["module_outputs"],
            final_result,
        )
    )
    update["result"] = final_result
    update["validation_errors"] = validation_errors
    update["status"] = "success" if not validation_errors else "needs_review"
    update["next_node"] = "write_output"
    return write_state_snapshot(update, "aggregation", "write_output")


def write_output_node(state):
    args = dict_to_namespace(state["args"])
    shared = state.get("shared_context", {})
    query_requirements = shared.get("query_requirements", [])
    output_inputs = {
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "key_description_path": str(Path(args.key_description_path)),
        "reference_papers": args.reference_papers,
        "human_advice_required_before_supervisor": True,
        "human_advice_provided": bool(state.get("human_advice")),
    }
    errors = state.get("validation_errors", [])
    result = state.get("result")
    output = {
        "step": "step8_section_design_agent_langgraph_human_gate",
        "framework": STEP8_FRAMEWORK,
        "inputs": output_inputs,
        "reference_paper_context_used": shared.get("reference_paper_context", ""),
        "agent_flow": section_agent.build_agent_flow_trace(
            output_inputs,
            state.get("module_outputs", {}),
            result,
            errors,
        ),
        "module_outputs": state.get("module_outputs", {}),
        "module_errors": state.get("module_errors", {}),
        "result": result,
        "validation_errors": errors,
        "status": state.get("status", "needs_review"),
        "attempts": {
            "modules": state.get("module_attempts", {}),
            "final_redo": [],
        },
    }
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    write_state_snapshot(merge_update(state, {"output": str(output_path)}), "write_output", "__end__")
    print(f"Saved result to {output_path}")
    print(f"Status: {output['status']}")
    return {"output": str(output_path), "status": output["status"]}


def build_human_gate_graph():
    graph = StateGraph(SectionDesignGraphState)
    graph.add_node("resume_router", resume_router_node)
    graph.add_node("prepare", prepare_state)
    graph.add_node("locating", locating_node)
    graph.add_node("mechanism", mechanism_node)
    graph.add_node("query_semantics", query_semantics_node)
    graph.add_node("evidence_model", evidence_model_node)
    graph.add_node("subjective_supervisor", subjective_supervisor_node)
    graph.add_node("topic_adaptation", topic_adaptation_node)
    graph.add_node("section_partition", section_partition_node)
    graph.add_node("field_planning", field_planning_node)
    graph.add_node("human_advice_gate", human_advice_gate_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("figure_classification", figure_classification_node)
    graph.add_node("schema_design", schema_design_node)
    graph.add_node("specialization_critic", specialization_critic_node)
    graph.add_node("aggregation", aggregation_node)
    graph.add_node("write_output", write_output_node)

    graph.add_edge(START, "resume_router")
    graph.add_conditional_edges("resume_router", interactive_resume_node)
    graph.add_edge("prepare", "locating")
    graph.add_conditional_edges("locating", lambda s: stop_if_errors(s, "mechanism"))
    graph.add_conditional_edges("mechanism", lambda s: stop_if_errors(s, "query_semantics"))
    graph.add_conditional_edges("query_semantics", lambda s: stop_if_errors(s, "evidence_model"))
    graph.add_conditional_edges("evidence_model", lambda s: stop_if_errors(s, "subjective_supervisor"))
    graph.add_conditional_edges("subjective_supervisor", lambda s: stop_if_errors(s, "topic_adaptation"))
    graph.add_conditional_edges("topic_adaptation", lambda s: stop_if_errors(s, "section_partition"))
    graph.add_conditional_edges("section_partition", lambda s: stop_if_errors(s, "field_planning"))
    graph.add_conditional_edges("field_planning", lambda s: stop_if_errors(s, "human_advice_gate"))
    graph.add_edge("human_advice_gate", "supervisor")
    graph.add_conditional_edges("supervisor", lambda s: stop_if_errors(s, "figure_classification"))
    graph.add_conditional_edges("figure_classification", lambda s: stop_if_errors(s, "schema_design"))
    graph.add_conditional_edges("schema_design", lambda s: stop_if_errors(s, "specialization_critic"))
    graph.add_conditional_edges("specialization_critic", lambda s: stop_if_errors(s, "aggregation"))
    graph.add_edge("aggregation", "write_output")
    graph.add_edge("write_output", END)
    return graph.compile(checkpointer=MemorySaver())


def build_parser():
    parser = section_agent.build_parser()
    for action in parser._actions:
        if action.dest in {
            "database_goal",
            "discipline",
            "query_requirements",
            "key_description_path",
        }:
            action.required = False
    parser.description = (
        "Run Step 8 Section Design through LangGraph. The graph pauses after "
        "field_planning_module and before supervisor_module until human advice "
        "is supplied."
    )
    parser.add_argument(
        "--thread-id",
        default="section-design-human-gate",
        help="LangGraph checkpoint thread id for this run.",
    )
    parser.add_argument(
        "--interactive-human-gate",
        action="store_true",
        help="Ask for human advice on stdin when the graph interrupts.",
    )
    parser.add_argument(
        "--resume-from-state",
        default="",
        help="Resume from a previously written *.state.json snapshot.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if not args.resume_from_state:
        missing = [
            name
            for name in [
                "database_goal",
                "discipline",
                "query_requirements",
                "key_description_path",
            ]
            if not getattr(args, name)
        ]
        if missing:
            raise SystemExit(
                "Missing required arguments for a new run: "
                + ", ".join(f"--{name.replace('_', '-')}" for name in missing)
            )

    app = build_human_gate_graph()
    config = {"configurable": {"thread_id": args.thread_id}}

    args_dict = namespace_to_dict(args)
    thread_id = args_dict.pop("thread_id")
    interactive = args_dict.pop("interactive_human_gate")
    resume_from_state = args_dict.pop("resume_from_state")

    if resume_from_state:
        initial_state = load_state_snapshot(resume_from_state)
        if args.human_advice or args.human_advice_path:
            human_advice = args.human_advice or section_agent.load_optional_text(args.human_advice_path)
            snapshot_args = deepcopy(initial_state.get("args", {}))
            snapshot_args["human_advice"] = human_advice
            snapshot_args["human_advice_path"] = ""
            shared = deepcopy(initial_state.get("shared_context", {}))
            shared["human_advice"] = human_advice
            initial_state["args"] = snapshot_args
            initial_state["shared_context"] = shared
            initial_state["human_advice"] = human_advice
            if initial_state.get("next_node") == "human_advice_gate":
                initial_state["next_node"] = "human_advice_gate"
    else:
        initial_state = {"args": args_dict, "next_node": "prepare"}

    result = app.invoke(initial_state, config=config)
    interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
    if interrupts and not interactive:
        print("Paused before supervisor_module. Human advice is required.")
        print(interrupts)
        return

    if interrupts and interactive:
        print("Paused before supervisor_module. Enter human advice, then press Enter:")
        advice = input("> ").strip()
        result = app.invoke(Command(resume=advice), config=config)

    if isinstance(result, dict):
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "output": result.get("output"),
                    "current_node": result.get("current_node"),
                    "next_node": result.get("next_node"),
                    "reference_papers": initial_state.get("args", {}).get("reference_papers", []),
                },
                ensure_ascii=False,
            )
        )
    else:
        print(result)
    if thread_id:
        print(f"Thread id: {thread_id}")


if __name__ == "__main__":
    main()
