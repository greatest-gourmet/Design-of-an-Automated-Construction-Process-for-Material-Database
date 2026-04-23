import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from section_design_agent_prompt import (
    FINAL_OUTPUT_SCHEMA_DESCRIPTION,
    STEP8_FRAMEWORK,
    build_aggregation_prompt,
    build_evidence_model_prompt,
    build_field_planning_prompt,
    build_figure_classification_prompt,
    build_locating_prompt,
    build_mechanism_requirement_prompt,
    build_module_redo_prompt,
    build_query_semantics_prompt,
    build_redo_prompt,
    build_schema_design_prompt,
    build_section_partition_prompt,
    build_shared_context_block,
    build_specialization_critic_prompt,
    build_supervisor_prompt,
    build_subjective_supervisor_prompt,
    build_topic_adaptation_prompt,
)


DEFAULT_SYSTEM_PROMPT = "You are an expert in scientific database schema design."
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"
DEFAULT_LLM_BACKEND = "openai"

MODULE_SCHEMAS = {
    "locating_module": {
        "required_keys": [
            "database_goal",
            "discipline",
            "query_requirements",
            "retrieval_unit",
            "organization_focus",
            "design_rationale",
        ],
        "schema_text": """
        {
          "database_goal": "string",
          "discipline": "string",
          "query_requirements": ["string"],
          "retrieval_unit": "string",
          "organization_focus": "string",
          "design_rationale": "string"
        }
        """.strip(),
    },
    "mechanism_requirement_module": {
        "required_keys": [
            "domain_focus",
            "must_have_concepts",
            "recommended_objects",
            "red_flag_patterns",
        ],
        "schema_text": """
        {
          "domain_focus": "string",
          "must_have_concepts": ["string"],
          "recommended_objects": ["string"],
          "red_flag_patterns": ["string"]
        }
        """.strip(),
    },
    "query_semantics_module": {
        "required_keys": ["query_objects"],
        "schema_text": """
        {
          "query_objects": [
            {
              "query_requirement": "string",
              "object_type": "string",
              "recommended_field_groups": ["string"],
              "comparison_axes": ["string"],
              "anti_generic_warning": "string"
            }
          ]
        }
        """.strip(),
    },
    "evidence_model_module": {
        "required_keys": [
            "evidence_layers",
            "figure_ownership_principles",
            "anti_generic_evidence_patterns",
        ],
        "schema_text": """
        {
          "evidence_layers": [
            {
              "layer_name": "string",
              "description": "string",
              "typical_methods": ["string"],
              "schema_implication": "string"
            }
          ],
          "figure_ownership_principles": ["string"],
          "anti_generic_evidence_patterns": ["string"]
        }
        """.strip(),
    },
    "subjective_supervisor_module": {
        "required_keys": [
            "database_nature",
            "modeling_position",
            "must_have_concepts",
            "must_not_become",
            "red_flags",
            "approved_section_strategy",
            "redo_directives",
        ],
        "schema_text": """
        {
          "database_nature": "string",
          "modeling_position": "string",
          "must_have_concepts": ["string"],
          "must_not_become": ["string"],
          "red_flags": ["string"],
          "approved_section_strategy": ["string"],
          "redo_directives": ["string"]
        }
        """.strip(),
    },
    "topic_adaptation_module": {
        "required_keys": [
            "topic_type",
            "adaptation_principles",
            "avoid_generic_template",
            "topic_specific_adjustments",
        ],
        "schema_text": """
        {
          "topic_type": "string",
          "adaptation_principles": ["string"],
          "avoid_generic_template": ["string"],
          "topic_specific_adjustments": ["string"]
        }
        """.strip(),
    },
    "section_partition_module": {
        "required_keys": ["core_sections", "non_core_sections"],
        "schema_text": """
        {
          "core_sections": [{"section_id": "section0"}],
          "non_core_sections": [{"section_id": "sectionX"}]
        }
        """.strip(),
    },
    "field_planning_module": {
        "required_keys": ["field_groups", "red_flag_fixes"],
        "schema_text": """
        {
          "field_groups": [
            {
              "section_id": "string",
              "group_name": "string",
              "purpose": "string",
              "recommended_fields": ["string"],
              "evidence_strategy": "string"
            }
          ],
          "red_flag_fixes": ["string"]
        }
        """.strip(),
    },
    "supervisor_module": {
        "required_keys": [
            "review_stage",
            "enable_figure_classification",
            "routing_decision",
            "decision_summary",
            "risk_signals",
            "risk_assessment",
            "trigger_sections",
            "expected_figure_fields",
            "skip_reason",
        ],
        "schema_text": """
        {
          "review_stage": "post_section_partition",
          "enable_figure_classification": true,
          "routing_decision": "figure_classification_path",
          "decision_summary": "string",
          "risk_signals": ["string"],
          "risk_assessment": "string",
          "trigger_sections": ["section4", "section5"],
          "expected_figure_fields": ["material_info.section4.R_T.figure"],
          "skip_reason": ""
        }
        """.strip(),
    },
    "figure_classification_module": {
        "required_keys": [
            "enable_figure_classification",
            "routing_status",
            "classification_strategy",
            "section_figure_plan",
            "skip_reason",
        ],
        "schema_text": """
        {
          "enable_figure_classification": true,
          "routing_status": "active",
          "classification_strategy": ["string"],
          "section_figure_plan": [
            {
              "section_id": "section4",
              "section_name": "string",
              "figure_scope": "string",
              "allowed_figure_categories": ["R_T"],
              "blocked_neighbor_sections": ["section5"],
              "blocked_figure_categories": ["band_structure"],
              "routing_rule": "string"
            }
          ],
          "skip_reason": ""
        }
        """.strip(),
    },
    "schema_design_module": {
        "required_keys": ["top_level_keys", "field_registry"],
        "schema_text": """
        {
          "top_level_keys": [{"key": "string", "description": "string"}],
          "field_registry": [
            {
              "field_path": "string",
              "section_id": "string",
              "field_name": "string",
              "data_type": "string",
              "required": true,
              "source_basis": ["text", "table", "figure"],
              "figure_constraint": {
                "uses_figure_classification": true,
                "allowed_sections": ["section4"],
                "allowed_figure_categories": ["R_T"],
                "why_needed": "string"
              },
              "reason": "string"
            }
          ]
        }
        """.strip(),
    },
    "specialization_critic_module": {
        "required_keys": [
            "specialization_status",
            "is_generic",
            "missing_concepts",
            "structural_weaknesses",
            "redo_needed",
            "redo_directives",
        ],
        "schema_text": """
        {
          "specialization_status": "pass",
          "is_generic": false,
          "missing_concepts": ["string"],
          "structural_weaknesses": ["string"],
          "redo_needed": false,
          "redo_directives": ["string"]
        }
        """.strip(),
    },
    "aggregation": {
        "required_keys": [
            "database_positioning",
            "section_design",
            "schema_definition",
            "quality_check",
        ],
        "schema_text": FINAL_OUTPUT_SCHEMA_DESCRIPTION,
    },
}


def get_client(base_url=None, api_key=None, backend=DEFAULT_LLM_BACKEND):
    if backend == "langchain":
        return get_langchain_client(base_url=base_url, api_key=api_key)
    if backend != "openai":
        raise ValueError(f"Unsupported LLM backend: {backend}")

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    return {"backend": "openai", "client": OpenAI(**kwargs)}


def get_langchain_client(base_url=None, api_key=None):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "LangChain backend requires the optional dependency `langchain-openai`. "
            "Install it with `pip install langchain-openai` or use `--llm-backend openai`."
        ) from exc

    return {
        "backend": "langchain",
        "chat_model_cls": ChatOpenAI,
        "base_url": base_url,
        "api_key": api_key,
    }


def build_langchain_chat_model(client, model, temperature):
    chat_model_cls = client["chat_model_cls"]
    common_kwargs = {
        "model": model,
        "temperature": temperature,
        "model_kwargs": {"response_format": {"type": "json_object"}},
    }
    if client.get("api_key"):
        common_kwargs["api_key"] = client["api_key"]
    if client.get("base_url"):
        common_kwargs["base_url"] = client["base_url"]

    try:
        return chat_model_cls(**common_kwargs)
    except TypeError:
        # Older langchain-openai versions used OpenAI-prefixed parameter names.
        fallback_kwargs = dict(common_kwargs)
        if "api_key" in fallback_kwargs:
            fallback_kwargs["openai_api_key"] = fallback_kwargs.pop("api_key")
        if "base_url" in fallback_kwargs:
            fallback_kwargs["openai_api_base"] = fallback_kwargs.pop("base_url")
        return chat_model_cls(**fallback_kwargs)


def chat(client, model, prompt, system_prompt=DEFAULT_SYSTEM_PROMPT, temperature=0):
    if isinstance(client, dict) and client.get("backend") == "langchain":
        chat_model = build_langchain_chat_model(client, model, temperature)
        response = chat_model.invoke(
            [
                ("system", system_prompt),
                ("user", prompt),
            ]
        )
        return response.content

    openai_client = client["client"] if isinstance(client, dict) else client
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def load_query_requirements(query_input):
    path = Path(query_input)
    if path.exists():
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return [item.strip() for item in query_input.split("||") if item.strip()]


def load_key_description_text(path):
    return Path(path).read_text(encoding="utf-8")


def load_optional_text(path):
    if not path:
        return ""
    return Path(path).read_text(encoding="utf-8").strip()


def _truncate_text(text, max_chars):
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars].rstrip() + "\n...[truncated]"


def load_reference_paper_context(reference_papers, max_chars_per_paper=12000):
    if not reference_papers:
        return ""

    paper_blocks = []
    for raw_path in reference_papers:
        path = Path(raw_path)
        if not path.exists():
            paper_blocks.append(f"[Missing paper] {raw_path}")
            continue
        text = path.read_text(encoding="utf-8")
        paper_blocks.append(
            f"Paper: {path.name}\n{_truncate_text(text, max_chars_per_paper)}"
        )
    return "\n\n".join(paper_blocks)


def parse_json_response(raw_response):
    try:
        return json.loads(raw_response), []
    except json.JSONDecodeError as exc:
        return None, [f"Invalid JSON response: {exc}"]


def normalize_module_result(module_name, result):
    if not isinstance(result, dict):
        return result

    if module_name == "specialization_critic_module":
        status = str(result.get("specialization_status", "")).strip().lower()
        status_aliases = {
            "pass": "pass",
            "ok": "pass",
            "success": "pass",
            "approved": "pass",
            "needs_redesign": "needs_redesign",
            "need_redesign": "needs_redesign",
            "redesign": "needs_redesign",
            "redo": "needs_redesign",
            "redo_needed": "needs_redesign",
            "fail": "needs_redesign",
            "failed": "needs_redesign",
            "reject": "needs_redesign",
            "rejected": "needs_redesign",
        }
        normalized_status = status_aliases.get(status)
        if normalized_status:
            result["specialization_status"] = normalized_status

        if result.get("specialization_status") == "pass":
            result["redo_needed"] = False
            if not isinstance(result.get("is_generic"), bool):
                result["is_generic"] = False
        elif result.get("specialization_status") == "needs_redesign":
            result["redo_needed"] = True
            if not isinstance(result.get("is_generic"), bool):
                result["is_generic"] = True

    return result


def validate_module_result(module_name, result):
    errors = []
    schema = MODULE_SCHEMAS[module_name]
    if not isinstance(result, dict):
        return [f"{module_name} output must be a JSON object"]
    for key in schema["required_keys"]:
        if key not in result:
            errors.append(f"{module_name} missing key: {key}")
    if module_name == "section_partition_module":
        if not isinstance(result.get("core_sections"), list) or not result.get("core_sections"):
            errors.append("section_partition_module.core_sections must be a non-empty list")
        if not isinstance(result.get("non_core_sections"), list):
            errors.append("section_partition_module.non_core_sections must be a list")
    if module_name == "mechanism_requirement_module":
        if not isinstance(result.get("must_have_concepts"), list) or not result.get("must_have_concepts"):
            errors.append("mechanism_requirement_module.must_have_concepts must be a non-empty list")
    if module_name == "query_semantics_module":
        if not isinstance(result.get("query_objects"), list) or not result.get("query_objects"):
            errors.append("query_semantics_module.query_objects must be a non-empty list")
    if module_name == "evidence_model_module":
        if not isinstance(result.get("evidence_layers"), list) or not result.get("evidence_layers"):
            errors.append("evidence_model_module.evidence_layers must be a non-empty list")
    if module_name == "subjective_supervisor_module":
        if not isinstance(result.get("must_have_concepts"), list) or not result.get("must_have_concepts"):
            errors.append("subjective_supervisor_module.must_have_concepts must be a non-empty list")
        if not isinstance(result.get("must_not_become"), list) or not result.get("must_not_become"):
            errors.append("subjective_supervisor_module.must_not_become must be a non-empty list")
    if module_name == "field_planning_module":
        if not isinstance(result.get("field_groups"), list) or not result.get("field_groups"):
            errors.append("field_planning_module.field_groups must be a non-empty list")
    if module_name == "supervisor_module":
        if not isinstance(result.get("enable_figure_classification"), bool):
            errors.append("supervisor_module.enable_figure_classification must be a boolean")
        if result.get("routing_decision") not in {"figure_classification_path", "direct_schema_path"}:
            errors.append(
                "supervisor_module.routing_decision must be figure_classification_path or direct_schema_path"
            )
    if module_name == "figure_classification_module":
        if not isinstance(result.get("enable_figure_classification"), bool):
            errors.append("figure_classification_module.enable_figure_classification must be a boolean")
        if result.get("routing_status") not in {"active", "skipped"}:
            errors.append("figure_classification_module.routing_status must be active or skipped")
        if not isinstance(result.get("classification_strategy"), list):
            errors.append("figure_classification_module.classification_strategy must be a list")
        if not isinstance(result.get("section_figure_plan"), list):
            errors.append("figure_classification_module.section_figure_plan must be a list")
        if result.get("enable_figure_classification") and result.get("routing_status") != "active":
            errors.append("figure_classification_module must be active when enable_figure_classification is true")
        if (not result.get("enable_figure_classification")) and result.get("routing_status") != "skipped":
            errors.append("figure_classification_module must be skipped when enable_figure_classification is false")
    if module_name == "schema_design_module":
        if not isinstance(result.get("field_registry"), list) or not result.get("field_registry"):
            errors.append("schema_design_module.field_registry must be a non-empty list")
    if module_name == "specialization_critic_module":
        if result.get("specialization_status") not in {"pass", "needs_redesign"}:
            errors.append("specialization_critic_module.specialization_status must be pass or needs_redesign")
        if not isinstance(result.get("is_generic"), bool):
            errors.append("specialization_critic_module.is_generic must be a boolean")
        if not isinstance(result.get("redo_needed"), bool):
            errors.append("specialization_critic_module.redo_needed must be a boolean")
    return errors


def summarize_field_paths(field_registry, owner_id, limit=6):
    matches = []
    for field in field_registry or []:
        if isinstance(field, dict) and field.get("section_id") == owner_id and field.get("field_path"):
            matches.append(field["field_path"])
    return matches[:limit]


def build_agent_flow_trace(inputs, modules, result, validation_errors):
    if not isinstance(result, dict):
        return {
            "user_input": inputs,
            "section_design_agent": {
                "subjective_supervisor_module": modules.get("subjective_supervisor_module", {"status": "unavailable"}),
                "locating_module": modules.get("locating_module", {"status": "unavailable"}),
                "mechanism_requirement_module": modules.get("mechanism_requirement_module", {"status": "unavailable"}),
                "query_semantics_module": modules.get("query_semantics_module", {"status": "unavailable"}),
                "evidence_model_module": modules.get("evidence_model_module", {"status": "unavailable"}),
                "topic_adaptation_module": modules.get("topic_adaptation_module", {"status": "unavailable"}),
                "section_partition_module": modules.get("section_partition_module", {"status": "unavailable"}),
                "supervisor_module": modules.get("supervisor_module", {"status": "unavailable"}),
                "figure_classification_module": modules.get("figure_classification_module", {"status": "unavailable"}),
                "field_planning_module": modules.get("field_planning_module", {"status": "unavailable"}),
                "schema_design_module": modules.get("schema_design_module", {"status": "unavailable"}),
                "specialization_critic_module": modules.get("specialization_critic_module", {"status": "unavailable"}),
                "aggregation": modules.get("aggregation", {"status": "unavailable"}),
            },
            "validator": {"status": "failed", "errors": validation_errors},
            "redo_agent": {"triggered": bool(validation_errors), "reason": validation_errors},
        }

    section_design = result.get("section_design", {})
    schema_definition = result.get("schema_definition", {})
    quality_check = result.get("quality_check", {})
    field_registry = schema_definition.get("field_registry", [])
    subjective_result = modules.get("subjective_supervisor_module", {})
    supervisor_result = modules.get("supervisor_module", {})
    figure_result = modules.get("figure_classification_module", {})
    critic_result = modules.get("specialization_critic_module", {})

    section_samples = []
    for section in section_design.get("core_sections", [])[:3]:
        if isinstance(section, dict):
            section_samples.append(
                {
                    "section_id": section.get("section_id"),
                    "section_name": section.get("section_name"),
                    "sample_fields": summarize_field_paths(field_registry, section.get("section_id")),
                }
            )

    figure_summary = {"text": 0, "table": 0, "figure": 0, "figure_classified": 0}
    for field in field_registry or []:
        if not isinstance(field, dict):
            continue
        source_basis = field.get("source_basis") or []
        if "text" in source_basis:
            figure_summary["text"] += 1
        if "table" in source_basis:
            figure_summary["table"] += 1
        if "figure" in source_basis:
            figure_summary["figure"] += 1
        figure_constraint = field.get("figure_constraint") or {}
        if figure_constraint.get("uses_figure_classification"):
            figure_summary["figure_classified"] += 1

    return {
        "user_input": inputs,
        "section_design_agent": {
            "subjective_supervisor_module": {
                **subjective_result,
                "must_have_count": len(subjective_result.get("must_have_concepts", [])),
            },
            "locating_module": modules.get("locating_module", {}),
            "mechanism_requirement_module": modules.get("mechanism_requirement_module", {}),
            "query_semantics_module": modules.get("query_semantics_module", {}),
            "evidence_model_module": modules.get("evidence_model_module", {}),
            "topic_adaptation_module": modules.get("topic_adaptation_module", {}),
            "section_partition_module": modules.get("section_partition_module", {}),
            "field_planning_module": modules.get("field_planning_module", {}),
            "supervisor_review": {
                **supervisor_result,
                "decision": (
                    "figure_classification_path"
                    if supervisor_result.get("enable_figure_classification")
                    else "direct_schema_path"
                ),
            },
            "risk_assessment": {
                "summary": supervisor_result.get("risk_assessment"),
                "risk_signals": supervisor_result.get("risk_signals", []),
                "trigger_sections": supervisor_result.get("trigger_sections", []),
                "expected_figure_fields": supervisor_result.get("expected_figure_fields", []),
            },
            "figure_classification_branch": {
                **figure_result,
                "planned_sections": [item.get("section_id") for item in figure_result.get("section_figure_plan", [])],
            },
            "schema_design_module": {
                **modules.get("schema_design_module", {}),
                "section_field_samples": section_samples,
                "figure_basis_summary": figure_summary,
            },
            "specialization_critic": critic_result,
            "aggregation": {
                **modules.get("aggregation", {}),
                "assembled_result_keys": list(result.keys()),
                "coverage_check": quality_check.get("coverage_check", []),
            },
        },
        "validator": {"status": "passed" if not validation_errors else "failed", "errors": validation_errors},
        "redo_agent": {
            "triggered": bool(validation_errors),
            "reason": validation_errors,
            "returns_to": "supervisor_agent",
        },
    }


def validate_result(result):
    errors = []
    required_top_keys = [
        "database_positioning",
        "section_design",
        "schema_definition",
        "quality_check",
    ]
    for key in required_top_keys:
        if key not in result:
            errors.append(f"Missing top-level key: {key}")
    if errors:
        return errors

    section_design = result.get("section_design", {})
    core_sections = section_design.get("core_sections")
    non_core_sections = section_design.get("non_core_sections")
    if not isinstance(core_sections, list) or not core_sections:
        errors.append("core_sections must be a non-empty list")
    if not isinstance(non_core_sections, list):
        errors.append("non_core_sections must be a list")

    schema_definition = result.get("schema_definition", {})
    top_level_keys = schema_definition.get("top_level_keys")
    field_registry = schema_definition.get("field_registry")
    if not isinstance(top_level_keys, list) or not top_level_keys:
        errors.append("top_level_keys must be a non-empty list")
    if not isinstance(field_registry, list) or not field_registry:
        errors.append("field_registry must be a non-empty list")

    valid_basis = {"text", "table", "figure"}
    has_core_field = False
    has_text_field = False
    has_figure_or_table_field = False
    has_figure_constraint = False
    section_ids = set()
    top_level_key_names = set()
    core_section_ids = set()

    for section in core_sections or []:
        if isinstance(section, dict) and section.get("section_id"):
            section_ids.add(section["section_id"])
            core_section_ids.add(section["section_id"])
    for section in non_core_sections or []:
        if isinstance(section, dict) and section.get("section_id"):
            section_ids.add(section["section_id"])
    for item in top_level_keys or []:
        if isinstance(item, dict) and item.get("key"):
            top_level_key_names.add(item["key"])

    for field in field_registry or []:
        if not isinstance(field, dict):
            errors.append("Each field_registry item must be an object")
            continue
        field_path = field.get("field_path")
        section_id = field.get("section_id")
        source_basis = field.get("source_basis")
        if not field_path:
            errors.append("field_registry contains an item without field_path")
        if not section_id:
            errors.append(f"field_registry item {field_path or '<unknown>'} missing section_id")
        elif section_id in core_section_ids:
            has_core_field = True
        inferred_top_level_key = field_path.split(".", 1)[0] if field_path else None
        valid_field_owner_ids = section_ids | top_level_key_names
        if section_id and section_id not in valid_field_owner_ids:
            errors.append(f"field {field_path} references unknown section_id {section_id}")
        if section_id in top_level_key_names and inferred_top_level_key != section_id:
            errors.append(
                f"field {field_path} uses top-level owner {section_id} but path starts with {inferred_top_level_key}"
            )
        if not isinstance(source_basis, list) or not source_basis:
            errors.append(f"field {field_path} must have a non-empty source_basis list")
        else:
            invalid = [item for item in source_basis if item not in valid_basis]
            if invalid:
                errors.append(f"field {field_path} has invalid source_basis values: {invalid}")
            if "text" in source_basis:
                has_text_field = True
            if "table" in source_basis or "figure" in source_basis:
                has_figure_or_table_field = True
        figure_constraint = field.get("figure_constraint")
        if "figure" in (source_basis or []):
            if not isinstance(figure_constraint, dict):
                errors.append(f"field {field_path} is figure-based but missing figure_constraint")
            else:
                if not isinstance(figure_constraint.get("uses_figure_classification"), bool):
                    errors.append(f"field {field_path} figure_constraint.uses_figure_classification must be boolean")
                allowed_sections = figure_constraint.get("allowed_sections")
                allowed_categories = figure_constraint.get("allowed_figure_categories")
                why_needed = figure_constraint.get("why_needed")
                if not isinstance(allowed_sections, list) or not allowed_sections:
                    errors.append(f"field {field_path} figure_constraint.allowed_sections must be a non-empty list")
                if not isinstance(allowed_categories, list) or not allowed_categories:
                    errors.append(
                        f"field {field_path} figure_constraint.allowed_figure_categories must be a non-empty list"
                    )
                if not isinstance(why_needed, str) or not why_needed.strip():
                    errors.append(f"field {field_path} figure_constraint.why_needed must be a non-empty string")
                has_figure_constraint = True

    quality_check = result.get("quality_check", {})
    adjustments = quality_check.get("topic_specific_adjustments")
    coverage = quality_check.get("coverage_check")
    if not isinstance(adjustments, list) or not adjustments:
        errors.append("quality_check.topic_specific_adjustments must be a non-empty list")
    if not isinstance(coverage, list) or not coverage:
        errors.append("quality_check.coverage_check must be a non-empty list")

    if not has_core_field:
        errors.append("No field_registry item is mapped to a core section")
    if not has_text_field:
        errors.append("Schema must include at least one text-based field")
    if not has_figure_or_table_field:
        errors.append("Schema must include at least one table or figure-based field")
    if not has_figure_constraint:
        errors.append("Schema must include at least one figure-bound field with figure_constraint")
    if len(core_sections or []) < 2:
        errors.append("At least two core sections are required for a useful design")
    return errors


def validate_specialization(shared_context, modules, result):
    errors = []
    query_requirements = [str(item).lower() for item in shared_context.get("query_requirements", [])]
    database_goal = str(shared_context.get("database_goal", "")).lower()
    field_registry = (((result or {}).get("schema_definition") or {}).get("field_registry")) or []
    field_paths = [field.get("field_path", "") for field in field_registry if isinstance(field, dict)]
    field_paths_lower = [item.lower() for item in field_paths]

    if "skyrmion" in database_goal or "斯格明子" in shared_context.get("database_goal", ""):
        required_pattern_groups = {
            "inversion_symmetry_broken": ["inversion_symmetry_broken"],
            "dmi": ["dmi"],
            "anisotropy": ["anisotropy"],
            "phase_window": ["phase_window", "stability_window"],
            "evidence": ["evidence"],
        }
        for label, patterns in required_pattern_groups.items():
            if not any(any(pattern in path for pattern in patterns) for path in field_paths_lower):
                errors.append(f"specialization missing skyrmion-critical concept: {label}")
        experimentally_useful_patterns = [
            "phase_identity.phase_type",
            "phase_identity.assignment_basis",
            "phase_identity.primary_method",
            "phase_identity.supporting_methods",
            "phase_identity.confidence_level",
            "phase_identity.hall_only_risk_flag",
            "dmi_source_type",
            "dmi_estimation_method",
            "dmi_evidence_links",
            "hamiltonian_terms",
            "calculation_or_simulation_method",
        ]
        for pattern in experimentally_useful_patterns:
            if not any(pattern in path for path in field_paths_lower):
                errors.append(f"specialization missing experimentally useful provenance field: {pattern}")
        if any(path.endswith(".figure") and "section1_skyrmion" in path.lower() for path in field_paths_lower):
            errors.append("specialization warning: generic section1_skyrmion.figure field should be replaced by object-level evidence links")
        if any(
            "phase_type" in path and ("hall" in path or "the" in path or "rho_xy" in path)
            for path in field_paths_lower
        ):
            errors.append(
                "specialization risk: Hall/THE transport fields must not directly own phase_type; use transport evidence plus phase_identity confidence/risk fields"
            )

    if any("图表" in req or "chart" in req for req in shared_context.get("query_requirements", [])):
        if not any("evidence" in path for path in field_paths_lower):
            errors.append("query coverage weak: chart-evidence retrieval requested but no evidence-linked field found")

    return errors


def infer_figure_constraint(field):
    field_path = str(field.get("field_path", ""))
    section_id = str(field.get("section_id", ""))
    path_lower = field_path.lower()
    section_lower = section_id.lower()

    allowed_sections = []
    allowed_categories = []
    why_needed = ""

    if path_lower == "section_magnetic_topology.phase_type" or (
        "section_skx" in path_lower
        or "section_skyrmion" in path_lower
        or "skyrmion_phases" in path_lower
        or "section_magnetic_topology" in path_lower
        or section_lower == "skyrmion_phases"
        or section_lower == "section_magnetic_topology"
    ):
        if any(token in path_lower for token in ["topological_hall", "critical_field", "rho_xy", "phase_window"]):
            allowed_sections = ["section4_mt", "topological_transport", "supporting_magnetic_properties"]
            allowed_categories = ["rho_xy_H", "phase_diagram_plot", "M_H", "hall_curve", "topological_hall"]
            why_needed = "This field records a skyrmion conclusion derived from transport, magnetization, or phase-diagram evidence."
        elif any(token in path_lower for token in ["helical_period", "skyrmion_size", "evidence", "stability_window", "phase_diagram", "extracted_parameters"]):
            allowed_sections = [
                "section3",
                "section4_mt",
                "magnetic_topology_characterization",
                "topological_transport",
                "supporting_magnetic_properties",
            ]
            allowed_categories = ["LTEM", "MFM", "SANS", "SP_STM", "rho_xy_H", "phase_diagram_plot", "hall_curve"]
            why_needed = "This field records a skyrmion conclusion or evidence link that must point to allowed imaging or transport evidence figures."
        else:
            allowed_sections = [
                "section3",
                "section4_mt",
                "magnetic_topology_characterization",
                "topological_transport",
                "supporting_magnetic_properties",
            ]
            allowed_categories = ["LTEM", "MFM", "rho_xy_H", "M_H", "phase_diagram_plot", "hall_curve", "SANS"]
            why_needed = "This field references evidence figures owned by skyrmion evidence sections."
    elif "magnetic_topology_characterization" in path_lower or section_lower == "magnetic_topology_characterization":
        allowed_sections = ["magnetic_topology_characterization", "section3", "section_magnetic_imaging", "section_magnetic_diffraction"]
        if any(token in path_lower for token in ["ltem", "skyrmion_diameter", "size", "diameter"]):
            allowed_categories = ["LTEM", "MFM", "SP_STM"]
        elif any(token in path_lower for token in ["sans", "scattering", "q_vector", "period", "reciprocal"]):
            allowed_categories = ["SANS", "REXS", "diffraction_pattern"]
        else:
            allowed_categories = ["LTEM", "MFM", "SANS", "REXS", "SP_STM", "diffraction_pattern"]
        why_needed = "This field belongs to magnetic imaging or scattering evidence and must keep ownership in the characterization section."
    elif "topological_transport" in path_lower or section_lower == "topological_transport":
        allowed_sections = ["topological_transport", "section4_mt", "section4_topological_transport"]
        if any(token in path_lower for token in ["hall", "rho_xy", "signal_peak", "peak_field"]):
            allowed_categories = ["topological_hall", "rho_xy_H", "hall_curve", "transport_phase_diagram"]
        else:
            allowed_categories = ["rho_xy_H", "hall_curve", "transport_phase_diagram"]
        why_needed = "This field belongs to topological transport evidence and must preserve Hall-curve ownership."
    elif "supporting_magnetic_properties" in path_lower or section_lower == "supporting_magnetic_properties":
        allowed_sections = ["supporting_magnetic_properties", "section4_mt", "section4_magnetization"]
        allowed_categories = ["M_H", "M_T", "magnetization_phase_diagram", "phase_diagram_plot"]
        why_needed = "This field belongs to supporting bulk magnetic evidence and must preserve magnetization-curve ownership."
    elif "material_info.general" in path_lower or section_lower in {"material_info.general", "material_general"}:
        allowed_sections = ["material_info.general", "material_general", "section0"]
        allowed_categories = ["XRD", "TEM", "crystal_structure_diagram"]
        why_needed = "This field belongs to general material or structural context and may depend on structure figures."
    elif "section1." in path_lower or section_lower == "section1":
        allowed_sections = ["section4_mt", "section3"]
        if "saturation_magnetization" in path_lower:
            allowed_categories = ["M_H", "M_T"]
        elif "anisotropy" in path_lower:
            allowed_categories = ["M_H", "M_T", "LTEM", "MFM"]
        elif "critical_field" in path_lower:
            allowed_categories = ["rho_xy_H", "M_H", "phase_diagram_plot"]
        else:
            allowed_categories = ["M_H", "M_T", "rho_xy_H"]
        why_needed = "This magnetic-property field is inferred from allowed experimental evidence and must preserve figure ownership boundaries."
    elif "section0" in path_lower or section_lower == "section0":
        allowed_sections = ["section0", "section3", "section_magnetic_imaging", "section_magnetic_diffraction", "material_general", "material_info.general"]
        if "stack_descriptor" in path_lower:
            allowed_categories = ["crystal_structure_diagram", "TEM", "XRD", "LTEM", "SANS"]
            why_needed = "This structural descriptor may be tied to structure or morphology figures and must preserve the originating evidence section."
        else:
            allowed_categories = ["crystal_structure_diagram", "TEM", "XRD"]
            why_needed = "This section0 field references structural or morphology evidence and must preserve figure ownership boundaries."
    elif (
        "section2.topological_hall" in path_lower
        or "section2.topological_hall" in path_lower.replace("_", "")
        or "section2.topological_transport_evidence" in path_lower
    ):
        allowed_sections = ["section2", "section4", "section4_topological_transport"]
        allowed_categories = ["topological_hall", "rho_xy_H", "hall_curve"]
        why_needed = "This field belongs to a topological Hall evidence record and must preserve transport-curve figure ownership."
    elif "section4_mt" in path_lower or section_lower == "section4_mt":
        allowed_sections = ["section4_mt"]
        if "rho_xy_h" in path_lower:
            allowed_categories = ["rho_xy_H"]
        elif "m_h" in path_lower:
            allowed_categories = ["M_H"]
        else:
            allowed_categories = ["rho_xy_H", "M_H", "phase_diagram_plot"]
        why_needed = "This field belongs to a transport or magnetization figure-owning section and must stay within section4_mt."
    elif "section4_topological_transport" in path_lower or section_lower == "section4_topological_transport":
        allowed_sections = ["section4_topological_transport"]
        allowed_categories = ["rho_xy_H", "hall_curve", "transport_phase_diagram"]
        why_needed = "This field belongs to the topological transport evidence-owning section and must stay within section4_topological_transport."
    elif "section4.topological_hall" in path_lower:
        allowed_sections = ["section4", "section4_topological_transport"]
        allowed_categories = ["topological_hall", "rho_xy_H", "hall_curve"]
        why_needed = "This field belongs to the dedicated topological Hall evidence within section4 and must preserve transport-curve figure ownership."
    elif "section4_magnetization" in path_lower or section_lower == "section4_magnetization":
        allowed_sections = ["section4_magnetization"]
        allowed_categories = ["M_H", "M_T", "magnetization_phase_diagram"]
        why_needed = "This field belongs to the magnetization evidence-owning section and must stay within section4_magnetization."
    elif "material_info.section4." in path_lower or path_lower.startswith("section4."):
        allowed_sections = ["section4"]
        if ".topological_hall" in path_lower:
            allowed_categories = ["topological_hall", "rho_xy_H", "hall_curve"]
            why_needed = "This field belongs to the dedicated topological Hall evidence within section4 and must preserve transport-curve figure ownership."
        elif ".m_h" in path_lower:
            allowed_categories = ["M_H", "magnetization_phase_diagram"]
            why_needed = "This field belongs to magnetization evidence within section4 and must preserve M-H figure ownership."
        elif ".r_h" in path_lower:
            allowed_categories = ["R_H", "rho_xy_H", "hall_curve"]
            why_needed = "This field belongs to Hall or magnetotransport evidence within section4 and must preserve section4 figure ownership."
        else:
            allowed_categories = ["topological_hall", "rho_xy_H", "R_H", "M_H", "phase_diagram_plot"]
            why_needed = "This field belongs to section4 physical-property evidence and must preserve section-owned figure boundaries."
    elif "material_info.section_sk." in path_lower or path_lower.startswith("section_sk."):
        if any(token in path_lower for token in ["topological_transport", "topological_hall", "resistivity"]):
            allowed_sections = ["section_sk", "section4", "section4_mt", "section4_topological_transport"]
            allowed_categories = ["topological_hall", "rho_xy_H", "hall_curve", "transport_phase_diagram"]
            why_needed = "This section_sk transport field must point to topological Hall evidence while preserving figure ownership boundaries."
        elif any(token in path_lower for token in ["critical_formation_field", "critical_annihilation_field", "stability_conditions", "temperature_stability_range"]):
            allowed_sections = ["section_sk", "section4", "section4_mt"]
            allowed_categories = ["phase_diagram_plot", "M_H", "M_T", "rho_xy_H", "topological_hall"]
            why_needed = "This section_sk stability or critical-field field must point to the phase-diagram, magnetization, or transport figure used to read the boundary."
        elif any(token in path_lower for token in ["skyrmion_size", "helical_period", "skyrmion_density", "phase_identity"]):
            allowed_sections = ["section_sk", "section3", "section_magnetic_imaging", "section_magnetic_diffraction"]
            allowed_categories = ["LTEM", "MFM", "SANS", "REXS", "SP_STM", "diffraction_pattern"]
            why_needed = "This section_sk phase or extracted-parameter field must point to direct imaging or magnetic-structure evidence."
        else:
            allowed_sections = ["section_sk", "section3", "section4", "section5"]
            allowed_categories = ["LTEM", "MFM", "SANS", "REXS", "rho_xy_H", "M_H", "phase_diagram_plot", "simulation_figures"]
            why_needed = "This section_sk field is figure-linked and must preserve the evidence section that owns the source figure."
    elif "section3" in path_lower or section_lower == "section3":
        allowed_sections = ["section3"]
        allowed_categories = ["LTEM", "MFM", "SANS", "SP_STM", "TEM", "XRD"]
        why_needed = "This field belongs to an imaging or diffraction figure-owning section and must stay within section3."
    elif "section_magnetic_imaging" in path_lower or section_lower == "section_magnetic_imaging":
        allowed_sections = ["section_magnetic_imaging"]
        allowed_categories = ["LTEM", "MFM", "SP_STM", "XMCD_PEEM"]
        why_needed = "This field belongs to the magnetic imaging evidence-owning section and must stay within section_magnetic_imaging."
    elif "section_magnetic_diffraction" in path_lower or section_lower == "section_magnetic_diffraction":
        allowed_sections = ["section_magnetic_diffraction"]
        allowed_categories = ["SANS", "REXS", "diffraction_pattern"]
        why_needed = "This field belongs to the magnetic diffraction evidence-owning section and must stay within section_magnetic_diffraction."
    elif "section5" in path_lower or section_lower == "section5":
        allowed_sections = ["section5"]
        allowed_categories = ["simulation_figures", "theoretical_model", "phase_diagram_simulation"]
        why_needed = "This field belongs to theory or simulation evidence and must stay within section5."
    elif path_lower.startswith("theory_mechanism.") or section_lower == "theory_mechanism":
        allowed_sections = ["theory_mechanism"]
        allowed_categories = ["simulation_figures", "theoretical_model", "phase_diagram_simulation", "calculation_result_plot"]
        why_needed = "This field belongs to the theory_mechanism evidence-owning section and must stay within theory_mechanism."
    elif path_lower.startswith("simulation_info.") or section_lower == "simulation_info":
        allowed_sections = ["simulation_info"]
        allowed_categories = ["simulation_figures", "phase_diagram_simulation", "calculation_result_plot", "simulation_spin_texture"]
        why_needed = "This field belongs to the simulation_info evidence-owning section and must preserve simulation figure ownership."

    if not allowed_sections:
        return None

    return {
        "uses_figure_classification": True,
        "allowed_sections": allowed_sections,
        "allowed_figure_categories": allowed_categories,
        "why_needed": why_needed,
    }


def ensure_field(field_registry, field_path, section_id, field_name, data_type, required, source_basis, reason, figure_constraint=None):
    existing_paths = {
        item.get("field_path")
        for item in field_registry
        if isinstance(item, dict) and item.get("field_path")
    }
    if field_path in existing_paths:
        return
    item = {
        "field_path": field_path,
        "section_id": section_id,
        "field_name": field_name,
        "data_type": data_type,
        "required": required,
        "source_basis": source_basis,
        "reason": reason,
    }
    if figure_constraint is not None:
        item["figure_constraint"] = figure_constraint
    field_registry.append(item)


def clone_figure_constraint(figure_constraint, why_needed=None, allowed_sections=None, allowed_categories=None):
    if not isinstance(figure_constraint, dict):
        return None
    cloned = {
        "uses_figure_classification": bool(figure_constraint.get("uses_figure_classification", True)),
        "allowed_sections": list(allowed_sections or figure_constraint.get("allowed_sections") or []),
        "allowed_figure_categories": list(allowed_categories or figure_constraint.get("allowed_figure_categories") or []),
        "why_needed": why_needed or figure_constraint.get("why_needed", ""),
    }
    if not cloned["allowed_sections"] or not cloned["allowed_figure_categories"] or not cloned["why_needed"]:
        return None
    return cloned


def normalize_source_basis_values(field_registry):
    basis_aliases = {
        "text": "text",
        "paper text": "text",
        "caption": "figure",
        "figure caption": "figure",
        "figure_caption": "figure",
        "image": "figure",
        "plot": "figure",
        "chart": "figure",
        "table": "table",
    }
    valid_basis = {"text", "table", "figure"}
    for field in field_registry:
        if not isinstance(field, dict):
            continue
        source_basis = field.get("source_basis")
        if not isinstance(source_basis, list):
            continue
        normalized = []
        for item in source_basis:
            normalized_item = basis_aliases.get(str(item).strip().lower(), str(item).strip().lower())
            if normalized_item in valid_basis and normalized_item not in normalized:
                normalized.append(normalized_item)
        field["source_basis"] = normalized or ["text"]


def detect_skyrmion_section_id(result, field_registry):
    section_design = result.get("section_design", {}) if isinstance(result, dict) else {}
    candidate_sections = []
    for group_name in ("core_sections", "non_core_sections"):
        for section in section_design.get(group_name, []) or []:
            if isinstance(section, dict) and section.get("section_id"):
                section_id = str(section["section_id"])
                section_name = str(section.get("section_name", "")).lower()
                if (
                    "skyrmion" in section_id.lower()
                    or "skx" in section_id.lower()
                    or "skyrmion" in section_name
                    or "斯格明子" in str(section.get("section_name", ""))
                ):
                    candidate_sections.append(section_id)
    if candidate_sections:
        return candidate_sections[0]

    for field in field_registry:
        if not isinstance(field, dict):
            continue
        section_id = str(field.get("section_id", ""))
        field_path = str(field.get("field_path", "")).lower()
        if "section_skyrmion_properties" in field_path or "section_skyrmion_properties" == section_id.lower():
            return "section_skyrmion_properties"
        if "section_skyrmion" in field_path or "section_skyrmion" == section_id.lower():
            return "section_skyrmion"
        if "section_skx" in field_path or "section_skx" == section_id.lower():
            return "section_skx"
    section_design = result.get("section_design", {}) if isinstance(result, dict) else {}
    all_sections = []
    for group_name in ("core_sections", "non_core_sections"):
        for section in section_design.get(group_name, []) or []:
            if isinstance(section, dict) and section.get("section_id"):
                all_sections.append(str(section["section_id"]))

    preferred_ids = [
        "section1",
        "section_skyrmion",
        "section_skx",
        "section4",
        "section3",
    ]
    for preferred_id in preferred_ids:
        if preferred_id in all_sections:
            return preferred_id
    return all_sections[0] if all_sections else "section1"


def detect_theory_section_id(result):
    section_design = result.get("section_design", {}) if isinstance(result, dict) else {}
    candidate_sections = []
    for group_name in ("core_sections", "non_core_sections"):
        for section in section_design.get(group_name, []) or []:
            if isinstance(section, dict) and section.get("section_id"):
                section_id = str(section["section_id"])
                section_name = str(section.get("section_name", "")).lower()
                if (
                    section_id.lower() == "section5"
                    or "theory" in section_name
                    or "mechanism" in section_name
                    or "simulation" in section_name
                ):
                    candidate_sections.append(section_id)
    if candidate_sections:
        return candidate_sections[0]
    return "section5"


def detect_general_section_id(result):
    section_design = result.get("section_design", {}) if isinstance(result, dict) else {}
    candidate_sections = []
    for group_name in ("core_sections", "non_core_sections"):
        for section in section_design.get(group_name, []) or []:
            if isinstance(section, dict) and section.get("section_id"):
                section_id = str(section["section_id"])
                section_name = str(section.get("section_name", "")).lower()
                section_id_lower = section_id.lower()
                if (
                    section_id_lower == "section0"
                    or "general" in section_id_lower
                    or "material_general" in section_id_lower
                    or "material system" in section_name
                    or "general information" in section_name
                    or "synthesis" in section_name
                ):
                    candidate_sections.append(section_id)
    if candidate_sections:
        return candidate_sections[0]
    return "section0"


def get_top_level_key_names(result):
    schema_definition = result.get("schema_definition", {}) if isinstance(result, dict) else {}
    top_level_keys = schema_definition.get("top_level_keys") or []
    names = set()
    for item in top_level_keys:
        if isinstance(item, dict) and item.get("key"):
            names.add(str(item["key"]))
    return names


def build_owner_field_path(result, owner_id, suffix):
    owner_id = str(owner_id)
    suffix = str(suffix).lstrip(".")
    top_level_keys = get_top_level_key_names(result)
    if owner_id in top_level_keys:
        return f"{owner_id}.{suffix}"
    if owner_id.startswith("material_info.") or owner_id.startswith("paper_info."):
        return f"{owner_id}.{suffix}"
    return f"material_info.{owner_id}.{suffix}"


def postprocess_result(shared_context, result):
    if not isinstance(result, dict):
        return result

    schema_definition = result.setdefault("schema_definition", {})
    field_registry = schema_definition.setdefault("field_registry", [])
    if not isinstance(field_registry, list):
        return result

    skyrmion_section_id = detect_skyrmion_section_id(result, field_registry)
    theory_section_id = detect_theory_section_id(result)
    general_section_id = detect_general_section_id(result)
    top_level_keys = get_top_level_key_names(result)
    normalize_source_basis_values(field_registry)

    for field in field_registry:
        if not isinstance(field, dict):
            continue
        field_path = str(field.get("field_path", ""))
        section_id = str(field.get("section_id", ""))
        if field_path.startswith("material_info.material_info."):
            field["field_path"] = field_path.replace("material_info.material_info.", "material_info.", 1)
            field_path = str(field.get("field_path", ""))
        if field_path.startswith("material_info.section0.") and general_section_id != "section0":
            field["field_path"] = field_path.replace(
                "material_info.section0.",
                build_owner_field_path(result, general_section_id, ""),
                1,
            ).replace("..", ".").rstrip(".")
            field_path = str(field.get("field_path", ""))
        if section_id == "section0" and general_section_id != "section0":
            field["section_id"] = general_section_id
            section_id = general_section_id
        if section_id in top_level_keys and field_path.startswith(f"material_info.{section_id}."):
            field["field_path"] = field_path.replace(f"material_info.{section_id}.", f"{section_id}.", 1)

    for field in field_registry:
        if not isinstance(field, dict):
            continue
        source_basis = field.get("source_basis") or []
        if "figure" in source_basis and not isinstance(field.get("figure_constraint"), dict):
            inferred = infer_figure_constraint(field)
            if not inferred:
                field_path = str(field.get("field_path", ""))
                if field_path.endswith(".figure"):
                    parent_path = field_path.rsplit(".", 1)[0]
                    inferred = infer_figure_constraint(
                        {
                            "field_path": parent_path,
                            "section_id": field.get("section_id", ""),
                        }
                    )
            if inferred:
                field["figure_constraint"] = inferred

    database_goal = str(shared_context.get("database_goal", ""))
    if "斯格明子" in database_goal or "skyrmion" in database_goal.lower():
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "inversion_symmetry_broken"),
            general_section_id,
            "Inversion Symmetry Broken",
            "string",
            False,
            ["text"],
            "Records whether broken inversion symmetry is explicitly stated, which is a core prerequisite for DMI-mediated skyrmion stabilization.",
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "dmi_origin"),
            general_section_id,
            "DMI Origin",
            "string",
            False,
            ["text"],
            "Records the explicitly stated origin of Dzyaloshinskii-Moriya interaction such as bulk, interfacial, or otherwise discussed symmetry-breaking source.",
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "dmi_source_type"),
            general_section_id,
            "DMI Source Type",
            "enum: experimental_estimate | simulation_fit | DFT_calculation | literature_assumption | author_interpretation | not_reported",
            False,
            ["text", "table", "figure"],
            "Distinguishes whether DMI information is measured, fitted, simulated, DFT-derived, assumed from literature, or only interpreted by authors.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt", "section5", "theory_mechanism", "simulation_info"],
                "allowed_figure_categories": ["M_H", "phase_diagram_plot", "simulation_figures", "calculation_result_plot", "theoretical_model"],
                "why_needed": "DMI provenance may come from experiment, fitting, simulation, or DFT and must not be mixed without evidence classification.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "dmi_value"),
            general_section_id,
            "DMI Value",
            "number or string",
            False,
            ["text", "table", "figure"],
            "Records the reported DMI magnitude only when a value is explicitly provided or extracted with a stated method.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section4_mt", "section5", "theory_mechanism", "simulation_info"],
                "allowed_figure_categories": ["M_H", "phase_diagram_plot", "simulation_figures", "calculation_result_plot"],
                "why_needed": "DMI values are often obtained from fitting, simulation, or DFT plots rather than direct experiment.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "dmi_unit"),
            general_section_id,
            "DMI Unit",
            "string",
            False,
            ["text", "table"],
            "Stores the unit attached to dmi_value so values from micromagnetic, atomistic, or DFT sources are not conflated.",
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "dmi_estimation_method"),
            general_section_id,
            "DMI Estimation Method",
            "string",
            False,
            ["text"],
            "Records the stated method used to obtain DMI, such as BLS, domain-wall fit, micromagnetic fitting, or DFT total-energy calculation.",
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "dmi_evidence_links"),
            general_section_id,
            "DMI Evidence Links",
            "array of evidence references",
            False,
            ["figure", "table", "text"],
            "Links DMI origin and values to the exact supporting evidence instead of leaving DMI as an unsupported mechanism keyword.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt", "section5", "theory_mechanism", "simulation_info"],
                "allowed_figure_categories": ["M_H", "domain_image", "phase_diagram_plot", "simulation_figures", "calculation_result_plot", "theoretical_model"],
                "why_needed": "DMI support must point to the method-specific evidence source.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "magnetic_anisotropy_type"),
            general_section_id,
            "Magnetic Anisotropy Type",
            "string",
            False,
            ["text", "figure"],
            "Records the explicitly stated anisotropy type relevant to skyrmion stabilization.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section4_mt", "section3"],
                "allowed_figure_categories": ["M_H", "M_T", "LTEM", "MFM"],
                "why_needed": "Magnetic anisotropy may be concluded from magnetization curves or discussed alongside imaging evidence.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.phase_type"),
            skyrmion_section_id,
            "Topological Magnetic Phase Type",
            "enum or string",
            False,
            ["text", "figure"],
            "Records the claimed phase type, such as skyrmion, antiskyrmion, meron, bimeron, bubble, stripe, cycloid, helix, bobber, or hopfion.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section_magnetic_imaging", "section_magnetic_diffraction"],
                "allowed_figure_categories": ["LTEM", "MFM", "SP_STM", "electron_holography", "SANS", "REXS", "diffraction_pattern"],
                "why_needed": "High-confidence phase type should be assigned from direct imaging or symmetry-sensitive magnetic-structure evidence, not transport alone.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.assignment_basis"),
            skyrmion_section_id,
            "Phase Assignment Basis",
            "array of strings",
            False,
            ["text", "figure"],
            "Records the actual basis used to assign the phase type, such as direct imaging, symmetry plus imaging, SANS, Hall support, magnetization anomaly, or simulation comparison.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt", "section_magnetic_imaging", "section_magnetic_diffraction", "section4_topological_transport"],
                "allowed_figure_categories": ["LTEM", "MFM", "SP_STM", "electron_holography", "SANS", "rho_xy_H", "M_H", "phase_diagram_plot"],
                "why_needed": "The database must preserve how the phase claim was assigned so experimental users can judge reliability.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.primary_method"),
            skyrmion_section_id,
            "Primary Phase Identification Method",
            "string",
            False,
            ["text", "figure"],
            "Stores the primary method supporting phase identification; direct imaging or symmetry-sensitive methods should be distinguishable from Hall-only support.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt", "section_magnetic_imaging", "section_magnetic_diffraction", "section4_topological_transport"],
                "allowed_figure_categories": ["LTEM", "MFM", "SP_STM", "electron_holography", "SANS", "rho_xy_H", "M_H", "phase_diagram_plot"],
                "why_needed": "Phase identity confidence depends strongly on the primary identification method.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.supporting_methods"),
            skyrmion_section_id,
            "Supporting Phase Identification Methods",
            "array of strings",
            False,
            ["text", "figure"],
            "Stores secondary evidence methods supporting the phase claim without promoting weak evidence to direct identification.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt", "section_magnetic_imaging", "section_magnetic_diffraction", "section4_topological_transport"],
                "allowed_figure_categories": ["LTEM", "MFM", "SP_STM", "electron_holography", "SANS", "rho_xy_H", "M_H", "phase_diagram_plot"],
                "why_needed": "Supporting evidence should be queryable separately from primary phase identification.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.confidence_level"),
            skyrmion_section_id,
            "Phase Identification Confidence Level",
            "enum: high | medium | low | disputed | not_assignable",
            False,
            ["text", "figure"],
            "Grades phase identity confidence based on evidence type; Hall-only transport support should generally be low confidence.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt", "section_magnetic_imaging", "section_magnetic_diffraction", "section4_topological_transport"],
                "allowed_figure_categories": ["LTEM", "MFM", "SP_STM", "electron_holography", "SANS", "rho_xy_H", "M_H", "phase_diagram_plot"],
                "why_needed": "Experimental users need a confidence grade before reusing phase labels.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.hall_only_risk_flag"),
            skyrmion_section_id,
            "Hall-Only Phase Assignment Risk Flag",
            "boolean",
            False,
            ["text", "figure"],
            "Flags cases where the skyrmion/topological phase assignment is based only on Hall/THE transport without direct imaging or symmetry-sensitive confirmation.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section4_mt", "section4_topological_transport"],
                "allowed_figure_categories": ["rho_xy_H", "hall_curve", "topological_hall", "transport_phase_diagram"],
                "why_needed": "Hall-only evidence is risky for phase-type assignment and must be explicitly marked.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.evidence_links"),
            skyrmion_section_id,
            "Phase Identity Evidence Links",
            "array of evidence references",
            False,
            ["figure", "table", "text"],
            "Links the phase identity claim to specific imaging, diffraction, transport, text, or table evidence records.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt", "section_magnetic_imaging", "section_magnetic_diffraction", "section4_topological_transport"],
                "allowed_figure_categories": ["LTEM", "MFM", "SP_STM", "electron_holography", "SANS", "REXS", "rho_xy_H", "M_H", "phase_diagram_plot"],
                "why_needed": "Phase labels must remain traceable to evidence sources.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_window.temperature_range"),
            skyrmion_section_id,
            "Skyrmion Phase Temperature Range",
            "string or array of strings",
            False,
            ["figure", "text"],
            "Records the temperature window of the skyrmion phase pocket rather than collapsing it into one critical field.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section4_mt"],
                "allowed_figure_categories": ["phase_diagram_plot"],
                "why_needed": "The skyrmion phase temperature range is typically extracted from H-T phase diagrams or equivalent transport evidence.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "phase_window.field_range"),
            skyrmion_section_id,
            "Skyrmion Phase Field Range",
            "string or array of strings",
            False,
            ["figure", "text"],
            "Records the field window of the skyrmion phase pocket rather than collapsing it into one critical field.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section4_mt"],
                "allowed_figure_categories": ["phase_diagram_plot", "M_H", "rho_xy_H"],
                "why_needed": "The skyrmion phase field range is extracted from phase-diagram or transport evidence that defines stability boundaries.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "stability_window.temperature_range"),
            skyrmion_section_id,
            "Stability Window Temperature Range",
            "string or array of strings",
            False,
            ["figure", "text"],
            "Records the temperature span over which the reported skyrmion phase remains stable under stated conditions.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section4_mt"],
                "allowed_figure_categories": ["phase_diagram_plot"],
                "why_needed": "Stability windows are usually determined from phase diagrams or equivalent mapped evidence.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "stability_window.field_range"),
            skyrmion_section_id,
            "Stability Window Field Range",
            "string or array of strings",
            False,
            ["figure", "text"],
            "Records the magnetic-field span over which the reported skyrmion phase remains stable under stated conditions.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section4_mt"],
                "allowed_figure_categories": ["phase_diagram_plot", "M_H", "rho_xy_H"],
                "why_needed": "Field stability windows are extracted from phase-diagram, Hall, or magnetization boundary evidence.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, skyrmion_section_id, "evidence_links"),
            skyrmion_section_id,
            "Evidence Links",
            "array of strings",
            False,
            ["figure"],
            "Aggregates evidence figure references that support skyrmion claims while preserving figure ownership in evidence sections.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt"],
                "allowed_figure_categories": ["LTEM", "MFM", "SANS", "SP_STM", "rho_xy_H", "M_H", "phase_diagram_plot"],
                "why_needed": "This field provides an evidence-level handle for chart and image retrieval without duplicating figure ownership.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, theory_section_id, "hamiltonian_terms"),
            theory_section_id,
            "Hamiltonian Terms",
            "array of objects",
            False,
            ["text", "table", "figure"],
            "Records explicit Hamiltonian terms such as exchange, DMI, anisotropy, Zeeman, and dipolar terms with parameter values when provided.",
            {
                "uses_figure_classification": True,
                "allowed_sections": [theory_section_id, "section5", "simulation_info"],
                "allowed_figure_categories": ["theoretical_model", "simulation_figures", "phase_diagram_simulation", "calculation_result_plot"],
                "why_needed": "Theory and simulation parameters must be kept separate from experimental observables.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, theory_section_id, "calculation_or_simulation_method"),
            theory_section_id,
            "Calculation Or Simulation Method",
            "string",
            False,
            ["text"],
            "Records whether the mechanism parameters are from DFT, micromagnetic simulation, Monte Carlo, atomistic spin simulation, analytical model, or another method.",
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "intrinsic_magnetic_parameters.determination_method"),
            general_section_id,
            "Intrinsic Magnetic Parameter Determination Method",
            "string",
            False,
            ["text"],
            "Records whether each intrinsic magnetic parameter was measured, fitted, estimated, simulated, or calculated.",
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "intrinsic_magnetic_parameters.source_figure"),
            general_section_id,
            "Intrinsic Magnetic Parameter Source Figure",
            "string",
            False,
            ["figure"],
            "Links each intrinsic magnetic parameter record to the specific source figure used for extraction or fitting.",
            {
                "uses_figure_classification": True,
                "allowed_sections": ["section3", "section4_mt", "section5", "simulation_info"],
                "allowed_figure_categories": ["M_H", "M_T", "phase_diagram_plot", "simulation_figures", "calculation_result_plot"],
                "why_needed": "Intrinsic magnetic parameters may be extracted from experiment, fitting, or simulation figures and need explicit provenance.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, theory_section_id, "theoretical_parameters.determination_method"),
            theory_section_id,
            "Theoretical Parameter Determination Method",
            "string",
            False,
            ["text"],
            "Records whether each theoretical or mechanism parameter comes from DFT, micromagnetic fitting, Monte Carlo, analytical modeling, or another method.",
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, theory_section_id, "theoretical_parameters.source_figure"),
            theory_section_id,
            "Theoretical Parameter Source Figure",
            "string",
            False,
            ["figure"],
            "Links each theoretical parameter record to the exact calculation or simulation figure that supports it.",
            {
                "uses_figure_classification": True,
                "allowed_sections": [theory_section_id, "section5", "simulation_info"],
                "allowed_figure_categories": ["theoretical_model", "simulation_figures", "phase_diagram_simulation", "calculation_result_plot"],
                "why_needed": "Theory parameters need explicit figure provenance when they are derived from calculations, fits, or simulation outputs.",
            },
        )
        ensure_field(
            field_registry,
            build_owner_field_path(result, general_section_id, "dmi_value.figure"),
            general_section_id,
            "DMI Value Source Figure",
            "string",
            False,
            ["figure"],
            "Links the reported DMI value to the exact source figure instead of leaving provenance only at the parent object level.",
            clone_figure_constraint(
                infer_figure_constraint(
                    {
                        "field_path": build_owner_field_path(result, general_section_id, "dmi_value"),
                        "section_id": general_section_id,
                    }
                ),
                why_needed="This figure link points to the source figure used to determine or fit the DMI value.",
            ),
        )
        for field_path, field_name, reason, figure_constraint in [
            (
                build_owner_field_path(result, skyrmion_section_id, "size.figure"),
                "Skyrmion Size Source Figure",
                "Links the reported skyrmion size to the exact imaging or scattering figure used for extraction.",
                {
                    "uses_figure_classification": True,
                    "allowed_sections": ["section3", "section4_mt", "section_magnetic_imaging", "section_magnetic_diffraction"],
                    "allowed_figure_categories": ["LTEM", "MFM", "SANS", "SP_STM", "phase_diagram_plot"],
                    "why_needed": "Skyrmion size should point to a concrete evidence figure instead of inheriting generic parent-level figure ownership.",
                },
            ),
            (
                build_owner_field_path(result, skyrmion_section_id, "critical_field_formation.figure"),
                "Formation Critical Field Source Figure",
                "Links the formation field value to the exact Hall, magnetization, or phase-diagram figure used for extraction.",
                {
                    "uses_figure_classification": True,
                    "allowed_sections": ["section4_mt", "section4_topological_transport"],
                    "allowed_figure_categories": ["rho_xy_H", "M_H", "phase_diagram_plot", "hall_curve"],
                    "why_needed": "Formation critical fields must trace back to the curve or phase diagram from which the boundary was read.",
                },
            ),
            (
                build_owner_field_path(result, skyrmion_section_id, "critical_field_annihilation.figure"),
                "Annihilation Critical Field Source Figure",
                "Links the annihilation field value to the exact Hall, magnetization, or phase-diagram figure used for extraction.",
                {
                    "uses_figure_classification": True,
                    "allowed_sections": ["section4_mt", "section4_topological_transport"],
                    "allowed_figure_categories": ["rho_xy_H", "M_H", "phase_diagram_plot", "hall_curve"],
                    "why_needed": "Annihilation critical fields must trace back to the curve or phase diagram from which the boundary was read.",
                },
            ),
            (
                build_owner_field_path(result, skyrmion_section_id, "topological_Hall_resistivity.figure"),
                "Topological Hall Resistivity Source Figure",
                "Links the topological Hall value to the exact transport curve used for extraction.",
                {
                    "uses_figure_classification": True,
                    "allowed_sections": ["section4_mt", "section4_topological_transport"],
                    "allowed_figure_categories": ["rho_xy_H", "hall_curve", "topological_hall"],
                    "why_needed": "Topological Hall values must preserve transport-curve provenance.",
                },
            ),
            (
                build_owner_field_path(result, skyrmion_section_id, "helical_period.figure"),
                "Helical Period Source Figure",
                "Links the helical period value to the exact scattering or imaging figure used for extraction.",
                {
                    "uses_figure_classification": True,
                    "allowed_sections": ["section3", "section_magnetic_diffraction", "section_magnetic_imaging"],
                    "allowed_figure_categories": ["SANS", "REXS", "diffraction_pattern", "LTEM", "MFM"],
                    "why_needed": "Helical period values must remain tied to the reciprocal-space or imaging evidence from which they were extracted.",
                },
            ),
        ]:
            ensure_field(
                field_registry,
                field_path,
                skyrmion_section_id,
                field_name,
                "string",
                False,
                ["figure"],
                reason,
                figure_constraint,
            )

        for required_phase_field in [
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.phase_type"),
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.assignment_basis"),
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.primary_method"),
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.supporting_methods"),
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.confidence_level"),
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.hall_only_risk_flag"),
            build_owner_field_path(result, skyrmion_section_id, "phase_identity.evidence_links"),
        ]:
            for field in field_registry:
                if isinstance(field, dict) and field.get("field_path") == required_phase_field:
                    field["required"] = True
                    break

    return result


def finalize_result(shared_context, result):
    if not isinstance(result, dict):
        return result

    result = postprocess_result(shared_context, result)
    schema_definition = result.setdefault("schema_definition", {})
    field_registry = schema_definition.get("field_registry")
    if isinstance(field_registry, list):
        normalize_source_basis_values(field_registry)
    return result


def call_module(client, model, module_name, prompt, temperature=0, max_retries=1):
    attempts = []
    current_prompt = prompt
    result = None
    errors = []
    for round_idx in range(max_retries + 1):
        raw_response = chat(client, model, current_prompt, temperature=temperature)
        attempts.append({"round": round_idx + 1, "prompt": current_prompt, "raw_response": raw_response})
        result, parse_errors = parse_json_response(raw_response)
        if parse_errors:
            errors = parse_errors
        else:
            result = normalize_module_result(module_name, result)
            errors = validate_module_result(module_name, result)
        if not errors:
            break
        if round_idx < max_retries:
            current_prompt = build_module_redo_prompt(
                module_name,
                raw_response,
                errors,
                MODULE_SCHEMAS[module_name]["schema_text"],
            )
    return result, errors, attempts


def run_pipeline(client, args, shared_context):
    module_outputs = {}
    module_attempts = {}
    module_errors = {}

    locating_result, locating_errors, locating_attempts = call_module(
        client,
        args.model,
        "locating_module",
        build_locating_prompt(
            args.database_goal,
            args.discipline,
            shared_context["query_requirements"],
            shared_context["key_description_text"],
            shared_context.get("reference_paper_context", ""),
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["locating_module"] = locating_result
    module_attempts["locating_module"] = locating_attempts
    module_errors["locating_module"] = locating_errors
    if locating_errors:
        return None, locating_errors, module_outputs, module_attempts, module_errors

    mechanism_result, mechanism_errors, mechanism_attempts = call_module(
        client,
        args.model,
        "mechanism_requirement_module",
        build_mechanism_requirement_prompt(shared_context, locating_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["mechanism_requirement_module"] = mechanism_result
    module_attempts["mechanism_requirement_module"] = mechanism_attempts
    module_errors["mechanism_requirement_module"] = mechanism_errors
    if mechanism_errors:
        return None, mechanism_errors, module_outputs, module_attempts, module_errors

    query_result, query_errors, query_attempts = call_module(
        client,
        args.model,
        "query_semantics_module",
        build_query_semantics_prompt(shared_context, locating_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["query_semantics_module"] = query_result
    module_attempts["query_semantics_module"] = query_attempts
    module_errors["query_semantics_module"] = query_errors
    if query_errors:
        return None, query_errors, module_outputs, module_attempts, module_errors

    evidence_result, evidence_errors, evidence_attempts = call_module(
        client,
        args.model,
        "evidence_model_module",
        build_evidence_model_prompt(shared_context, locating_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["evidence_model_module"] = evidence_result
    module_attempts["evidence_model_module"] = evidence_attempts
    module_errors["evidence_model_module"] = evidence_errors
    if evidence_errors:
        return None, evidence_errors, module_outputs, module_attempts, module_errors

    subjective_result, subjective_errors, subjective_attempts = call_module(
        client,
        args.model,
        "subjective_supervisor_module",
        build_subjective_supervisor_prompt(
            shared_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["subjective_supervisor_module"] = subjective_result
    module_attempts["subjective_supervisor_module"] = subjective_attempts
    module_errors["subjective_supervisor_module"] = subjective_errors
    if subjective_errors:
        return None, subjective_errors, module_outputs, module_attempts, module_errors

    topic_result, topic_errors, topic_attempts = call_module(
        client,
        args.model,
        "topic_adaptation_module",
        build_topic_adaptation_prompt(shared_context, locating_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["topic_adaptation_module"] = topic_result
    module_attempts["topic_adaptation_module"] = topic_attempts
    module_errors["topic_adaptation_module"] = topic_errors
    if topic_errors:
        return None, topic_errors, module_outputs, module_attempts, module_errors

    section_result, section_errors, section_attempts = call_module(
        client,
        args.model,
        "section_partition_module",
        build_section_partition_prompt(
            shared_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            topic_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["section_partition_module"] = section_result
    module_attempts["section_partition_module"] = section_attempts
    module_errors["section_partition_module"] = section_errors
    if section_errors:
        return None, section_errors, module_outputs, module_attempts, module_errors

    field_plan_result, field_plan_errors, field_plan_attempts = call_module(
        client,
        args.model,
        "field_planning_module",
        build_field_planning_prompt(
            shared_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            topic_result,
            section_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["field_planning_module"] = field_plan_result
    module_attempts["field_planning_module"] = field_plan_attempts
    module_errors["field_planning_module"] = field_plan_errors
    if field_plan_errors:
        return None, field_plan_errors, module_outputs, module_attempts, module_errors

    human_advice = shared_context.get("human_advice", "")
    if getattr(args, "require_human_advice_before_supervisor", False) and not human_advice:
        module_outputs["human_advice_gate"] = {
            "status": "waiting_for_human_advice",
            "position": "before_supervisor_module",
            "reason": (
                "Human advice is required before the supervisor can decide the "
                "figure-classification routing path."
            ),
            "expected_advice": (
                "Provide expert suggestions about figure ownership risks, sections "
                "that must be reviewed by the supervisor, and any fields that should "
                "or should not use figure evidence."
            ),
        }
        module_errors["human_advice_gate"] = ["needs_human_advice_before_supervisor"]
        module_attempts["human_advice_gate"] = []
        return (
            None,
            ["needs_human_advice_before_supervisor"],
            module_outputs,
            module_attempts,
            module_errors,
        )

    supervisor_result, supervisor_errors, supervisor_attempts = call_module(
        client,
        args.model,
        "supervisor_module",
        build_supervisor_prompt(
            shared_context,
            locating_result,
            topic_result,
            section_result,
            field_plan_result,
            human_advice=human_advice,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["supervisor_module"] = supervisor_result
    module_attempts["supervisor_module"] = supervisor_attempts
    module_errors["supervisor_module"] = supervisor_errors
    if supervisor_errors:
        return None, supervisor_errors, module_outputs, module_attempts, module_errors

    figure_result, figure_errors, figure_attempts = call_module(
        client,
        args.model,
        "figure_classification_module",
        build_figure_classification_prompt(shared_context, section_result, field_plan_result, supervisor_result),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["figure_classification_module"] = figure_result
    module_attempts["figure_classification_module"] = figure_attempts
    module_errors["figure_classification_module"] = figure_errors
    if figure_errors:
        return None, figure_errors, module_outputs, module_attempts, module_errors
    if supervisor_result.get("enable_figure_classification") != figure_result.get("enable_figure_classification"):
        return (
            None,
            ["Supervisor decision and figure classification branch are inconsistent"],
            module_outputs,
            module_attempts,
            module_errors,
        )

    schema_result, schema_errors, schema_attempts = call_module(
        client,
        args.model,
        "schema_design_module",
        build_schema_design_prompt(
            shared_context,
            locating_result,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            topic_result,
            section_result,
            field_plan_result,
            supervisor_result,
            figure_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["schema_design_module"] = schema_result
    module_attempts["schema_design_module"] = schema_attempts
    module_errors["schema_design_module"] = schema_errors
    if schema_errors:
        return None, schema_errors, module_outputs, module_attempts, module_errors

    critic_result, critic_errors, critic_attempts = call_module(
        client,
        args.model,
        "specialization_critic_module",
        build_specialization_critic_prompt(
            shared_context,
            mechanism_result,
            query_result,
            evidence_result,
            subjective_result,
            section_result,
            field_plan_result,
            schema_result,
        ),
        temperature=args.temperature,
        max_retries=args.max_retries,
    )
    module_outputs["specialization_critic_module"] = critic_result
    module_attempts["specialization_critic_module"] = critic_attempts
    module_errors["specialization_critic_module"] = critic_errors
    if critic_errors:
        return None, critic_errors, module_outputs, module_attempts, module_errors

    aggregation_prompt = build_aggregation_prompt(
        shared_context,
        locating_result,
        mechanism_result,
        query_result,
        evidence_result,
        subjective_result,
        topic_result,
        section_result,
        field_plan_result,
        supervisor_result,
        figure_result,
        schema_result,
        critic_result,
    )
    final_result, final_errors, aggregation_attempts = call_module(
        client,
        args.model,
        "aggregation",
        aggregation_prompt,
        temperature=args.temperature,
        max_retries=0,
    )
    module_outputs["aggregation"] = final_result
    module_attempts["aggregation"] = aggregation_attempts
    module_errors["aggregation"] = final_errors
    if final_errors:
        return None, final_errors, module_outputs, module_attempts, module_errors

    final_result = finalize_result(shared_context, final_result)
    validation_errors = validate_result(final_result)
    validation_errors.extend(validate_specialization(shared_context, module_outputs, final_result))
    return final_result, validation_errors, module_outputs, module_attempts, module_errors


def run_section_design(args):
    client = get_client(base_url=args.base_url, api_key=args.api_key, backend=args.llm_backend)
    query_requirements = load_query_requirements(args.query_requirements)
    key_description_text = load_key_description_text(args.key_description_path)
    reference_paper_context = load_reference_paper_context(args.reference_papers)
    human_advice = (
        args.human_advice.strip()
        if args.human_advice
        else load_optional_text(args.human_advice_path)
    )
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

    result, errors, module_outputs, module_attempts, module_errors = run_pipeline(client, args, shared_context)
    final_attempts = []

    if result is not None and errors and args.max_retries > 0:
        redo_prompt = build_redo_prompt(json.dumps(result, ensure_ascii=False, indent=2), errors)
        raw_response = chat(client, args.model, redo_prompt, temperature=args.temperature)
        final_attempts.append({"round": 1, "prompt": redo_prompt, "raw_response": raw_response})
        redo_result, parse_errors = parse_json_response(raw_response)
        if not parse_errors:
            redo_result = finalize_result(shared_context, redo_result)
            redo_validation_errors = validate_result(redo_result)
            redo_validation_errors.extend(validate_specialization(shared_context, module_outputs, redo_result))
            if len(redo_validation_errors) <= len(errors):
                result = redo_result
                errors = redo_validation_errors

    output_inputs = {
        "database_goal": args.database_goal,
        "discipline": args.discipline,
        "query_requirements": query_requirements,
        "key_description_path": str(Path(args.key_description_path)),
        "reference_papers": args.reference_papers,
        "human_advice_required_before_supervisor": args.require_human_advice_before_supervisor,
        "human_advice_provided": bool(human_advice),
    }
    output = {
        "step": "step8_section_design_agent",
        "framework": STEP8_FRAMEWORK,
        "inputs": output_inputs,
        "agent_flow": build_agent_flow_trace(output_inputs, module_outputs, result, errors),
        "module_outputs": module_outputs,
        "module_errors": module_errors,
        "result": result,
        "validation_errors": errors,
        "status": "success" if not errors else "needs_review",
        "attempts": {
            "modules": module_attempts,
            "final_redo": final_attempts,
        },
    }

    output_path = Path(args.output)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved result to {output_path}")
    print(f"Status: {output['status']}")
    if errors:
        print("Validation errors:")
        for item in errors:
            print(f"- {item}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database-goal", required=True)
    parser.add_argument("--discipline", required=True)
    parser.add_argument("--query-requirements", required=True)
    parser.add_argument("--key-description-path", required=True)
    parser.add_argument("--reference-papers", nargs="*", default=[])
    parser.add_argument("--output", default="./section_design_output.json")
    parser.add_argument("--model", default=os.getenv("SECTION_AGENT_MODEL", DEFAULT_DEEPSEEK_MODEL))
    parser.add_argument("--base-url", default=os.getenv("SECTION_AGENT_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL))
    parser.add_argument("--api-key", default=os.getenv("SECTION_AGENT_API_KEY"))
    parser.add_argument(
        "--llm-backend",
        choices=["openai", "langchain"],
        default=os.getenv("SECTION_AGENT_LLM_BACKEND", DEFAULT_LLM_BACKEND),
        help="LLM client backend. Use `langchain` to call through LangChain ChatOpenAI.",
    )
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--require-human-advice-before-supervisor",
        action="store_true",
        help="Stop before supervisor_module unless human advice is provided.",
    )
    parser.add_argument(
        "--human-advice",
        default="",
        help="Human expert advice injected before supervisor_module.",
    )
    parser.add_argument(
        "--human-advice-path",
        default="",
        help="Path to a UTF-8 text file containing human expert advice.",
    )
    return parser


if __name__ == "__main__":
    run_section_design(build_parser().parse_args())
