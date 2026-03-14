## Current Task
{{ handoff.current_task }}

## Key Decisions & Findings
{% if handoff.key_decisions.is_empty() %}
- None recorded.
{% else %}
{% for item in handoff.key_decisions %}
- {{ item }}
{% endfor %}
{% endif %}

## Files & Paths
{% if handoff.files_paths.is_empty() %}
- None recorded.
{% else %}
{% for item in handoff.files_paths %}
- {{ item }}
{% endfor %}
{% endif %}

## Errors & Fixes
{% if handoff.errors_fixes.is_empty() %}
- None recorded.
{% else %}
{% for item in handoff.errors_fixes %}
- {{ item }}
{% endfor %}
{% endif %}

## Pending Work
{% if handoff.pending_work.is_empty() %}
- None recorded.
{% else %}
{% for item in handoff.pending_work %}
- {{ item }}
{% endfor %}
{% endif %}

## Recovery Hints
{% if handoff.recovery_hints.is_empty() %}
- None recorded.
{% else %}
{% for item in handoff.recovery_hints %}
- {{ item }}
{% endfor %}
{% endif %}
