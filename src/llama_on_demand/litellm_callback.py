"""Expose per-request, non-fatal CPU offload warnings through LiteLLM."""

from litellm.integrations.custom_logger import CustomLogger


class OffloadHeaders(CustomLogger):
    async def async_post_call_response_headers_hook(
        self, data, user_api_key_dict, response, request_headers=None, litellm_call_info=None
    ):
        hidden = getattr(response, "_hidden_params", {}) or {}
        headers = hidden.get("additional_headers", {}) or {}
        warning = headers.get("llm_provider-x-local-llm-warning")
        if not warning:
            raw = getattr(response, "_response_headers", {}) or hidden.get("headers", {}) or {}
            warning = raw.get("x-local-llm-warning")
        return {"x-local-llm-warning": warning} if warning else {}


callback = OffloadHeaders()
