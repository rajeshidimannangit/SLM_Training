"""Side-by-side Gradio chat: base SLM vs fine-tuned SLM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from slm_finetune.inference.engine import (
    DEFAULT_INSTRUCTION,
    DualModelSession,
    clean_completion,
    list_finetuned_adapters,
    session_from_configs,
)
from slm_finetune.utils.config import load_yaml, resolve_path
from slm_finetune.utils.logging import setup_logging

logger = setup_logging()

# Module-level session so Gradio callbacks share loaded weights
_SESSION: DualModelSession | None = None

_CSS = """
.cmp-col { min-height: 420px; }
"""


def _finetuned_choices(finetuned_root: str = "models/finetuned") -> list[str]:
    names = list_finetuned_adapters(finetuned_root)
    return [str(resolve_path(finetuned_root) / n) for n in names]


def _fmt_reply(text: str, latency: float) -> str:
    body = clean_completion(text) or "(empty)"
    return f"{body}\n\n— _{latency:.2f}s_"


def _message_text(content: Any) -> str:
    """Normalize Gradio chatbot content (str | list[{text,type}] | dict) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if "text" in content:
            return str(content.get("text") or "")
        if "content" in content:
            return _message_text(content.get("content"))
        return str(content)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            parts.append(_message_text(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _strip_latency(text: str) -> str:
    if "\n\n— _" in text:
        return text.split("\n\n— _")[0].strip()
    return text.strip()


def _is_usable_assistant(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t.startswith("⚠️") or t.startswith("❌"):
        return False
    if t.startswith("[{") and "type" in t:
        return False
    return True


def _history_pairs(chat_hist: list | None) -> list[tuple[str, str]]:
    paired: list[tuple[str, str]] = []
    pending_user: str | None = None
    for turn in chat_hist or []:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        content = _strip_latency(_message_text(turn.get("content")))
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            if _is_usable_assistant(content):
                paired.append((pending_user, content))
            pending_user = None
    return paired


def build_app(
    *,
    model_config: str = "configs/model/default.yaml",
    finetuned_model_dir: str | None = None,
    finetuned_root: str = "models/finetuned",
    share: bool = False,
):
    try:
        import gradio as gr
    except ImportError as exc:
        raise ImportError(
            "Gradio is required for the compare chat UI. Install with: pip install gradio"
        ) from exc

    model_cfg = load_yaml(model_config)
    base_name = model_cfg["model"]["name_or_path"]
    choices = _finetuned_choices(finetuned_root)
    default_ft = finetuned_model_dir
    if default_ft:
        default_ft = str(resolve_path(default_ft))
    elif choices:
        default_ft = choices[0]

    with gr.Blocks(title="Base vs Fine-tuned — SLM Compare Chat") as demo:
        gr.Markdown(
            f"""
# Base vs Fine-tuned compare chat
One prompt → answers from **base** (`{base_name}`) and your **fine-tuned** adapter side by side.

Each send is a **fresh** classification prompt by default (no chat memory), matching training.
"""
        )

        with gr.Row():
            ft_dropdown = gr.Dropdown(
                label="Fine-tuned adapter",
                choices=choices,
                value=default_ft,
                interactive=True,
                scale=3,
            )
            refresh_btn = gr.Button("Refresh list", scale=1)
            load_btn = gr.Button("Load models", variant="primary", scale=1)

        status = gr.Markdown("_Models not loaded yet. Choose an adapter and click **Load models**._")

        with gr.Accordion("Prompt & generation", open=False):
            instruction = gr.Textbox(
                label="Instruction (Alpaca)",
                value=DEFAULT_INSTRUCTION,
                lines=4,
            )
            prompt_mode = gr.Radio(
                choices=["alpaca", "raw"],
                value="alpaca",
                label="Prompt mode",
                info="alpaca wraps the message with Instruction/Input/Response; raw sends text as-is",
            )
            use_history = gr.Checkbox(
                label="Multi-turn history (off = one independent classification per message)",
                value=False,
            )
            with gr.Row():
                max_new_tokens = gr.Slider(16, 256, value=48, step=8, label="Max new tokens")
                temperature = gr.Slider(0.0, 1.5, value=0.0, step=0.05, label="Temperature (0 = greedy)")
            do_sample = gr.Checkbox(label="Sample (use temperature)", value=False)

        with gr.Row():
            base_chat = gr.Chatbot(
                label="Base SLM",
                height=420,
                elem_classes=["cmp-col"],
            )
            ft_chat = gr.Chatbot(
                label="Fine-tuned SLM",
                height=420,
                elem_classes=["cmp-col"],
            )

        with gr.Row():
            user_in = gr.Textbox(
                label="Message",
                placeholder="e.g. Please help — someone stole my card at the mall. freeze it immediately.",
                lines=2,
                scale=5,
                autofocus=True,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear chat")
            gr.Examples(
                examples=[
                    ["Please help — someone stole my card at the mall. freeze it immediately."],
                    ["Hello, Accidental block on card 3294. Restore access. Thanks."],
                    ["Please help — I paid ₹4,500 yesterday — has it posted to my card? Regards."],
                    ["I want to raise my credit limit to ₹2 lakh."],
                ],
                inputs=[user_in],
            )

        def refresh_adapters() -> dict[str, Any]:
            opts = _finetuned_choices(finetuned_root)
            val = opts[0] if opts else None
            return gr.update(choices=opts, value=val)

        def load_models(ft_path: str, instr: str):
            """Load models and clear polluted chat history."""
            global _SESSION
            if not ft_path:
                return "⚠️ Select a fine-tuned adapter path first.", gr.update(), gr.update()
            path = Path(ft_path)
            if not path.exists():
                return f"⚠️ Path not found: {ft_path}", gr.update(), gr.update()
            try:
                if _SESSION is not None:
                    _SESSION.unload()
                _SESSION = session_from_configs(
                    finetuned_model_dir=ft_path,
                    model_config=model_config,
                    instruction=instr or DEFAULT_INSTRUCTION,
                )
                msg = _SESSION.load()
                # Clear chats so prior "Load models first" turns never enter the prompt.
                return f"✅ {msg}", [], []
            except Exception as exc:
                logger.exception("Failed to load models")
                _SESSION = None
                return f"❌ Load failed: {exc}", gr.update(), gr.update()

        def respond(
            message: str,
            base_hist: list,
            ft_hist: list,
            instr: str,
            mode: str,
            max_tokens: float,
            temp: float,
            sample: bool,
            multi_turn: bool,
        ):
            global _SESSION
            message = (message or "").strip()
            if not message:
                return base_hist, ft_hist, ""
            if _SESSION is None or not _SESSION.loaded:
                # Do not append into chat history — that was poisoning later prompts.
                gr.Warning("Load models first (click Load models), then send again.")
                return base_hist, ft_hist, message

            _SESSION.instruction = instr or DEFAULT_INSTRUCTION
            paired = _history_pairs(base_hist) if multi_turn else None

            try:
                result = _SESSION.generate_pair(
                    message,
                    history=paired,
                    max_new_tokens=int(max_tokens),
                    do_sample=bool(sample),
                    temperature=float(temp),
                    prompt_mode=mode,
                    use_history=bool(multi_turn),
                )
                base_reply = _fmt_reply(result["base"], result["base_latency_sec"])
                ft_reply = _fmt_reply(result["finetuned"], result["finetuned_latency_sec"])
            except Exception as exc:
                logger.exception("Generation failed")
                base_reply = ft_reply = f"❌ {exc}"

            base_hist = list(base_hist or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": base_reply},
            ]
            ft_hist = list(ft_hist or []) + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ft_reply},
            ]
            return base_hist, ft_hist, ""

        def clear_chats():
            return [], [], ""

        refresh_btn.click(refresh_adapters, outputs=[ft_dropdown])
        load_btn.click(
            load_models,
            inputs=[ft_dropdown, instruction],
            outputs=[status, base_chat, ft_chat],
        )
        gen_inputs = [
            user_in,
            base_chat,
            ft_chat,
            instruction,
            prompt_mode,
            max_new_tokens,
            temperature,
            do_sample,
            use_history,
        ]
        send_btn.click(respond, inputs=gen_inputs, outputs=[base_chat, ft_chat, user_in])
        user_in.submit(respond, inputs=gen_inputs, outputs=[base_chat, ft_chat, user_in])
        clear_btn.click(clear_chats, outputs=[base_chat, ft_chat, user_in])

    return demo


def launch_compare_chat(
    *,
    model_config: str = "configs/model/default.yaml",
    finetuned_model_dir: str | None = None,
    host: str = "127.0.0.1",
    port: int = 7860,
    share: bool = False,
) -> None:
    demo = build_app(
        model_config=model_config,
        finetuned_model_dir=finetuned_model_dir,
    )
    demo.queue().launch(server_name=host, server_port=port, share=share, css=_CSS)
