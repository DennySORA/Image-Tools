"""
模型載入相容性修補模組

修補 transformers >= 5.0 與自訂模型程式碼（如 BiRefNet）的相容性問題：
1. get_init_context 無條件使用 torch.device("meta")，
   但自訂模型在 __init__ 中呼叫 .item()，meta tensor 不支援。
2. 自訂模型未呼叫 post_init()，導致 all_tied_weights_keys 未設定。
"""

import logging
import os
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
import transformers
from transformers import AutoModelForImageSegmentation
from transformers.modeling_utils import PreTrainedModel


@contextmanager
def _suppress_loading_noise() -> Iterator[None]:
    """暫時抑制模型載入期間的冗餘輸出（httpx、transformers、timm、tqdm）"""
    # 保存原始狀態
    orig_verbosity = transformers.logging.get_verbosity()
    httpx_logger = logging.getLogger("httpx")
    orig_httpx_level = httpx_logger.level
    hf_logger = logging.getLogger("huggingface_hub")
    orig_hf_level = hf_logger.level
    orig_tqdm_disable = os.environ.get("TQDM_DISABLE")

    # 抑制: transformers 日誌、httpx HTTP 請求、huggingface_hub、tqdm 進度條
    transformers.logging.set_verbosity_error()  # type: ignore[no-untyped-call]
    httpx_logger.setLevel(logging.WARNING)
    hf_logger.setLevel(logging.WARNING)
    os.environ["TQDM_DISABLE"] = "1"

    # 抑制: timm FutureWarning
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
        try:
            yield
        finally:
            transformers.logging.set_verbosity(orig_verbosity)  # type: ignore[no-untyped-call]
            httpx_logger.setLevel(orig_httpx_level)
            hf_logger.setLevel(orig_hf_level)
            if orig_tqdm_disable is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = orig_tqdm_disable


def load_pretrained_no_meta(model_name: str) -> Any:
    """
    載入預訓練 ImageSegmentation 模型，修補 meta device 相容性問題

    自動抑制載入期間的冗餘輸出（HTTP 請求日誌、進度條、FutureWarning 等）

    Args:
        model_name: HuggingFace 模型名稱

    Returns:
        載入完成的模型
    """

    # 修補 1: 移除 meta device context
    orig_context = PreTrainedModel.__dict__["get_init_context"]

    @classmethod  # type: ignore[misc]
    def _safe_context(
        cls: type,
        dtype: torch.dtype,
        is_quantized: bool,
        _is_ds_init_called: bool,
    ) -> list[Any]:
        bound_original = orig_context.__get__(None, cls)
        contexts: list[Any] = bound_original(dtype, is_quantized, _is_ds_init_called)
        return [
            c
            for c in contexts
            if not (isinstance(c, torch.device) and c.type == "meta")
        ]

    # 修補 2: 確保 all_tied_weights_keys 存在
    orig_finalize = PreTrainedModel.__dict__["_finalize_model_loading"]

    @classmethod  # type: ignore[misc]
    def _safe_finalize(cls: type, model: Any, *args: Any, **kwargs: Any) -> Any:
        if not hasattr(model, "all_tied_weights_keys"):
            model.all_tied_weights_keys = {}
        return orig_finalize.__get__(None, cls)(model, *args, **kwargs)

    PreTrainedModel.get_init_context = _safe_context  # type: ignore[assignment]
    PreTrainedModel._finalize_model_loading = _safe_finalize  # type: ignore[assignment]
    try:
        with _suppress_loading_noise():
            return AutoModelForImageSegmentation.from_pretrained(
                model_name, trust_remote_code=True
            )
    finally:
        PreTrainedModel.get_init_context = orig_context  # type: ignore[method-assign]
        PreTrainedModel._finalize_model_loading = orig_finalize  # type: ignore[method-assign]
