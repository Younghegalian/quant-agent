import numpy as np
import torch


def preprocess_state(state, normalize=True):
    """
    상태 벡터 전처리
    - 입력: dict 또는 ndarray
    - 출력: torch.Tensor (1, T, F)
    """
    if isinstance(state, dict):
        state = np.array(list(state.values()), dtype=np.float32)

    if normalize:
        state = (state - np.mean(state)) / (np.std(state) + 1e-8)

    if state.ndim == 1:
        state = np.expand_dims(state, axis=0)  # (1, F)
    return torch.tensor(state, dtype=torch.float32)