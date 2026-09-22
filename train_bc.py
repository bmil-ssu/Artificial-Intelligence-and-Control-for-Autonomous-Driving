"""collect_pomdp_data.py로 수집한 NPZ/JSONL 데이터로 Behavior Cloning 학습.

사용 예시:
    python train_bc.py --data data/pomdp_example.npz
    tensorboard --logdir results
    브라우저에서 http://localhost:6006 접속

학습 대상:
    observation -> action_raw

BCPolicy는 algorithms/bc.py에 정의된
state_dim -> 256 -> 256 -> action_dim MLP를 사용하며,
전체 action vector에 대해 MSE loss로 학습함.

TensorBoard 이벤트 파일은 model.pt와 같은 run 폴더(results/bc_월일_시분/)에
바로 저장되므로, train.py의 run_* 결과와 나란히 비교할 수 있다.
"""

import argparse
import csv
import json
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from algorithms.bc import BCPolicy


BASE = Path(__file__).resolve().parent


def load_data(path):
    """NPZ 또는 JSONL에서 observation, action_raw, episode을 불러옴."""
    path = Path(path)

    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            required = ("observation", "action_raw", "episode")
            missing = [key for key in required if key not in data]

            if missing:
                raise ValueError(
                    f"필수 데이터가 없습니다: {missing}. "
                    f"필요한 key = {required}"
                )

            obs = data["observation"]
            actions = data["action_raw"]
            episodes = data["episode"]

    elif path.suffix == ".jsonl":
        obs, actions, episodes = [], [], []

        with path.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue

                try:
                    row = json.loads(line)
                    obs.append(row["observation"])
                    actions.append(row["action_raw"])
                    episodes.append(row["episode"])
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{path}:{line_no}: 잘못된 transition"
                    ) from exc

    else:
        raise ValueError("데이터는 .npz 또는 .jsonl이어야 합니다.")

    obs = np.asarray(obs, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    episodes = np.asarray(episodes)

    n = len(obs)

    if n == 0:
        raise ValueError("데이터가 비어 있습니다.")

    if obs.ndim != 2 or obs.shape[1] == 0:
        raise ValueError(
            f"observation shape은 (N, state_dim)이어야 합니다. 현재 shape={obs.shape}"
        )

    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(
            f"action_raw shape은 (N, 2)이어야 합니다. 현재 shape={actions.shape}"
        )

    if len(actions) != n or episodes.shape != (n,):
        raise ValueError(
            "observation, action_raw, episode의 sample 수가 서로 다릅니다."
        )

    if not np.isfinite(obs).all():
        raise ValueError("observation에 NaN 또는 Inf가 있습니다.")

    if not np.isfinite(actions).all():
        raise ValueError("action_raw에 NaN 또는 Inf가 있습니다.")

    if not np.issubdtype(episodes.dtype, np.number) or not np.isfinite(episodes).all():
        raise ValueError("episode은 유한한 숫자 ID여야 합니다.")
    if not np.equal(episodes, np.floor(episodes)).all():
        raise ValueError("episode ID는 정수여야 합니다.")
    if (np.abs(actions) > 1).any():
        raise ValueError("action_raw는 환경에 전달된 [-1, 1] 범위의 행동이어야 합니다.")

    return obs, actions, episodes


def split_episodes(episodes, val_fraction, seed):
    """같은 episode의 transition이 train/validation에 동시에 들어가지 않도록 분리함."""
    unique_episodes = np.unique(episodes)

    if len(unique_episodes) < 2:
        raise ValueError(
            "episode 단위 train/validation 분리를 위해 최소 2개 episode가 필요합니다."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_episodes)

    n_val = max(1, round(len(unique_episodes) * val_fraction))
    n_val = min(n_val, len(unique_episodes) - 1)

    val_episodes = shuffled[:n_val]
    val_mask = np.isin(episodes, val_episodes)

    train_idx = np.flatnonzero(~val_mask)
    val_idx = np.flatnonzero(val_mask)

    return train_idx, val_idx


def make_loader(obs, actions, indices, batch_size, shuffle):
    """Numpy array를 PyTorch DataLoader로 변환함."""
    dataset = TensorDataset(
        torch.from_numpy(obs[indices]),
        torch.from_numpy(actions[indices]),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


def run_epoch(model, loader, device, optimizer=None):
    """한 epoch 학습 또는 validation을 수행함."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for state, target_action in loader:
            state = state.to(device)
            target_action = target_action.to(device)

            pred_action = model(state)

            # 기존 BC 코드와 동일하게 전체 action vector를 MSE로 학습함.
            loss = F.mse_loss(pred_action, target_action)

            if not torch.isfinite(loss):
                raise ValueError(
                    "학습 loss가 NaN 또는 Inf입니다. "
                    "데이터 값 또는 learning rate를 확인하세요."
                )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = state.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def save_model(model, path):
    """BCPolicy.load()와 호환되도록 구조 정보와 정규화 통계도 저장한다."""
    model.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help="collect_pomdp_data.py로 생성한 .npz 또는 .jsonl 파일",
    )

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="학습 결과 저장 폴더",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs는 1 이상이어야 합니다.")

    if args.batch_size < 1:
        parser.error("--batch-size는 1 이상이어야 합니다.")

    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr은 양수여야 합니다.")

    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction은 0과 1 사이여야 합니다.")

    # --help와 데이터 유틸 함수는 TensorBoard 없이도 쓸 수 있도록 여기서 import한다.
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        parser.error(f"TensorBoard를 불러올 수 없습니다: {exc}. "
                     "현재 Python 환경에서 python -m pip install tensorboard를 실행하세요.")

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device 선택
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Dataset
    obs, actions, episodes = load_data(args.data)

    state_dim = obs.shape[1]
    action_dim = actions.shape[1]

    train_idx, val_idx = split_episodes(
        episodes,
        args.val_fraction,
        args.seed,
    )

    train_loader = make_loader(
        obs,
        actions,
        train_idx,
        args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        obs,
        actions,
        val_idx,
        args.batch_size,
        shuffle=False,
    )

    # Model
    model = BCPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
    ).to(device)

    # 검증 데이터가 통계에 섞이지 않도록 학습 관측만으로 정규화한다.
    # 상수 항목은 표준편차를 1로 두어 검증/추론 입력의 폭증을 방지한다.
    obs_mean = obs[train_idx].mean(axis=0)
    obs_std = obs[train_idx].std(axis=0)
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
    model.obs_mean.copy_(torch.as_tensor(obs_mean, device=device))
    model.obs_std.copy_(torch.as_tensor(obs_std, device=device))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    # 결과 폴더 — 짧고 읽기 쉬운 이름: bc_월일_시분 (예: bc_0922_1145)
    # 같은 분(minute)에 여러 번 실행하면 _2, _3 … 접미사가 자동으로 붙는다.
    if args.out_dir is None:
        stem = datetime.now().strftime("bc_%m%d_%H%M")
        out_dir = BASE / "results" / stem
        k = 2
        while out_dir.exists():
            out_dir = BASE / "results" / f"{stem}_{k}"
            k += 1
    else:
        out_dir = args.out_dir

    out_dir.mkdir(parents=True, exist_ok=False)

    # 설정 저장
    config = {
        "data": str(args.data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_fraction": args.val_fraction,
        "seed": args.seed,
        "device": device,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "train_episodes": np.unique(episodes[train_idx]).tolist(),
        "val_episodes": np.unique(episodes[val_idx]).tolist(),
    }

    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("Behavior Cloning Training")
    print("=" * 70)
    print(f"data          : {args.data}")
    print(f"device        : {device}")
    print(f"state_dim     : {state_dim}")
    print(f"action_dim    : {action_dim}")
    print(f"train samples : {len(train_idx)}")
    print(f"val samples   : {len(val_idx)}")
    print(f"output        : {out_dir}")
    print(f"tensorboard   : tensorboard --logdir \"{out_dir.parent}\"  → http://localhost:6006")
    print("=" * 70)

    best_val_loss = float("inf")

    log_path = out_dir / "training_log.csv"

    # TensorBoard 이벤트는 model.pt와 같은 폴더(out_dir)에 바로 쓴다 (train.py와 동일한 구조).
    # closing(): 오류나 Ctrl+C로 종료되어도 대기 중인 이벤트를 파일에 저장한다.
    with closing(SummaryWriter(log_dir=str(out_dir))) as tb_writer, log_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        tb_writer.add_text("config", "```json\n" + json.dumps(config, indent=2) + "\n```", 0)

        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
            ],
        )

        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
            )

            val_loss = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
            )

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            stream.flush()

            # validation loss가 가장 낮은 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_model(
                    model,
                    out_dir / "model.pt",
                )

            # x축은 epoch. CSV와 동일한 값을 기록한다.
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/validation", val_loss, epoch)
            tb_writer.add_scalar("Loss/best_validation", best_val_loss, epoch)
            tb_writer.add_scalar("Optimization/learning_rate", optimizer.param_groups[0]["lr"], epoch)
            # 학습 중에도 브라우저에서 완료된 epoch 결과를 바로 볼 수 있게 한다.
            tb_writer.flush()

            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

    # 마지막 epoch 모델 저장
    save_model(
        model,
        out_dir / "last_model.pt",
    )

    print("=" * 70)
    print("Training finished")
    print(f"Best validation loss : {best_val_loss:.6f}")
    print(f"Best model           : {out_dir / 'model.pt'}")
    print(f"Last model           : {out_dir / 'last_model.pt'}")
    print(f"Training log         : {log_path}")
    print(f"Evaluation           : python test.py {out_dir / 'model.pt'} --algorithm bc --nogui")
    print(f"TensorBoard          : tensorboard --logdir \"{out_dir.parent}\"  → http://localhost:6006")
    print("=" * 70)


if __name__ == "__main__":
    main()
