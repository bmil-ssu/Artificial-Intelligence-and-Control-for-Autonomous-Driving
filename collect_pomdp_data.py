"""POMDP trajectory data collector for the SUMO highway environment.

저장 단위는 POMDP transition 하나다:
    (o_t, a_t, r_t, o_{t+1}, terminated, truncated, s_t, s_{t+1})

- o_t / o_{t+1}: agent가 실제로 보는 부분관측
- a_t: [가감속 raw, 차선변경 raw]
- lane_change: 양자화된 요청 명령 {-1, 0, +1} (실행 여부는 별도 저장)
- s_t / s_{t+1}: SUMO 내부의 privileged simulator state.
  정책 입력에는 쓰지 않고 데이터 검증/representation/belief 학습용으로만 저장한다.

예시:
    python collect_pomdp_data.py --model results/run_example/model.pt --episodes 100
    python collect_pomdp_data.py --model results/run_example/model.pt --episodes 10 --gui
    python collect_pomdp_data.py --policy keep-lane --episodes 20

출력:
    data/pomdp_YYYYMMDD_HHMMSS.jsonl
    data/pomdp_YYYYMMDD_HHMMSS.npz
"""
import argparse
import json
import os
import hashlib
from datetime import datetime

import numpy as np
import torch

from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env import road_builder
from env.sumo_env import SumoHighwayEnv
from train import REWARD, HPARAMS  # PPO 네트워크 설정과 보상
from algorithms.ppo import PPO
from algorithms.bc import BCPolicy

BASE = os.path.dirname(os.path.abspath(__file__))


def load_policy(path, algorithm, obs_dim, act_dim, seed):
    """BC는 저장된 구조, 기존 PPO는 train.py의 구조 설정으로 복원한다."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    detected = "bc" if checkpoint.get("algorithm") == "bc" else "ppo"
    if algorithm != "auto" and algorithm != detected:
        raise ValueError(f"지정한 알고리즘 {algorithm}과 체크포인트 {detected}가 다릅니다.")
    if detected == "bc":
        policy = BCPolicy.load(path)
        if policy.state_dim != obs_dim or policy.action_dim != act_dim:
            raise ValueError(f"BC 모델 입출력 ({policy.state_dim}, {policy.action_dim})과 "
                             f"현재 환경 ({obs_dim}, {act_dim}) 차원을 확인하세요.")
    else:
        model_obs_dim = checkpoint["trunk.0.weight"].shape[1]
        if model_obs_dim != obs_dim:
            raise ValueError(
                f"PPO 모델 관측은 {model_obs_dim}차원, 현재 환경은 {obs_dim}차원입니다. "
                "모델 학습 당시의 환경 설정을 복원하거나 현재 환경으로 학습한 모델을 지정하세요.")
        policy = PPO(obs_dim=obs_dim, act_dim=act_dim,
                     **{**HPARAMS, "device": "cpu", "seed": seed})
        try:
            policy.policy.load_state_dict(checkpoint)
        except RuntimeError as exc:
            raise ValueError("PPO 모델 구조가 train.py의 HPARAMS와 다릅니다. 학습 당시 설정을 사용하세요.") from exc
        policy.policy.eval()
    return policy, detected


def behavior_action(policy: str, rng: np.random.Generator) -> np.ndarray:
    """데이터 수집용 behavior policy.

    random:
      - 가감속은 연속 uniform[-1,1]
      - 차선변경은 {-1,0,+1}에서 직접 샘플링 (유지는 더 자주)
    keep-lane:
      - 완만한 가속 + 차선 유지. GUI/파이프라인 점검용.
    """
    if policy == "keep-lane":
        return np.asarray([0.25, 0.0], dtype=np.float32)

    accel = rng.uniform(-1.0, 1.0)
    lane = rng.choice(np.asarray([-1.0, 0.0, 1.0]), p=[0.15, 0.70, 0.15])
    return np.asarray([accel, lane], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="SUMO POMDP trajectory collector")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--policy", choices=["model", "random", "keep-lane"], default="model")
    parser.add_argument("--model", help="수집에 사용할 학습된 model.pt 경로 (기본 정책에서 필수)")
    parser.add_argument("--algorithm", choices=["auto", "ppo", "bc"], default="auto")
    parser.add_argument("--stochastic", action="store_true",
                        help="모델 행동을 확률적으로 샘플링. 기본은 결정적 행동")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true",
                        help="SUMO GUI로 수집. 기본은 headless")
    parser.add_argument("--out-dir", default=os.path.join(BASE, "data"))
    parser.add_argument("--name", default=None,
                        help="출력 basename. 생략 시 날짜/시각 자동 생성")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes는 1 이상이어야 합니다.")
    if args.policy == "model" and not args.model:
        parser.error("학습된 모델 수집에는 --model results/.../model.pt가 필요합니다.")
    if args.policy != "model" and (args.model or args.stochastic or args.algorithm != "auto"):
        parser.error("--model/--stochastic/--algorithm은 --policy model에서만 사용하세요.")
    if args.model:
        args.model = os.path.abspath(args.model)
        if not os.path.isfile(args.model):
            parser.error(f"모델 파일이 없습니다: {args.model}")

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or datetime.now().strftime("pomdp_%Y%m%d_%H%M%S_%f")
    jsonl_path = os.path.join(args.out_dir, name + ".jsonl")
    npz_path = os.path.join(args.out_dir, name + ".npz")
    metadata_path = os.path.join(args.out_dir, name + ".metadata.json")
    if any(os.path.exists(p) for p in (jsonl_path, npz_path, metadata_path)):
        parser.error("같은 이름의 수집 결과가 있습니다. 다른 --name을 사용하세요.")

    rng = np.random.default_rng(args.seed)
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC,
                                 os.path.join(BASE, "env", "sumo"))
    env = SumoHighwayEnv(
        cfg_path=sumocfg,
        road=ROAD, ego=EGO,
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD,
        traffic=TRAFFIC,
        gui=args.gui,
        gui_autostart=True,
    )

    # NPZ는 학습에서 바로 쓰기 좋은 고정 크기 값만 모은다.
    obs_buf, next_obs_buf, action_raw_buf = [], [], []
    lane_cmd_buf, reward_buf = [], []
    term_buf, trunc_buf, episode_buf, step_buf = [], [], [], []
    detected_xy_buf = []   # 스텝별 감지 차량 (x,y) — 가변 길이 object 배열로 저장

    total = 0
    try:
        policy, algorithm = (load_policy(args.model, args.algorithm,
                            env.observation_space.shape[0], env.action_space.shape[0], args.seed)
                            if args.policy == "model" else (None, args.policy))
        torch.manual_seed(args.seed)
        metadata = {"policy": args.policy, "algorithm": algorithm, "model": args.model,
                    "deterministic": not args.stochastic, "seed": args.seed,
                    "episodes": args.episodes, "observation_dim": env.observation_space.shape[0],
                    "road": ROAD, "ego": EGO, "traffic": TRAFFIC,
                    "simulation": SIMULATION, "observation": OBSERVATION,
                    "action": ACTION, "reward": REWARD}
        if args.model:
            with open(args.model, "rb") as model_file:
                metadata["model_sha256"] = hashlib.sha256(model_file.read()).hexdigest()
        with open(metadata_path, "w", encoding="utf-8") as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)
        print(f"수집 정책: {algorithm} | 모델: {args.model} | deterministic={not args.stochastic}")
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for ep in range(args.episodes):
                obs, _ = env.reset(seed=args.seed + ep)
                done = False
                t = 0
                ep_return = 0.0

                while not done:
                    state_t = env.get_privileged_state()
                    # 이번 관측 시점에 시야(W) 안에 들어온 차량들의 좌표.
                    # 관측 벡터와 별도로 저장하는 부가 데이터 (분석/시각화용).
                    detected = env.get_detected_vehicles()
                    action = (policy.predict(obs, deterministic=not args.stochastic)
                              if policy is not None else behavior_action(args.policy, rng))
                    action = np.asarray(action, dtype=np.float32)
                    if action.shape != (2,) or not np.isfinite(action).all():
                        raise ValueError("정책 행동은 유한한 (2,) 배열이어야 합니다.")
                    # 환경에 실제 전달하는 raw 행동을 저장 (확률적 PPO는 범위를 넘을 수 있음).
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    state_tp1 = env.get_privileged_state()

                    record = {
                        "episode": ep,
                        "t": t,
                        "observation": np.asarray(obs, dtype=float).tolist(),
                        "action_raw": np.asarray(action, dtype=float).tolist(),
                        "action": {
                            "accel_raw": float(info["accel_raw"]),
                            "lane_change_raw": float(info["lane_change_raw"]),
                            "lane_change": int(info["lane_change"]),
                            "lane_change_applied": bool(info["lane_change_applied"]),
                        },
                        "reward": float(reward),
                        "next_observation": np.asarray(next_obs, dtype=float).tolist(),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "state": state_t,
                        "next_state": state_tp1,
                        # 감지 차량 좌표 (관측과 같은 시점 t 기준)
                        "detected_vehicles": detected,
                        "event": {
                            "collided": bool(info["collided"]),
                            "arrived": bool(info["arrived"]),
                            "lane_before": info["lane_before"],
                            "lane_after": info["lane_after"],
                        },
                    }
                    jf.write(json.dumps(record, ensure_ascii=False) + "\n")

                    obs_buf.append(np.asarray(obs, dtype=np.float32))
                    next_obs_buf.append(np.asarray(next_obs, dtype=np.float32))
                    action_raw_buf.append(np.asarray(action, dtype=np.float32))
                    lane_cmd_buf.append(int(info["lane_change"]))
                    reward_buf.append(float(reward))
                    term_buf.append(bool(terminated))
                    trunc_buf.append(bool(truncated))
                    episode_buf.append(ep)
                    step_buf.append(t)
                    # (x, y) 좌표만 뽑아 (N_t, 2) 배열로 — 스텝마다 감지 수가
                    # 달라 가변 길이이므로 npz에는 object 배열로 저장된다.
                    detected_xy_buf.append(
                        np.asarray([[d["x"], d["y"]] for d in detected],
                                   dtype=np.float32).reshape(-1, 2))

                    total += 1
                    ep_return += reward
                    t += 1
                    obs = next_obs
                    done = terminated or truncated

                print(f"episode {ep + 1:4d}/{args.episodes} | "
                      f"steps={t:4d} return={ep_return:8.3f}")
    finally:
        env.close()

    detected_xy = np.empty(len(detected_xy_buf), dtype=object)
    detected_xy[:] = detected_xy_buf
    np.savez_compressed(
        npz_path,
        observation=np.asarray(obs_buf, dtype=np.float32),
        action_raw=np.asarray(action_raw_buf, dtype=np.float32),
        lane_change=np.asarray(lane_cmd_buf, dtype=np.int8),
        reward=np.asarray(reward_buf, dtype=np.float32),
        next_observation=np.asarray(next_obs_buf, dtype=np.float32),
        terminated=np.asarray(term_buf, dtype=np.bool_),
        truncated=np.asarray(trunc_buf, dtype=np.bool_),
        episode=np.asarray(episode_buf, dtype=np.int32),
        t=np.asarray(step_buf, dtype=np.int32),
        # 스텝별 감지 차량 (x,y): 길이 N인 object 배열, 원소는 (N_t, 2) float32.
        # 읽을 때: np.load(path, allow_pickle=True)["detected_xy"][i]
        detected_xy=detected_xy,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )

    print("\n수집 완료")
    print(f"  transitions: {total}")
    print(f"  JSONL(full POMDP + privileged state): {jsonl_path}")
    print(f"  NPZ(training-friendly):               {npz_path}")
    print(f"  수집 정책/설정:                        {metadata_path}")


if __name__ == "__main__":
    main()
