"""POMDP trajectory data collector for the SUMO highway environment.

저장 단위는 POMDP transition 하나다:
    (o_t, a_t, r_t, o_{t+1}, terminated, truncated, s_t, s_{t+1})

- o_t / o_{t+1}: agent가 실제로 보는 부분관측 (19차원)
- a_t: [가감속 raw, 차선변경 raw]
- lane_change: 실제 환경에 적용된 {-1, 0, +1}
- s_t / s_{t+1}: SUMO 내부의 privileged simulator state.
  정책 입력에는 쓰지 않고 데이터 검증/representation/belief 학습용으로만 저장한다.

예시:
    python collect_pomdp_data.py --episodes 100
    python collect_pomdp_data.py --episodes 10 --gui
    python collect_pomdp_data.py --policy keep-lane --episodes 20

출력:
    data/pomdp_YYYYMMDD_HHMMSS.jsonl
    data/pomdp_YYYYMMDD_HHMMSS.npz
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np

from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env import road_builder
from env.sumo_env import SumoHighwayEnv
from train import REWARD  # 학습과 동일한 보상 조건으로 수집

BASE = os.path.dirname(os.path.abspath(__file__))


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
    parser.add_argument("--policy", choices=["random", "keep-lane"], default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gui", action="store_true",
                        help="SUMO GUI로 수집. 기본은 headless")
    parser.add_argument("--out-dir", default=os.path.join(BASE, "data"))
    parser.add_argument("--name", default=None,
                        help="출력 basename. 생략 시 날짜/시각 자동 생성")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    name = args.name or datetime.now().strftime("pomdp_%Y%m%d_%H%M%S")
    jsonl_path = os.path.join(args.out_dir, name + ".jsonl")
    npz_path = os.path.join(args.out_dir, name + ".npz")

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
                    action = behavior_action(args.policy, rng)
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
        detected_xy=np.asarray(detected_xy_buf, dtype=object),
    )

    print("\n수집 완료")
    print(f"  transitions: {total}")
    print(f"  JSONL(full POMDP + privileged state): {jsonl_path}")
    print(f"  NPZ(training-friendly):               {npz_path}")


if __name__ == "__main__":
    main()
