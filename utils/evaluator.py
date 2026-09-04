"""정책 평가 유틸 — 주행 지표 측정.

알고리즘과 분리해둔 이유: 평가는 "정책이 무엇으로 학습됐는지"와 무관하다.
`predict(obs, deterministic=True) -> action` 메서드만 있으면
PPO든 SAC든, 심지어 규칙 기반 컨트롤러든 똑같이 평가할 수 있다.

왜 별도 평가가 필요한가:
    학습 중 에피소드는 탐험 노이즈(σ)가 섞인 확률적 행동의 결과라
    "지금 정책의 실제 운전 실력"을 정확히 보여주지 못한다.
    그래서 주기적으로 노이즈 없는(deterministic) 정책으로 주행시켜
    충돌률 / 완주율 / 평균속도 / 차간거리를 따로 측정한다.
"""
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class DrivingMetrics:
    """평가 1회(n_episodes 평균)의 주행 지표."""
    collision_rate: float   # 충돌로 끝난 에피소드 비율 (0~1, 낮을수록 좋음)
    success_rate: float     # 도로 끝까지 완주한 비율 (0~1, 높을수록 좋음)
    timeout_rate: float     # 시간초과로 끝난 비율 (= 1 - 충돌 - 완주)
    mean_speed: float       # 전 스텝 평균 주행속도 (m/s)
    mean_gap: float         # 선행차 감지 스텝에서의 평균 차간거리 (m)
    min_gap: float          # 관측된 최소 차간거리 (m) — 위험 운전 지표
    ep_return: float        # 에피소드당 평균 누적 보상
    ep_length: float        # 에피소드당 평균 스텝 수
    lane_changes: float     # 에피소드당 평균 "실행된" 차선변경 횟수
                            # (추월 시나리오에서 정책이 실제로 차선을
                            #  활용하는지 보는 핵심 지표)

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """콘솔 한 줄 출력용 문자열."""
        return (f"충돌률={self.collision_rate:.0%}  "
                f"완주율={self.success_rate:.0%}  "
                f"평균속도={self.mean_speed:.1f}m/s  "
                f"평균차간={self.mean_gap:.1f}m  "
                f"최소차간={self.min_gap:.1f}m  "
                f"차선변경={self.lane_changes:.1f}회  "
                f"보상={self.ep_return:.1f}")


def evaluate_policy(policy, env, n_episodes: int = 5,
                    deterministic: bool = True) -> DrivingMetrics:
    """정책을 n_episodes 주행시키고 주행 지표 평균을 반환.

    Args:
        policy: `predict(obs, deterministic) -> action` 을 가진 객체
                (PPOAgent, 또는 같은 인터페이스의 무엇이든)
        env:    SumoHighwayEnv (mean_gap 기본값용 visibility 속성만 사용)
        n_episodes: 평가 에피소드 수. 많을수록 지표가 안정적이지만 느리다
        deterministic: True면 탐험 노이즈 없이 평균 행동만 사용

    Note:
        모든 지표는 env.step()의 info(collided/arrived/speed/gap)에서 읽는다.
        관측(obs) 벡터의 구성과 무관하므로, 관측을 바꿔도 이 코드는 안 깨진다.
    """
    n_collide, n_arrive, n_timeout = 0, 0, 0
    speeds, gaps, returns, lengths, lane_changes = [], [], [], [], []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done, ep_ret, ep_len = False, 0.0, 0
        ep_lc = 0
        ended_by = None

        while not done:
            action = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += reward
            ep_len += 1
            done = terminated or truncated

            # 실행된 차선변경 집계 (유지 명령의 changeLane 핀 동작은 제외)
            if info.get("lane_change", 0) != 0 and info.get("lane_change_applied"):
                ep_lc += 1

            if info.get("collided"):
                ended_by = "collision"
            elif info.get("arrived"):
                ended_by = "arrival"

            # 속도/차간거리는 env가 info로 넘겨주는 원시값(m/s, m)을 쓴다.
            # 관측(obs) 레이아웃이 바뀌어도 평가 코드는 영향받지 않는다.
            # 종료 스텝은 None (ego가 이미 제거됨) → 통계에서 제외.
            if info.get("speed") is not None:
                speeds.append(float(info["speed"]))
            if info.get("gap") is not None:
                gaps.append(float(info["gap"]))

        if ended_by == "collision":
            n_collide += 1
        elif ended_by == "arrival":
            n_arrive += 1
        else:
            n_timeout += 1

        returns.append(ep_ret)
        lengths.append(ep_len)
        lane_changes.append(ep_lc)

    return DrivingMetrics(
        collision_rate=n_collide / n_episodes,
        success_rate=n_arrive / n_episodes,
        timeout_rate=n_timeout / n_episodes,
        mean_speed=float(np.mean(speeds)) if speeds else 0.0,
        mean_gap=float(np.mean(gaps)) if gaps else float(env.visibility),
        min_gap=float(np.min(gaps)) if gaps else float(env.visibility),
        ep_return=float(np.mean(returns)),
        ep_length=float(np.mean(lengths)),
        lane_changes=float(np.mean(lane_changes)),
    )
