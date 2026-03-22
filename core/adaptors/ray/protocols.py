from typing import Protocol
from ray.rllib.utils.typing import ResultDict
from core.loggers.results import (
    LearnerStats,
    PolicyResult,
)


class PolicyResultMapper(Protocol):
    def __call__(self, result: ResultDict) -> PolicyResult: ...


def from_old_api(result: ResultDict) -> PolicyResult:
    pass


def from_new_api(result: ResultDict) -> PolicyResult:
    policy_result: PolicyResult = {}

    learners = result.get("learners", {}) or {}
    for policy_id, data in learners.items():
        if policy_id == "__all_modules__":
            continue
        learner_stats = LearnerStats(
            batch_size=float(data.get("module_train_batch_size_mean")),
            total_loss=float(data.get("total_loss")),
            residual_variance=None,
            sample_staleness=float(data.get("diff_num_grad_updates_vs_sampler_policy")),
            policy_loss=float(data.get("policy_loss")),
            policy_entropy=float(data.get("entropy")),
            policy_entropy_coeff=float(data.get("curr_entropy_coeff")),
            policy_relative_entropy=None,  # There is no max entropy over continous actions
            policy_kl=float(data.get("mean_kl_loss")),
            policy_kl_coeff=float(data.get("mean_kl_loss")),
            value_loss=float(data.get("vf_loss")),
            value_mean=None,
            value_target=None,
            gradient_norm=float(data.get("gradients_default_optimizer_global_norm")),
            gradient_noise=None,
        )
        policy_result[policy_id] = learner_stats
    return policy_result
