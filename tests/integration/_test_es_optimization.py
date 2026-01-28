from core.world.base import World
from src.es.config import ESConfig

# TODO move dummy classes
from tests.integration._test_framework import (
    DummyContextWrapper,
    DummyEnv,
    DummyOptimizerConfig,
)


# @pytest.mark.integration
def test_es_optimizer_integration():
    world = World()

    # Shouldn't ES get its own environment ?
    # which stores the state and the action ?
    # No the ES algorithm should be decoupled from the algorithm
    wrapped_env = DummyContextWrapper(env=DummyEnv, world=world)

    child_cfg = DummyOptimizerConfig().environment(wrapped_env)
    child = child_cfg.build_optimizer()
    child.set_id("child")
    world.register_optimizer(child)

    # This is a problem now because we dont necessarily want es to be bilevel
    # actually its almost as if we need a wrapper for an optimizer to ingect inner optimizer
    # TODO must rethink the core design
    # TODO specify the parameters that need to exist in the config
    es_opt = (
        ESConfig()
        .training()
        .environment()
        .evaluation()
        .reporting()
        .checkpointing()
        .fault_tolerance()
        .experimental()
    )
    es_opt.set_downstream(opt=child)

    # TODO this will quickly lead into code duplication
    # TODO there is the optimizer algorithm itself and then there is the child loop downstream.

    # downstream algorithm may run before or after.
    # generally downstream algorithm may retreive or publish context to world
    # perhaps the builder should take care of the responsability of assembling the optimizer
    # we should have an orchestrator that takes care of connecting the optimizers and
    # and passing context between them
    # optimizer should not know about another optimizer. an optimizer only
    # knows how to pubish and receive context from the world.
    # but this is handled to world
    # what the optimizer does it - this is hardcoded into each optimizer.
    # the config builds the optimization

    # TODO we use callbacks to define exactly how rewards what contexty and when is written into the world
    # TODO the worker handles how and when contexts are written
    # TODO Env creator builds an envrunner with an env that we specify and the worker uses is to step the environment
    # TODO the only thing im still unsure about is how the worker will run downstream optimizers
    # TODO why not have used ray callbacks to build meta-optimizers.?
    # TODO Execution module still required - rollout and train ops. however to really generalize, these ops need to treat an optimizer like a black box
    """
    while not meta_optimizer.converged():
    # 1. Meta proposes a candidate
    candidate = meta_optimizer.sample()

    # 2. Launch inner execution
    result = executor.run_execution(
        optimizer=inner_optimizer,
        params=candidate,
        budget=execution_budget,
    )

    # 3. Meta update based on result
    meta_optimizer.update(
        candidate=candidate,
        outcome=result.metrics
    )

    """


# @pytest.mark.integration
# def test_ppo_regulator_cartpole():
#     world = World()

#     cfg = (
#         PPOptimizerConfig()
#         .environment(env="CartPole-v1")
#         .framework(framework="torch")
#         .resources(num_gpus=0)
#         .training(train_batch_size=200, gamma=0.99)
#     )

#     ppo = cfg.build_optimizer()

#     env = DummyRegulatorEnv(world, ppo, iters=2)

#     for _ in range(3):
#         env.step()

#     ctxs = world.get_ctx_ids()
#     assert len(ctxs) > 0
