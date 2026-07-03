from dataclasses import asdict
from math import pi

import pytest
import torch

from sakigo_model import (
    BLACK,
    COMPOSE,
    DistillationInputProjection,
    GameStateBatch,
    GROUP_SIZE,
    INVERSE,
    BoardToRegisterAttention,
    InvariantHead,
    KataGoInputProjection,
    KataGoInputs,
    ProjectedModelAdapter,
    RegularGQAAttention,
    RegularLift,
    RegularLinear1x1,
    RegularPointwiseMLP,
    RegularRMSNorm,
    RegisterToBoardAttention,
    SakiGoInference,
    SakiGoInputProjection,
    SakiGoModel,
    SakiGoModelConfig,
    ScalarSakiGoModel,
    TrunkBlock,
    WHITE,
    config_from_checkpoint,
    config_from_spec,
    model_from_spec,
    model_spec_names,
    transform_action_logits,
    transform_board,
    transform_cell,
    transform_policy_logits,
    transform_regular_board,
    transform_regular_registers,
)


torch.manual_seed(7)


def tiny_config(board_size: int = 5) -> SakiGoModelConfig:
    return SakiGoModelConfig(
        board_size=board_size,
        stem_channels=(6, 8, 16),
        rule_mlp_channels=(10, 16, 32),
        block_count=2,
        register_count=2,
        trunk_channels=16,
        bottleneck_channels=8,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        broadcast_blocks=(2,),
        wdl_hidden=4,
        score_hidden=4,
        ownership_hidden=4,
        policy_hidden=4,
        policy_pass_hidden=4,
        budget_hidden=4,
        budget_pass_hidden=4,
    )


def assert_close(actual: torch.Tensor, expected: torch.Tensor, atol: float = 1e-5) -> None:
    torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)


def assert_regular_board_equivariant(module, x: torch.Tensor, atol: float = 1e-5) -> None:
    module.eval()
    with torch.no_grad():
        base = module(x)
        for transform in range(GROUP_SIZE):
            lhs = module(transform_regular_board(x, transform))
            rhs = transform_regular_board(base, transform)
            assert_close(lhs, rhs, atol=atol)


def random_inputs(batch: int = 2, size: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    board = torch.randn(batch, 6, size, size)
    rules = torch.randn(batch, 10)
    return board, rules


def sample_game_state() -> GameStateBatch:
    stones = torch.tensor(
        [
            [
                [BLACK, 0, WHITE],
                [0, BLACK, 0],
                [WHITE, 0, 0],
            ],
            [
                [WHITE, BLACK, 0],
                [0, WHITE, 0],
                [BLACK, 0, 0],
            ],
        ]
    )
    illegal = torch.zeros(2, 3, 3)
    illegal[0, 1, 0] = 1.0
    illegal[1, 2, 2] = 1.0
    return GameStateBatch(
        stones=stones,
        to_move=torch.tensor([BLACK, WHITE]),
        scoring_rule=torch.tensor([0, 3]),
        ko_rule=torch.tensor([0, 1]),
        suicide_rule=torch.tensor([1, 0]),
        komi=torch.tensor([7.5, 6.5]),
        captures=torch.tensor([[2.0, 5.0], [4.0, 1.0]]),
        non_trivial_illegal=illegal,
    )


def test_sakigo_projection_encodes_canonical_game_state() -> None:
    state = sample_game_state()
    projected = SakiGoInputProjection()(state)

    assert projected.board.shape == (2, 6, 3, 3)
    assert projected.rules.shape == (2, 10)
    assert_close(projected.board[0, 0], (state.stones[0] == BLACK).float())
    assert_close(projected.board[0, 1], (state.stones[0] == WHITE).float())
    assert_close(projected.board[1, 0], (state.stones[1] == WHITE).float())
    assert_close(projected.board[1, 1], (state.stones[1] == BLACK).float())
    assert_close(projected.board[:, 2], (state.stones == 0).float())
    assert projected.board[0, 3, 0, 0] == 1.0
    assert projected.board[0, 4, 0, 1] == 1.0
    assert projected.board[0, 4, 0, 0] == 0.0
    assert projected.board[0, 5, 1, 0] == 1.0
    assert projected.board[1, 5, 2, 2] == 1.0

    assert_close(projected.rules[0, :4], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert_close(projected.rules[1, :4], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert_close(projected.rules[0, 4:6], torch.tensor([1.0, 0.0]))
    assert_close(projected.rules[1, 4:6], torch.tensor([0.0, 1.0]))
    assert_close(projected.rules[0, 6:8], torch.tensor([0.0, 1.0]))
    assert_close(projected.rules[1, 6:8], torch.tensor([1.0, 0.0]))
    assert_close(projected.rules[:, 8], torch.tensor([-7.5 / 9.0, 6.5 / 9.0]))
    assert_close(projected.rules[:, 9], torch.tensor([-3.0 / 9.0, -3.0 / 9.0]))


def test_katago_projection_uses_native_features_or_encoder() -> None:
    state = sample_game_state()
    native_spatial = torch.randn(2, 22, 3, 3)
    native_global = torch.randn(2, 19)
    native_state = GameStateBatch(
        stones=state.stones,
        to_move=state.to_move,
        katago_spatial=native_spatial,
        katago_global=native_global,
    )

    native_projected = KataGoInputProjection()(native_state)
    assert native_projected.spatial is native_spatial
    assert native_projected.global_features is native_global

    def encoder(encoded_state: GameStateBatch) -> KataGoInputs:
        assert encoded_state is state
        return KataGoInputs(
            spatial=torch.ones(2, 22, 3, 3),
            global_features=torch.ones(2, 19),
        )

    encoded = KataGoInputProjection(encoder=encoder)(state)
    assert encoded.spatial.shape == (2, 22, 3, 3)
    assert encoded.global_features.shape == (2, 19)
    assert encoded.spatial.sum() == 2 * 22 * 3 * 3


def test_katago_projection_requires_exact_native_projection() -> None:
    with pytest.raises(ValueError, match="exact encoder"):
        KataGoInputProjection()(sample_game_state())


def test_distillation_projection_and_model_adapters_share_state() -> None:
    state = sample_game_state()
    native_spatial = torch.randn(2, 22, 3, 3)
    native_global = torch.randn(2, 19)
    state = GameStateBatch(
        stones=state.stones,
        to_move=state.to_move,
        scoring_rule=state.scoring_rule,
        ko_rule=state.ko_rule,
        suicide_rule=state.suicide_rule,
        komi=state.komi,
        captures=state.captures,
        non_trivial_illegal=state.non_trivial_illegal,
        katago_spatial=native_spatial,
        katago_global=native_global,
    )
    projected = DistillationInputProjection()(state)
    assert projected.student.board.shape == (2, 6, 3, 3)
    assert projected.teacher.spatial is native_spatial

    class Teacher(torch.nn.Module):
        def forward(self, spatial: torch.Tensor, global_features: torch.Tensor) -> torch.Tensor:
            return spatial.mean(dim=(1, 2, 3)) + global_features.mean(dim=1)

    teacher = ProjectedModelAdapter(Teacher(), KataGoInputProjection())
    assert teacher(state).shape == (2,)

    student = ProjectedModelAdapter(
        SakiGoModel(tiny_config(board_size=3)).eval(),
        SakiGoInputProjection(),
    )
    with torch.no_grad():
        output = student(state)
    assert output["policy_logits"].shape == (2, 10)


def test_d4_transform_order_matches_solver_examples() -> None:
    assert [transform_cell(5, 1, transform) for transform in range(GROUP_SIZE)] == [
        1,
        9,
        23,
        15,
        3,
        21,
        5,
        19,
    ]


def test_d4_transform_order_matches_solver_3x3_table() -> None:
    expected = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [2, 5, 8, 1, 4, 7, 0, 3, 6],
        [8, 7, 6, 5, 4, 3, 2, 1, 0],
        [6, 3, 0, 7, 4, 1, 8, 5, 2],
        [2, 1, 0, 5, 4, 3, 8, 7, 6],
        [6, 7, 8, 3, 4, 5, 0, 1, 2],
        [0, 3, 6, 1, 4, 7, 2, 5, 8],
        [8, 5, 2, 7, 4, 1, 6, 3, 0],
    ]
    for transform, cells in enumerate(expected):
        assert [transform_cell(3, cell, transform) for cell in range(9)] == cells


def test_d4_composition_and_inverse_tables() -> None:
    for first in range(GROUP_SIZE):
        for second in range(GROUP_SIZE):
            composed = COMPOSE[first][second]
            for cell in range(25):
                sequential = transform_cell(5, transform_cell(5, cell, second), first)
                assert sequential == transform_cell(5, cell, composed)
        assert COMPOSE[first][INVERSE[first]] == 0
        assert COMPOSE[INVERSE[first]][first] == 0


def test_policy_and_action_logit_transforms() -> None:
    policy = torch.randn(2, 25)
    action = torch.randn(2, 26)
    for transform in range(GROUP_SIZE):
        transformed_policy = transform_policy_logits(policy, transform, 5)
        transformed_action = transform_action_logits(action, transform, 5)
        assert_close(transformed_action[:, :-1], transform_policy_logits(action[:, :-1], transform, 5))
        assert_close(transformed_action[:, -1:], action[:, -1:])
        assert transformed_policy.shape == policy.shape
        assert transformed_action.shape == action.shape


def test_regular_feature_transform_convention() -> None:
    size = 5
    x = torch.arange(2 * 3 * GROUP_SIZE * size * size, dtype=torch.float32).reshape(
        2,
        3,
        GROUP_SIZE,
        size,
        size,
    )
    for transform in range(GROUP_SIZE):
        y = transform_regular_board(x, transform)
        inv = INVERSE[transform]
        for out_component in range(GROUP_SIZE):
            source_component = COMPOSE[inv][out_component]
            for cell in range(size * size):
                source_cell = transform_cell(size, cell, inv)
                assert_close(
                    y[:, :, out_component, cell // size, cell % size],
                    x[:, :, source_component, source_cell // size, source_cell % size],
                )


def test_regular_lift_is_equivariant() -> None:
    module = RegularLift()
    x = torch.randn(2, 6, 5, 5)
    with torch.no_grad():
        base = module(x)
        assert base.shape == (2, 6, GROUP_SIZE, 5, 5)
        for transform in range(GROUP_SIZE):
            lhs = module(transform_board(x, transform))
            rhs = transform_regular_board(base, transform)
            assert_close(lhs, rhs)


def test_regular_layers_are_equivariant() -> None:
    assert_regular_board_equivariant(RegularLinear1x1(3, 5), torch.randn(2, 3, GROUP_SIZE, 5, 5))
    assert_regular_board_equivariant(RegularRMSNorm(4), torch.randn(2, 4, GROUP_SIZE, 5, 5))
    assert_regular_board_equivariant(
        RegularPointwiseMLP((4, 6, 3)),
        torch.randn(2, 4, GROUP_SIZE, 5, 5),
    )


def test_spatial_attention_with_rope_is_equivariant() -> None:
    module = RegularGQAAttention(
        channels=8,
        board_size=5,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(pi / 2,),
    )
    x = torch.randn(2, 8, GROUP_SIZE, 5, 5)
    assert_regular_board_equivariant(module, x, atol=2e-5)


def test_register_gather_and_broadcast_are_equivariant() -> None:
    gather = RegisterToBoardAttention(
        register_channels=8,
        board_channels=8,
        board_size=5,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(pi / 2,),
    )
    broadcast = BoardToRegisterAttention(
        board_channels=8,
        register_channels=8,
        board_size=5,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(pi / 2,),
    )
    registers = torch.randn(2, 2, 8, GROUP_SIZE)
    board = torch.randn(2, 8, GROUP_SIZE, 5, 5)
    with torch.no_grad():
        gathered = gather(registers, board)
        broadcasted = broadcast(board, registers)
        for transform in range(GROUP_SIZE):
            lhs_registers = gather(
                transform_regular_registers(registers, transform),
                transform_regular_board(board, transform),
            )
            lhs_board = broadcast(
                transform_regular_board(board, transform),
                transform_regular_registers(registers, transform),
            )
            assert_close(lhs_registers, transform_regular_registers(gathered, transform), atol=2e-5)
            assert_close(lhs_board, transform_regular_board(broadcasted, transform), atol=2e-5)


def test_trunk_block_is_equivariant() -> None:
    module = TrunkBlock(
        trunk_channels=16,
        bottleneck_channels=8,
        board_size=5,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        global_rope_frequencies=(pi,),
        local_rope_frequencies=(pi / 2,),
        block_count=2,
        eps=1e-6,
        activation="silu",
    )
    board = torch.randn(2, 16, GROUP_SIZE, 5, 5)
    registers = torch.randn(2, 2, 16, GROUP_SIZE)
    module.eval()
    with torch.no_grad():
        base_board, base_registers = module(board, registers)
        for transform in range(GROUP_SIZE):
            lhs_board, lhs_registers = module(
                transform_regular_board(board, transform),
                transform_regular_registers(registers, transform),
            )
            assert_close(lhs_board, transform_regular_board(base_board, transform), atol=3e-5)
            assert_close(
                lhs_registers,
                transform_regular_registers(base_registers, transform),
                atol=3e-5,
            )


def test_invariant_head_collapses_regular_axis_correctly() -> None:
    head = InvariantHead("mean")
    board = torch.randn(2, 3, GROUP_SIZE, 5, 5)
    registers = torch.randn(2, 2, 3, GROUP_SIZE)
    with torch.no_grad():
        base_board = head(board)
        base_registers = head(registers)
        for transform in range(GROUP_SIZE):
            assert_close(head(transform_regular_board(board, transform)), transform_board(base_board, transform))
            assert_close(head(transform_regular_registers(registers, transform)), base_registers)


def test_full_model_shapes_and_equivariance() -> None:
    model = SakiGoModel(tiny_config()).eval()
    board, rules = random_inputs()
    with torch.no_grad():
        base = model(board, rules)
        assert base["wdl_logits"].shape == (2, 3)
        assert base["score"].shape == (2, 1)
        assert base["ownership_logits"].shape == (2, 25)
        assert base["policy_logits"].shape == (2, 26)
        assert base["budget_logits"].shape == (2, 26)
        for value in base.values():
            assert torch.isfinite(value).all().item()

        for transform in range(GROUP_SIZE):
            actual = model(transform_board(board, transform), rules)
            assert_close(actual["wdl_logits"], base["wdl_logits"], atol=3e-5)
            assert_close(actual["score"], base["score"], atol=3e-5)
            assert_close(
                actual["ownership_logits"],
                transform_policy_logits(base["ownership_logits"], transform, 5),
                atol=3e-5,
            )
            assert_close(
                actual["policy_logits"],
                transform_action_logits(base["policy_logits"], transform, 5),
                atol=3e-5,
            )
            assert_close(
                actual["budget_logits"],
                transform_action_logits(base["budget_logits"], transform, 5),
                atol=3e-5,
            )


def test_full_model_accepts_smaller_boards() -> None:
    model = SakiGoModel(tiny_config(board_size=5)).eval()
    with torch.no_grad():
        for size in (3, 4):
            board, rules = random_inputs(size=size)
            output = model(board, rules)
            assert output["ownership_logits"].shape == (2, size * size)
            assert output["policy_logits"].shape == (2, size * size + 1)
            assert output["budget_logits"].shape == (2, size * size + 1)


def test_model_rejects_bad_inputs() -> None:
    model = SakiGoModel(tiny_config()).eval()
    board, rules = random_inputs()
    with pytest.raises(ValueError, match="expected 6 input planes"):
        model(board[:, :5], rules)
    with pytest.raises(ValueError, match="batch sizes must match"):
        model(board, rules[:1])
    with pytest.raises(ValueError, match="expected 10 rule features"):
        model(board, rules[:, :9])


def test_model_specs_build_expected_config() -> None:
    assert model_spec_names() == (
        "model1",
        "model1_control_params",
        "model1_control_compute",
    )
    config = config_from_spec("model1")
    assert config.board_size == 32
    assert config.input_planes == 6
    assert config.rule_dim == 10
    assert config.trunk_channels == 32
    assert config.rule_mlp_channels == (10, 32, 64)
    assert config.policy_pass_outputs == 1


def test_scalar_control_specs_build_models() -> None:
    params_config = config_from_spec("model1_control_params")
    compute_config = config_from_spec("model1_control_compute")
    assert params_config.architecture == "ScalarSakiGoModel"
    assert compute_config.architecture == "ScalarSakiGoModel"
    assert params_config.trunk_channels == 91
    assert compute_config.trunk_channels == 8 * config_from_spec("model1").trunk_channels
    assert compute_config.bottleneck_channels == 8 * config_from_spec("model1").bottleneck_channels
    assert isinstance(model_from_spec("model1"), SakiGoModel)
    assert isinstance(model_from_spec("model1_control_params"), ScalarSakiGoModel)
    assert ScalarSakiGoModel().config.architecture == "ScalarSakiGoModel"


def test_scalar_control_specs_forward() -> None:
    board, rules = random_inputs(batch=1, size=3)
    for name in ("model1_control_params", "model1_control_compute"):
        model = model_from_spec(name).eval()
        with torch.no_grad():
            output = model(board, rules)
        assert output["wdl_logits"].shape == (1, 3)
        assert output["score"].shape == (1, 1)
        assert output["ownership_logits"].shape == (1, 9)
        assert output["policy_logits"].shape == (1, 10)
        assert output["budget_logits"].shape == (1, 10)


def test_scalar_control_forward_shapes() -> None:
    config = SakiGoModelConfig(
        architecture="ScalarSakiGoModel",
        board_size=5,
        stem_channels=(6, 12, 16),
        rule_mlp_channels=(10, 32),
        block_count=2,
        register_count=2,
        trunk_channels=16,
        bottleneck_channels=8,
        q_heads=1,
        kv_heads=1,
        head_dim=8,
        broadcast_blocks=(2,),
        wdl_hidden=4,
        score_hidden=4,
        ownership_hidden=4,
        policy_hidden=4,
        policy_pass_hidden=4,
        budget_hidden=4,
        budget_pass_hidden=4,
    )
    model = ScalarSakiGoModel(config).eval()
    board, rules = random_inputs()
    with torch.no_grad():
        output = model(board, rules)
    assert output["wdl_logits"].shape == (2, 3)
    assert output["score"].shape == (2, 1)
    assert output["ownership_logits"].shape == (2, 25)
    assert output["policy_logits"].shape == (2, 26)
    assert output["budget_logits"].shape == (2, 26)
    assert all(torch.isfinite(value).all().item() for value in output.values())


def test_control_parameter_profile_is_intentional() -> None:
    equivariant = model_from_spec("model1")
    params_control = model_from_spec("model1_control_params")
    compute_control = model_from_spec("model1_control_compute")

    def count(model: torch.nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    equivariant_count = count(equivariant)
    params_ratio = count(params_control) / equivariant_count
    compute_ratio = count(compute_control) / equivariant_count
    assert 0.75 <= params_ratio <= 1.25
    assert compute_ratio > params_ratio


def test_invalid_config_dimension_guards() -> None:
    with pytest.raises(ValueError, match="head_dim"):
        SakiGoModel(tiny_config().__class__(head_dim=4))
    bad = tiny_config()
    bad = SakiGoModelConfig(**{**asdict(bad), "rule_mlp_channels": (10, 8, 8)})
    with pytest.raises(ValueError, match="rule MLP output"):
        SakiGoModel(bad)


def test_checkpoint_config_round_trip() -> None:
    config = tiny_config()
    checkpoint = {"model_config": asdict(config), "config": {"model_board_size": 3}}
    loaded = config_from_checkpoint(checkpoint, minimum_board_size=5)
    assert loaded == config


def test_plain_inference_wrapper_matches_model() -> None:
    model = SakiGoModel(tiny_config()).eval()
    board, rules = random_inputs()
    wrapper = SakiGoInference(model, dtype=torch.float32)
    with torch.no_grad():
        expected = model(board, rules)
    actual = wrapper(board, rules)
    for key in expected:
        assert_close(actual[key], expected[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_bf16_and_graph_inference_smoke() -> None:
    torch.manual_seed(11)
    config = tiny_config(board_size=5)
    model = SakiGoModel(config)
    board, rules = random_inputs(batch=1, size=5)
    cuda_board = board.cuda()
    cuda_rules = rules.cuda()
    fp32 = SakiGoInference(model, device="cuda", dtype=torch.float32)
    fp32_out = fp32(cuda_board, cuda_rules)
    assert all(torch.isfinite(value).all().item() for value in fp32_out.values())

    graph_model = SakiGoModel(config)
    graph_model.load_state_dict(fp32.model.state_dict())
    graph = SakiGoInference(graph_model, device="cuda", dtype=torch.float32, use_cuda_graph=True)
    graph_out = graph(cuda_board, cuda_rules)
    plain_out = fp32(cuda_board, cuda_rules)
    for key in plain_out:
        assert_close(graph_out[key], plain_out[key], atol=1e-5)

    bf16_model = SakiGoModel(config)
    bf16_model.load_state_dict(fp32.model.state_dict())
    bf16 = SakiGoInference(bf16_model, device="cuda", dtype=torch.bfloat16)
    bf16_out = bf16(cuda_board, cuda_rules)
    assert all(torch.isfinite(value.float()).all().item() for value in bf16_out.values())
