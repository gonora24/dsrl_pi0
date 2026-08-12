import tempfile
import numpy as np
import gym
import pytest

from jaxrl2.data.replay_buffer import ReplayBuffer


def make_buffer(capacity=10, obs_dim=3, action_dim=2, chunk_size=0):
    obs_space = gym.spaces.Dict(
        {
            "pixels": gym.spaces.Box(
                low=0,
                high=255,
                shape=(obs_dim,),
                dtype=np.uint8,
            ),
            "state": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            ),
        }
    )

    action_space = gym.spaces.Box(
        low=-1,
        high=1,
        shape=(action_dim,),
        dtype=np.float32,
    )

    return ReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        capacity=capacity,
        chunk_size=chunk_size,
    )


def make_transition(i, chunk_size=0):
    obs = {
        "pixels": np.full((3,), i, dtype=np.uint8),
        "state": np.full((3,), float(i), dtype=np.float32),
    }

    next_obs = {
        "pixels": np.full((3,), i + 1, dtype=np.uint8),
        "state": np.full((3,), float(i + 1), dtype=np.float32),
    }

    transition = {
        "observations": obs,
        "next_observations": next_obs,
        "actions": np.array([i, i + 1], dtype=np.float32),
        "next_actions": np.array([i + 1, i + 2], dtype=np.float32),
        "masks": 1.0,
        "discount": 0.99,
    }

    if chunk_size > 0:
        transition["rewards"] = np.full(chunk_size, i, dtype=np.float32)
        transition["terminations"] = np.zeros(chunk_size, dtype=np.bool_)
    else:
        transition["rewards"] = float(i)

    return transition


class TestReplayBufferOnlineTraining:

    def test_chunk_reward_storage(self):
        chunk_size = 4

        buffer = make_buffer(
            capacity=10,
            chunk_size=chunk_size,
        )

        transition = make_transition(0, chunk_size)

        transition["rewards"] = np.array(
            [-1.0, -1.0, 0.0, 0.0],
            dtype=np.float32,
        )

        transition["terminations"] = np.array(
            [False, False, True, True],
            dtype=np.bool_,
        )

        buffer.insert(transition)

        np.testing.assert_array_equal(
            buffer.data["rewards"][0],
            transition["rewards"],
        )

        np.testing.assert_array_equal(
            buffer.data["terminations"][0],
            transition["terminations"],
        )

    def test_chunk_reward_sampling_shape(self):
        chunk_size = 8

        buffer = make_buffer(
            capacity=20,
            chunk_size=chunk_size,
        )

        for i in range(10):
            buffer.insert(make_transition(i, chunk_size))

        batch = buffer.sample(5)

        assert batch["rewards"].shape == (5, chunk_size)
        assert batch["terminations"].shape == (5, chunk_size)

    def test_discount_horizon_storage(self):
        query_freq = 5
        gamma = 0.99

        buffer = make_buffer(capacity=10)

        transition = make_transition(0)
        transition["discount"] = gamma ** query_freq

        buffer.insert(transition)

        assert np.isclose(
            buffer.data["discount"][0],
            gamma ** query_freq,
        )

    def test_streaming_buffer_sampling_limit(self):
        buffer = make_buffer(capacity=200)

        for i in range(200):
            buffer.insert(make_transition(i))

        buffer.streaming_buffer_size = 50

        for _ in range(100):
            batch = buffer.sample(32)

            assert np.max(batch["actions"][:, 0]) < 50

    def test_chunk_size_zero(self):
        buffer = make_buffer(
            capacity=10,
            chunk_size=0,
        )

        buffer.insert(make_transition(0))

        assert buffer.data["rewards"].shape == (10,)
        assert "terminations" not in buffer.data

    def test_chunk_size_nonzero(self):
        chunk_size = 6

        buffer = make_buffer(
            capacity=10,
            chunk_size=chunk_size,
        )

        buffer.insert(make_transition(0, chunk_size))

        assert buffer.data["rewards"].shape == (
            10,
            chunk_size,
        )

        assert buffer.data["terminations"].shape == (
            10,
            chunk_size,
        )

    def test_masks_and_terminations_consistency(self):
        chunk_size = 4

        buffer = make_buffer(
            capacity=10,
            chunk_size=chunk_size,
        )

        transition = make_transition(0, chunk_size)

        transition["masks"] = 0.0
        transition["terminations"] = np.ones(
            chunk_size,
            dtype=np.bool_,
        )

        buffer.insert(transition)

        assert buffer.data["masks"][0] == 0.0
        assert np.all(buffer.data["terminations"][0])

    def test_multiple_chunk_insertions(self):
        chunk_size = 3

        buffer = make_buffer(
            capacity=20,
            chunk_size=chunk_size,
        )

        for t in range(10):
            transition = make_transition(t, chunk_size)

            transition["rewards"] = np.array(
                [t, t + 1, t + 2],
                dtype=np.float32,
            )

            buffer.insert(transition)

        assert len(buffer) == 10

        np.testing.assert_array_equal(
            buffer.data["rewards"][5],
            np.array([5, 6, 7], dtype=np.float32),
        )

    def test_add_online_data_style_insertion(self):
        """
        Mimics add_online_data_to_buffer().
        """

        chunk_size = 4

        buffer = make_buffer(
            capacity=20,
            chunk_size=chunk_size,
        )

        num_queries = 5

        for t in range(num_queries):

            transition = {
                "observations": {
                    "pixels": np.ones(3, dtype=np.uint8) * t,
                    "state": np.ones(3, dtype=np.float32) * t,
                },
                "next_observations": {
                    "pixels": np.ones(3, dtype=np.uint8) * (t + 1),
                    "state": np.ones(3, dtype=np.float32) * (t + 1),
                },
                "actions": np.ones(2, dtype=np.float32) * t,
                "next_actions": np.ones(2, dtype=np.float32) * (t + 1),
                "rewards": np.array(
                    [-1, -1, -1, 0],
                    dtype=np.float32,
                ),
                "terminations": np.array(
                    [False, False, False, True]
                ),
                "masks": 0.0 if t == num_queries - 1 else 1.0,
                "discount": 0.95,
            }

            buffer.insert(transition)

        buffer.increment_traj_counter()

        assert len(buffer) == num_queries
        assert buffer._traj_counter == 1

        start, end = buffer.traj_bounds[0]

        assert start == 0
        assert end == num_queries

    def test_get_random_traj_preserves_episode_length(self):
        buffer = make_buffer(capacity=100)

        lengths = [3, 7, 5]

        for traj_len in lengths:
            for i in range(traj_len):
                buffer.insert(make_transition(i))
            buffer.increment_traj_counter()

        batch = buffer.get_random_trajs(3)

        returned_lengths = [
            len(x)
            for x in batch["actions"]
        ]

        for length in returned_lengths:
            assert length in lengths

    def test_action_stats_after_normalization(self):
        buffer = make_buffer(capacity=100)

        for i in range(50):
            buffer.insert(make_transition(i))

        stats = buffer.compute_action_stats()

        buffer.normalize_actions(stats)

        normalized = buffer.data["actions"][:buffer.size]

        np.testing.assert_allclose(
            normalized[:, :-1].mean(axis=0),
            np.zeros(normalized.shape[1] - 1),
            atol=1e-5,
        )

    def test_capacity_growth_keeps_chunk_rewards(self):
        chunk_size = 5

        buffer = make_buffer(
            capacity=2,
            chunk_size=chunk_size,
        )

        for i in range(10):
            buffer.insert(make_transition(i, chunk_size))

        assert buffer.capacity >= 10

        np.testing.assert_array_equal(
            buffer.data["rewards"][0],
            np.zeros(chunk_size, dtype=np.float32),
        )

        np.testing.assert_array_equal(
            buffer.data["rewards"][5],
            np.full(chunk_size, 5, dtype=np.float32),
        )


def _make_raw_traj(T, action_dim=7, done_at=None):
    """Build a synthetic collect_traj_chunked raw trajectory of T env steps.

    done_at: optional step index (0-based) at which done=True fires early.
    """
    obs_shape = (3,)  # dummy spatial dims

    # T+1 obs dicts with a fake batch dim matching the real collector format
    all_obs = [
        {
            "pixels": np.full((1, *obs_shape, 1), float(t), dtype=np.float32),
        }
        for t in range(T + 1)
    ]

    all_actions = [np.full((action_dim,), float(t), dtype=np.float32) for t in range(T)]

    step_rewards = []
    step_terminations = []
    for t in range(T):
        is_done = (done_at is not None and t >= done_at)
        step_rewards.append(0.0 if is_done else -1.0)
        step_terminations.append(is_done)

    return {
        "observations": all_obs,
        "all_actions": all_actions,
        "step_rewards": step_rewards,
        "step_terminations": step_terminations,
        "episode_return": float(sum(step_rewards)),
        "is_success": False,
        "env_steps": T,
    }


class TestBuildChunkedInsertTraj:

    def test_transition_count_clean_episode(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T = 4, 12
        raw = _make_raw_traj(T)
        traj = _build_chunked_insert_traj(raw, Q)
        assert traj["num_transitions"] == T - Q + 1
        assert len(traj["observations"]) == T - Q + 1
        assert len(traj["actions"]) == T - Q + 1

    def test_zero_transitions_when_episode_shorter_than_chunk(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T = 10, 5
        raw = _make_raw_traj(T)
        traj = _build_chunked_insert_traj(raw, Q)
        assert traj["num_transitions"] == 0
        assert len(traj["observations"]) == 0

    def test_obs_next_obs_alignment(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T = 3, 9
        raw = _make_raw_traj(T)
        traj = _build_chunked_insert_traj(raw, Q)
        for t in range(traj["num_transitions"]):
            # obs shape is (1, obs_dim, 1); [0, 0, 0] gives the first element = t
            np.testing.assert_allclose(
                traj["observations"][t]["pixels"][0, 0, 0],
                float(t),
            )
            np.testing.assert_allclose(
                traj["next_observations"][t]["pixels"][0, 0, 0],
                float(t + Q),
            )

    def test_actions_cover_correct_steps(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T, A = 4, 10, 7
        raw = _make_raw_traj(T, action_dim=A)
        traj = _build_chunked_insert_traj(raw, Q)
        for t in range(traj["num_transitions"]):
            chunk = traj["actions"][t]
            assert chunk.shape == (Q, A)
            for k in range(Q):
                np.testing.assert_allclose(chunk[k], float(t + k))

    def test_next_actions_fallback_for_final_window(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T = 4, 10
        raw = _make_raw_traj(T)
        traj = _build_chunked_insert_traj(raw, Q)
        N = traj["num_transitions"]
        last_t = N - 1  # t = T - Q = 6
        # For the last window t=6: t + 2Q = 14 > T=10, so next_actions = current chunk
        np.testing.assert_array_equal(
            traj["next_actions"][last_t],
            traj["actions"][last_t],
        )
        # For a non-final window t=0: t + 2Q = 8 <= T=10, proper next chunk
        np.testing.assert_array_equal(
            traj["next_actions"][0],
            np.stack(raw["all_actions"][Q: 2 * Q]),
        )

    def test_rewards_shape_and_values(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T = 3, 9
        raw = _make_raw_traj(T)  # no done → all rewards are -1
        traj = _build_chunked_insert_traj(raw, Q)
        for t in range(traj["num_transitions"]):
            r = traj["rewards"][t]
            assert r.shape == (Q,)
            assert r.dtype == np.float32
            np.testing.assert_array_equal(r, np.full(Q, -1.0, dtype=np.float32))

    def test_masks_zero_when_termination_in_window(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T = 4, 8
        done_at = 5  # done fires at step 5
        raw = _make_raw_traj(T, done_at=done_at)
        traj = _build_chunked_insert_traj(raw, Q)
        for t in range(traj["num_transitions"]):
            # Window [t, t+Q) contains a terminal step if done_at is within it
            has_terminal = any(raw["step_terminations"][t: t + Q])
            expected_mask = 0.0 if has_terminal else 1.0
            assert traj["masks"][t] == expected_mask, (
                f"window {t}: expected mask {expected_mask}, got {traj['masks'][t]}"
            )

    def test_terminations_shape(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T = 5, 15
        raw = _make_raw_traj(T)
        traj = _build_chunked_insert_traj(raw, Q)
        for t in range(traj["num_transitions"]):
            term = traj["terminations"][t]
            assert term.shape == (Q,)
            assert term.dtype == np.bool_

    def test_buffer_insert_succeeds_with_chunked_traj(self):
        """End-to-end: _build_chunked_insert_traj output can be inserted into a chunked buffer."""
        import types
        from examples.train_utils_sim import _build_chunked_insert_traj, add_online_data_to_buffer
        import gym.spaces as spaces

        Q, T, obs_dim, action_dim = 4, 12, 3, 7

        # obs from _make_raw_traj have shape (1, obs_dim, 1); after batch-dim strip → (obs_dim, 1)
        obs_space = spaces.Dict({
            "pixels": spaces.Box(low=0.0, high=255.0, shape=(obs_dim, 1), dtype=np.float32),
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(Q, action_dim), dtype=np.float32)
        buffer = ReplayBuffer(obs_space, action_space, capacity=50, chunk_size=Q)

        raw = _make_raw_traj(T, action_dim=action_dim)
        traj = _build_chunked_insert_traj(raw, Q)

        variant = types.SimpleNamespace(
            query_freq=Q,
            discount=0.99,
            add_states=False,
        )
        add_online_data_to_buffer(variant, traj, buffer)

        assert buffer.size == traj["num_transitions"]
        assert buffer.data["rewards"].shape[1] == Q
        assert buffer.data["actions"].shape == (buffer.capacity, Q, action_dim)


def _make_raw_noise_traj(T, H=50, noise_dim=32, query_freq=10, done_at=None):
    """Synthetic collect_traj_chunked(store_noise=True) raw trajectory.

    Noise is constant within each replan block of length query_freq.
    """
    obs_shape = (3,)
    all_obs = [
        {"pixels": np.full((1, *obs_shape, 1), float(t), dtype=np.float32)}
        for t in range(T + 1)
    ]
    all_actions = []
    step_rewards = []
    step_terminations = []
    for t in range(T):
        block = t // query_freq
        noise = np.full((H, noise_dim), float(block), dtype=np.float32)
        all_actions.append(noise)
        is_done = done_at is not None and t >= done_at
        step_rewards.append(0.0 if is_done else -1.0)
        step_terminations.append(is_done)
    return {
        "observations": all_obs,
        "all_actions": all_actions,
        "step_rewards": step_rewards,
        "step_terminations": step_terminations,
        "episode_return": float(sum(step_rewards)),
        "is_success": False,
        "env_steps": T,
    }


class TestBuildChunkedInsertTrajNoiseMode:

    def test_transition_count_and_shapes(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T, H, D = 4, 12, 50, 32
        raw = _make_raw_noise_traj(T, H=H, noise_dim=D, query_freq=Q)
        traj = _build_chunked_insert_traj(raw, Q, stack_actions=False)
        assert traj["num_transitions"] == T - Q + 1
        for t in range(traj["num_transitions"]):
            assert traj["actions"][t].shape == (H, D)
            assert traj["next_actions"][t].shape == (H, D)
            assert traj["rewards"][t].shape == (Q,)

    def test_noise_constant_within_replan_block(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T, H, D = 4, 12, 8, 4
        raw = _make_raw_noise_traj(T, H=H, noise_dim=D, query_freq=Q)
        traj = _build_chunked_insert_traj(raw, Q, stack_actions=False)
        # Windows starting in the same replan block share the same noise
        np.testing.assert_array_equal(traj["actions"][0], traj["actions"][1])
        np.testing.assert_array_equal(traj["actions"][0], traj["actions"][Q - 1])
        # First window of the next block differs
        assert not np.array_equal(traj["actions"][0], traj["actions"][Q])

    def test_next_action_uses_step_t_plus_Q(self):
        from examples.train_utils_sim import _build_chunked_insert_traj
        Q, T, H, D = 4, 12, 8, 4
        raw = _make_raw_noise_traj(T, H=H, noise_dim=D, query_freq=Q)
        traj = _build_chunked_insert_traj(raw, Q, stack_actions=False)
        np.testing.assert_array_equal(traj["next_actions"][0], raw["all_actions"][Q])
        # Last window: t = T - Q = 8; t + Q = 12 == T → fallback to current
        last_t = traj["num_transitions"] - 1
        np.testing.assert_array_equal(
            traj["next_actions"][last_t], traj["actions"][last_t]
        )

    def test_buffer_insert_succeeds_with_noise_traj(self):
        import types
        from examples.train_utils_sim import _build_chunked_insert_traj, add_online_data_to_buffer
        import gym.spaces as spaces

        Q, T, H, D, obs_dim = 4, 12, 50, 32, 3
        obs_space = spaces.Dict({
            "pixels": spaces.Box(low=0.0, high=255.0, shape=(obs_dim, 1), dtype=np.float32),
        })
        action_space = spaces.Box(low=-1.0, high=1.0, shape=(H, D), dtype=np.float32)
        buffer = ReplayBuffer(obs_space, action_space, capacity=50, chunk_size=Q)

        raw = _make_raw_noise_traj(T, H=H, noise_dim=D, query_freq=Q)
        traj = _build_chunked_insert_traj(raw, Q, stack_actions=False)

        variant = types.SimpleNamespace(
            query_freq=Q,
            discount=0.99,
            add_states=False,
        )
        add_online_data_to_buffer(variant, traj, buffer)

        assert buffer.size == traj["num_transitions"]
        assert buffer.data["rewards"].shape[1] == Q
        assert buffer.data["actions"].shape == (buffer.capacity, H, D)


def _make_chunk_reward_buffer(obs_dim, action_dim, Q, capacity):
    """Buffer shaped like DummyEnv's dsrl_na + chunk_reward action space:
    actions/next_actions are (Q, action_dim) chunks, not flat vectors.
    """
    obs_space = gym.spaces.Dict(
        {
            "pixels": gym.spaces.Box(low=0, high=255, shape=(obs_dim,), dtype=np.uint8),
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
        }
    )
    action_space = gym.spaces.Box(low=-1, high=1, shape=(Q, action_dim), dtype=np.float32)
    return ReplayBuffer(
        observation_space=obs_space,
        action_space=action_space,
        capacity=capacity,
        chunk_size=Q,
    )


def _write_chunk_reward_hdf5(path, T, done_at, obs_dim=3, action_dim=2):
    """Write a single-demo HDF5 file in the format load_from_hdf5 expects.

    Termination fires at step `done_at` (0-indexed into the T-step episode).
    """
    import h5py

    pixels = np.repeat(np.arange(T + 1, dtype=np.uint8)[:, None], obs_dim, axis=1)
    state = pixels.astype(np.float32)
    actions = np.arange(T * action_dim, dtype=np.float32).reshape(T, action_dim)
    terminations = np.zeros(T, dtype=np.bool_)
    terminations[done_at] = True
    rewards = np.where(terminations, 0.0, -1.0).astype(np.float32)
    masks = np.logical_not(terminations).astype(np.float32)

    with h5py.File(path, "w") as f:
        data_grp = f.create_group("data")
        demo = data_grp.create_group("demo_0")
        obs_grp = demo.create_group("obs")
        obs_grp.create_dataset("pixels", data=pixels)
        obs_grp.create_dataset("state", data=state)
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("rewards", data=rewards)
        demo.create_dataset("terminations", data=terminations)
        demo.create_dataset("masks", data=masks)


class TestLoadFromHdf5ChunkRewardMasks:
    """Regression tests for the overlapping-window mask computation in
    ReplayBuffer.load_from_hdf5's chunk_reward branch (see replay_buffer.py).
    """

    def test_mask_zero_for_window_containing_terminal_step(self):
        # T=10, Q=4 -> 7 overlapping windows (t = 0..6). done_at=9 (the very
        # last step) only falls inside the *last* window (t=6, covering steps
        # 6..9); masks[t] for t=6 is 1.0 (not terminal at t=6 itself), so the
        # old buggy code (`masks[t]`) would incorrectly report mask=1.0 there.
        T, Q, done_at, obs_dim, action_dim = 10, 4, 9, 3, 2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/traj.hdf5"
            _write_chunk_reward_hdf5(path, T, done_at, obs_dim, action_dim)

            buffer = _make_chunk_reward_buffer(obs_dim, action_dim, Q, capacity=32)
            buffer.load_from_hdf5(path, chunk_reward=True, query_freq=Q, discount=0.99)

            num_windows = T - Q + 1
            assert buffer.size == num_windows

            for t in range(num_windows):
                expected_mask = 0.0 if t <= done_at < t + Q else 1.0
                assert buffer.data["masks"][t] == expected_mask, (
                    f"window t={t}: expected mask {expected_mask}, "
                    f"got {buffer.data['masks'][t]}"
                )
            # Sanity check: exactly the last window should be affected here.
            assert buffer.data["masks"][num_windows - 1] == 0.0
            assert np.all(buffer.data["masks"][: num_windows - 1] == 1.0)

    def test_mask_zero_for_every_window_overlapping_terminal_step(self):
        # done_at in the middle of the episode: every window whose range
        # [t, t+Q) covers done_at should get mask=0.0, not just the last one.
        T, Q, done_at, obs_dim, action_dim = 12, 4, 6, 3, 2
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/traj.hdf5"
            _write_chunk_reward_hdf5(path, T, done_at, obs_dim, action_dim)

            buffer = _make_chunk_reward_buffer(obs_dim, action_dim, Q, capacity=32)
            buffer.load_from_hdf5(path, chunk_reward=True, query_freq=Q, discount=0.99)

            num_windows = T - Q + 1
            for t in range(num_windows):
                expected_mask = 0.0 if t <= done_at < t + Q else 1.0
                assert buffer.data["masks"][t] == expected_mask, (
                    f"window t={t}: expected mask {expected_mask}, "
                    f"got {buffer.data['masks'][t]}"
                )


if __name__ == "__main__":
    pytest.main([__file__])