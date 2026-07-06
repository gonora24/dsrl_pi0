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


if __name__ == "__main__":
    pytest.main([__file__])