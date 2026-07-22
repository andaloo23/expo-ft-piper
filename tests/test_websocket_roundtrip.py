"""Round-trip the five-operation protocol against a FakeEnv (no hardware).

Runs piper_client.run_client's handler on a local port and drives it with
a synchronous websocket client, asserting the learner-visible contract.
"""

import asyncio
import threading
import time

import numpy as np
import pytest
import websockets.asyncio.server as _server
from openpi_client import msgpack_numpy
from websockets.sync.client import connect

import piper_client.run_client as rc

PORT = 8199


@pytest.fixture(scope="module")
def server():
    rc._config_task_path = "tests/fakes/fake_task.py"
    rc._env_storage.clear()
    started = threading.Event()
    shutdown: dict = {}

    async def _serve():
        shutdown["loop"] = asyncio.get_running_loop()
        shutdown["event"] = asyncio.Event()
        async with _server.serve(
            rc._handle_environment_request, "127.0.0.1", PORT,
            compression=None, max_size=None, ping_interval=None, ping_timeout=None,
        ):
            started.set()
            await shutdown["event"].wait()

    t = threading.Thread(target=lambda: asyncio.run(_serve()), daemon=True)
    t.start()
    assert started.wait(timeout=10)
    yield f"ws://127.0.0.1:{PORT}"
    shutdown["loop"].call_soon_threadsafe(shutdown["event"].set)
    t.join(timeout=5)


def _rpc(ws, packer, request):
    ws.send(packer.pack(request))
    return msgpack_numpy.unpackb(ws.recv())


def test_five_operation_roundtrip(server):
    packer = msgpack_numpy.Packer()
    with connect(server, max_size=None, compression=None) as ws:
        # create_env
        resp = _rpc(ws, packer, {"operation": "create_env", "env_usage": "train", "video_dir": ""})
        assert resp["status"] == "success"
        assert resp["task_description"] == "pick up the cube"
        env_id = resp["env_id"]
        assert env_id == "fake_pick_train"

        # reset
        resp = _rpc(ws, packer, {"operation": "reset", "env_id": env_id})
        assert resp["status"] == "success" and resp["done"] is False
        obs = resp["observation"]
        assert obs["exterior_image_1_left"].shape == (180, 320, 3)
        assert obs["cartesian_position"].shape == (6,)
        assert obs["gripper_position"].shape == (1,)
        assert obs["prompt"] == "pick up the cube"

        # get_observation
        resp = _rpc(ws, packer, {"operation": "get_observation", "env_id": env_id})
        assert resp["status"] == "success"
        assert set(resp["observation"]) >= {
            "exterior_image_1_left", "exterior_image_2_left", "wrist_image_left",
            "cartesian_position", "gripper_position", "prompt",
        }

        # step returns the executed (safety-filtered) action, not the request
        action = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        resp = _rpc(ws, packer, {"operation": "step", "env_id": env_id, "action": action})
        assert resp["status"] == "success"
        assert resp["action_type"] == "policy"
        np.testing.assert_allclose(resp["action"], np.asarray(action) * 0.5)

        # NaN action is neutralized, not crashed on
        bad = [float("nan")] * 7
        resp = _rpc(ws, packer, {"operation": "step", "env_id": env_id, "action": bad})
        assert resp["status"] == "success"
        np.testing.assert_allclose(resp["action"], 0.0)

        # invalid sentinel action (all -1) is not executed
        resp = _rpc(ws, packer, {"operation": "step", "env_id": env_id, "action": [-1.0] * 7})
        assert resp["status"] == "success"
        np.testing.assert_allclose(resp["action"], -1.0)

        # get_info_for_step
        resp = _rpc(ws, packer, {"operation": "get_info_for_step", "env_id": env_id})
        assert resp["status"] == "success"
        assert set(resp) >= {"done", "success", "reward", "mask"}

        # unknown operation errors cleanly
        resp = _rpc(ws, packer, {"operation": "bogus"})
        assert resp["status"] == "error"
