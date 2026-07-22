"""Rollout server for RL training (Piper actor).

WebSocket server implementing the same five operations as upstream EXPO-FT
(create_env, reset, step, get_observation, get_info_for_step) so the GPU
learner is unchanged. Human override comes from the master Piper arm
(teleop/intervention.py) instead of a spacemouse.

Safety: motion is off unless --enable-motion is passed AND the operator
confirms at the prompt. --dry-run reads sensors and evaluates actions but
never commands the arm.

Run the actor without GPU visibility so it cannot consume learner VRAM:
    CUDA_VISIBLE_DEVICES= python -m piper_client.run_client --enable-motion ...
"""

import asyncio
import dataclasses
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
import websockets
import websockets.asyncio.server as _server
from openpi_client import msgpack_numpy

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tyro

def load_task_config(config_path: Optional[str]):
    """Load task config from module path, similar to config_flags.DEFINE_config_file."""
    if config_path is None:
        return None

    if '/' in config_path or '.py' in config_path:
        config_path = config_path.replace('.py', '').replace('/', '.')

    try:
        module = __import__(config_path, fromlist=['get_config'])
        return module.get_config()
    except Exception as e:
        raise ImportError(f"Failed to load task config from '{config_path}': {e}")


@dataclasses.dataclass
class Args:
    """Configuration arguments for the Piper rollout server."""

    server_host: str = "0.0.0.0"
    server_port: int = 8102
    config_task_path: str = "configs/task/piper_pick.py"
    dry_run: bool = False
    """Read sensors and evaluate actions, never command motion."""
    enable_motion: bool = False
    """Permit commands after operator confirmation."""


_env_storage: Dict[str, Any] = {}
_config_task_path: Optional[str] = None
_task_config: Optional[Any] = None
_motion_enabled: bool = False


def _get_human_override_action(env) -> tuple:
    """Return (action_7d or None, is_human) from the env's master arm."""
    try:
        return env.get_human_override_action()
    except Exception as e:
        logging.getLogger(__name__).warning("Master arm unavailable (%s), using policy action.", e)
        return None, False


async def _handle_environment_request(websocket: _server.ServerConnection):
    """Handle environment operation requests from the training server."""
    global _task_config
    logger = logging.getLogger(__name__)
    packer = msgpack_numpy.Packer()

    try:
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                operation = request.get("operation")

                if operation == "create_env":
                    task_config = load_task_config(_config_task_path)
                    _task_config = task_config
                    env_name = task_config.env_name
                    env_usage = request["env_usage"]
                    env_id = f"{env_name}_{env_usage}"

                    logger.info(f"Creating environment {env_id}...")
                    env_kwargs = dict(task_config)
                    env_kwargs["video_dir"] = request.get("video_dir") or ""
                    env_kwargs["dry_run"] = not _motion_enabled
                    env_kwargs["enable_motion"] = _motion_enabled
                    env = task_config.env(**env_kwargs)
                    _env_storage[env_id] = env
                    logger.info(f"Environment {env_id} created successfully")

                    task_description = task_config.language_instruction
                    response = {"status": "success", "env_id": env_id, "task_description": task_description}
                    await websocket.send(packer.pack(response))
                    logger.info(f"Sent create_env response for {env_id}")

                elif operation == "reset":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        obs = env.reset()
                        response = {
                            "status": "success",
                            "observation": obs,
                            "done": False,
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "step":
                    env_id = request["env_id"]
                    sent_action = np.array(request["action"])
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        sent_action = sent_action.astype(np.float64)
                        if not np.isfinite(sent_action).all():
                            logger.warning(
                                "Action contains NaN/Inf; replacing with zeros. "
                                "Check policy inputs (observations, encoder), training stability, or checkpoint."
                            )
                            sent_action = np.where(np.isfinite(sent_action), sent_action, 0.0)
                        real_action = sent_action.copy()
                        action_type = "policy"
                        is_human = False
                        if _task_config is not None and _task_config.env_type == "piper":
                            human_action, is_human = _get_human_override_action(env)
                            if is_human and human_action is not None:
                                real_action[:6] = human_action[:6]
                                real_action[6] = human_action[6]
                                action_type = "human"
                        sent_is_invalid = np.allclose(sent_action, -1.0)
                        if is_human or not sent_is_invalid:
                            step_result = env.step(real_action)
                            executed_action = np.array(
                                step_result["executed_action"],
                                dtype=np.float64,
                            )
                        else:
                            executed_action = real_action

                        response = {
                            "status": "success",
                            "action": executed_action.tolist(),
                            "action_type": action_type,
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "get_observation":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        obs = env.get_observation()
                        response = {
                            "status": "success",
                            "observation": obs,
                        }
                    await websocket.send(packer.pack(response))

                elif operation == "get_info_for_step":
                    env_id = request["env_id"]
                    env = _env_storage.get(env_id)

                    if env is None:
                        response = {"status": "error", "message": f"Environment {env_id} not found"}
                    else:
                        done, success, reward, mask = env.get_info_for_step()
                        response = {
                            "status": "success",
                            "done": bool(done),
                            "success": bool(success),
                            "reward": float(reward),
                            "mask": float(mask),
                        }
                    await websocket.send(packer.pack(response))

                else:
                    response = {"status": "error", "message": f"Unknown operation: {operation}"}
                    await websocket.send(packer.pack(response))

            except websockets.exceptions.ConnectionClosed:
                logger.debug(f"Connection closed by client {websocket.remote_address}")
                break
            except Exception as e:
                logger.error(f"Error handling request: {e}", exc_info=True)
                try:
                    response = {"status": "error", "message": str(e)}
                    await websocket.send(packer.pack(response))
                except websockets.exceptions.ConnectionClosed:
                    logger.debug("Connection closed while sending error response")
                    break

    except websockets.exceptions.ConnectionClosed:
        logger.debug(f"Connection closed: {websocket.remote_address}")
    except Exception as e:
        logger.error(f"Unexpected error in request handler: {e}", exc_info=True)


async def _run_server(host: str, port: int, config_task_path: Optional[str]):
    """Run the websocket server for environment operations."""
    global _config_task_path
    _config_task_path = config_task_path
    logger = logging.getLogger(__name__)

    async with _server.serve(
        _handle_environment_request,
        host,
        port,
        compression=None,
        max_size=None,
        # This server handles potentially long blocking work (env init/step).
        # Disable keepalive pings to avoid ping timeouts while busy.
        ping_interval=None,
        ping_timeout=None,
        close_timeout=100,
    ) as server:
        logger.info(f"Environment operations server started on {host}:{port}")
        await server.serve_forever()

async def main_async(args: Args) -> None:
    """Main async entry point."""
    await _run_server(args.server_host, args.server_port, args.config_task_path)

def main(args: Args) -> None:
    """Main entry point."""
    global _motion_enabled
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logging.getLogger("websockets.server").setLevel(logging.WARNING)

    if args.dry_run and args.enable_motion:
        raise SystemExit("--dry-run and --enable-motion are mutually exclusive")
    if args.enable_motion:
        print("!!! MOTION ENABLED — the follower arm WILL move.")
        print("Confirm: e-stop reachable, workspace clear, no other process on this CAN bus.")
        answer = input("Type 'yes' to continue: ").strip().lower()
        if answer != "yes":
            raise SystemExit("Motion not confirmed; exiting.")
        _motion_enabled = True
    else:
        print("Running DRY-RUN: sensors and action pipeline live, motion disabled "
              "(pass --enable-motion to move the arm).")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
