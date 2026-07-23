import numpy as np

from piper_client.success.operator_panel import OperatorPanel


def make_panel():
    return OperatorPanel(key_source=iter(()))  # inert source; drive handle_key directly


def test_pause_toggles_and_clears_on_label():
    p = make_panel()
    assert not p.paused
    p.handle_key(" ")
    assert p.paused
    p.handle_key(" ")
    assert not p.paused
    # labeling while paused unpauses (episode is ending anyway)
    p.handle_key(" ")
    p.handle_key("s")
    assert not p.paused
    assert p.consume_label() == "success"


def test_label_is_consumed_once():
    p = make_panel()
    p.handle_key("f")
    assert p.consume_label() == "reset"
    assert p.consume_label() is None


def test_reset_episode_clears_state():
    p = make_panel()
    p.handle_key(" ")
    p.handle_key("s")
    p.handle_key(" ")
    p.reset_episode()
    assert not p.paused
    assert p.consume_label() is None


def test_takeover_toggles_and_clears_on_reset():
    p = make_panel()
    assert not p.takeover
    p.handle_key("t")
    assert p.takeover
    p.handle_key("t")
    assert not p.takeover
    p.handle_key("t")
    p.reset_episode()
    assert not p.takeover


def test_unknown_keys_ignored():
    p = make_panel()
    for ch in ("x", "1", "\n", "?"):
        p.handle_key(ch)
    assert not p.paused
    assert p.consume_label() is None
