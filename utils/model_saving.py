import flax.serialization


def save_checkpoint(state, path):
    bytes_output = flax.serialization.to_bytes(state)
    with open(path, "wb") as f:
        f.write(bytes_output)


def load_checkpoint(path, blank_state):
    """
    blank_state should be initialized outside of this function, e.g.:
    blank_state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    """
    with open(path, "rb") as f:
        bytes_input = f.read()
    state = flax.serialization.from_bytes(blank_state, bytes_input)
    return state
