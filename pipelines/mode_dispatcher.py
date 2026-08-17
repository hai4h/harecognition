"""Per-frame mode dispatcher: branches on the active UI mode."""


class ModeDispatcher:
    def __init__(self, pipeline_mode1, pipeline_mode2):
        self._mode1 = pipeline_mode1
        self._mode2 = pipeline_mode2

    def process(self, frame, mode: int):
        """Returns (overlay_frame, status/telemetry dict) for the active mode."""
        if mode == 1:
            return self._mode1.process(frame)
        if mode == 2:
            return self._mode2.process(frame)
        raise ValueError(f"unknown mode: {mode}")