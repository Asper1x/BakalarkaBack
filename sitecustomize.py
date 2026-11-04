"""
Automatic LiveStreamer hook for 3D Gaussian Splatting / 2D-SuGaR training.
Loaded automatically via PYTHONPATH by Python site module.
"""
import os
import sys

live_packet = os.getenv("LIVE_STREAM_PACKET")
if live_packet:
    try:
        from live_streamer import LiveStreamer

        streamer = LiveStreamer(
            job_id=os.getenv("JOB_ID"),
            output_packet_path=live_packet,
            streaming_flag_path=os.getenv("STREAMING_FLAG"),
            stream_interval=int(os.getenv("STREAM_INTERVAL", "15"))
        )

        import builtins
        _orig_import = builtins.__import__

        def _custom_import(name, *args, **kwargs):
            mod = _orig_import(name, *args, **kwargs)
            # When scene.gaussian_model or GaussianModel is imported, patch its optimizer.step
            if name == "scene.gaussian_model" or name == "scene":
                try:
                    target_cls = getattr(mod, "GaussianModel", None)
                    if target_cls and not getattr(target_cls, "_streamer_patched", False):
                        target_cls._streamer_patched = True
                        orig_setup = target_cls.training_setup

                        def patched_training_setup(self, *s_args, **s_kwargs):
                            orig_setup(self, *s_args, **s_kwargs)
                            opt = getattr(self, "optimizer", None)
                            if opt:
                                opt._iteration_counter = 0
                                orig_opt_step = opt.step

                                def patched_opt_step(*o_args, **o_kwargs):
                                    res = orig_opt_step(*o_args, **o_kwargs)
                                    opt._iteration_counter += 1
                                    try:
                                        streamer.step(opt._iteration_counter, self)
                                    except Exception:
                                        pass
                                    return res

                                opt.step = patched_opt_step

                        target_cls.training_setup = patched_training_setup
                except Exception:
                    pass
            return mod

        builtins.__import__ = _custom_import
    except Exception:
        pass
