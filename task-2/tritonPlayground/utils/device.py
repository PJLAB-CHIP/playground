import triton


def is_hip():
    """
    Whether the current backend is HIP (a backend for AMD GPUs).
    """
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    """
    Whether the current target is a CDNA architecture (e.g., MI100).
    """
    return (
        is_hip()
        and triton.runtime.driver.active.get_current_target().arch
        in (
            "gfx940",
            "gfx941",
            "gfx942",
            "gfx90a",
            "gfx908",
        )
    )


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip_mi200():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx90a"
