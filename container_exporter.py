import os
import psutil
import logging
import uvicorn
import multiprocessing

from typing import Optional
from prometheus_client import Gauge, make_asgi_app, REGISTRY


logger = logging.getLogger(__name__)
ENV_DISABLE_DOCKER_CPU_WARNING = False


def get_system_memory(
    # For cgroups v1:
    memory_limit_filename="/sys/fs/cgroup/memory/memory.limit_in_bytes",
    # For cgroups v2:
    memory_limit_filename_v2="/sys/fs/cgroup/memory.max",
):
    """Return the total amount of system memory in bytes.

    Returns:
        The total amount of system memory in bytes.
    """
    # Try to accurately figure out the memory limit if we are in a docker
    # container. Note that this file is not specific to Docker and its value is
    # often much larger than the actual amount of memory.
    docker_limit = None
    if os.path.exists(memory_limit_filename):
        with open(memory_limit_filename, "r") as f:
            docker_limit = int(f.read().strip())
    elif os.path.exists(memory_limit_filename_v2):
        with open(memory_limit_filename_v2, "r") as f:
            # Don't forget to strip() the newline:
            max_file = f.read().strip()
            if max_file.isnumeric():
                docker_limit = int(max_file)
            else:
                # max_file is "max", i.e. is unset.
                docker_limit = None

    # Use psutil if it is available.
    psutil_memory_in_bytes = psutil.virtual_memory().total

    if docker_limit is not None:
        # We take the min because the cgroup limit is very large if we aren't
        # in Docker.
        return min(docker_limit, psutil_memory_in_bytes)

    return psutil_memory_in_bytes


def get_cgroupv1_used_memory(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        cache_bytes = -1
        rss_bytes = -1
        inactive_file_bytes = -1
        working_set = -1
        for line in lines:
            if "total_rss " in line:
                rss_bytes = int(line.split()[1])
            elif "cache " in line:
                cache_bytes = int(line.split()[1])
            elif "inactive_file" in line:
                inactive_file_bytes = int(line.split()[1])
        if cache_bytes >= 0 and rss_bytes >= 0 and inactive_file_bytes >= 0:
            working_set = rss_bytes + cache_bytes - inactive_file_bytes
            assert working_set >= 0
            return working_set
        return None


def get_cgroupv2_used_memory(stat_file, usage_file):
    # Uses same calculation as libcontainer, that is:
    # memory.current - memory.stat[inactive_file]
    # Source: https://github.com/google/cadvisor/blob/24dd1de08a72cfee661f6178454db995900c0fee/container/libcontainer/handler.go#L836  # noqa: E501
    inactive_file_bytes = -1
    current_usage = -1
    with open(usage_file, "r") as f:
        current_usage = int(f.read().strip())
    with open(stat_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "inactive_file" in line:
                inactive_file_bytes = int(line.split()[1])
        if current_usage >= 0 and inactive_file_bytes >= 0:
            working_set = current_usage - inactive_file_bytes
            assert working_set >= 0
            return working_set
        return None


def get_used_memory():
    """Return the currently used system memory in bytes

    Returns:
        The total amount of used memory
    """
    # Try to accurately figure out the memory usage if we are in a docker
    # container.
    docker_usage = None
    # For cgroups v1:
    memory_usage_filename = "/sys/fs/cgroup/memory/memory.stat"
    # For cgroups v2:
    memory_usage_filename_v2 = "/sys/fs/cgroup/memory.current"
    memory_stat_filename_v2 = "/sys/fs/cgroup/memory.stat"
    if os.path.exists(memory_usage_filename):
        docker_usage = get_cgroupv1_used_memory(memory_usage_filename)
    elif os.path.exists(memory_usage_filename_v2) and os.path.exists(
        memory_stat_filename_v2
    ):
        docker_usage = get_cgroupv2_used_memory(
            memory_stat_filename_v2, memory_usage_filename_v2
        )

    if docker_usage is not None:
        return docker_usage
    return psutil.virtual_memory().used


def _get_docker_cpus(
    cpu_quota_file_name="/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
    cpu_period_file_name="/sys/fs/cgroup/cpu/cpu.cfs_period_us",
    cpuset_file_name="/sys/fs/cgroup/cpuset/cpuset.cpus",
    cpu_max_file_name="/sys/fs/cgroup/cpu.max",
) -> Optional[float]:
    # TODO (Alex): Don't implement this logic oursleves.
    # Docker has 2 underyling ways of implementing CPU limits:
    # https://docs.docker.com/config/containers/resource_constraints/#configure-the-default-cfs-scheduler
    # 1. --cpuset-cpus 2. --cpus or --cpu-quota/--cpu-period (--cpu-shares is a
    # soft limit so we don't worry about it). For Ray's purposes, if we use
    # docker, the number of vCPUs on a machine is whichever is set (ties broken
    # by smaller value).

    cpu_quota = None
    # See: https://bugs.openjdk.java.net/browse/JDK-8146115
    if os.path.exists(cpu_quota_file_name) and os.path.exists(cpu_period_file_name):
        try:
            with open(cpu_quota_file_name, "r") as quota_file, open(
                cpu_period_file_name, "r"
            ) as period_file:
                cpu_quota = float(quota_file.read()) / float(period_file.read())
        except Exception:
            logger.exception("Unexpected error calculating docker cpu quota.")
    # Look at cpu.max for cgroups v2
    elif os.path.exists(cpu_max_file_name):
        try:
            max_file = open(cpu_max_file_name).read()
            quota_str, period_str = max_file.split()
            if quota_str.isnumeric() and period_str.isnumeric():
                cpu_quota = float(quota_str) / float(period_str)
            else:
                # quota_str is "max" meaning the cpu quota is unset
                cpu_quota = None
        except Exception:
            logger.exception("Unexpected error calculating docker cpu quota.")
    if (cpu_quota is not None) and (cpu_quota < 0):
        cpu_quota = None
    elif cpu_quota == 0:
        # Round up in case the cpu limit is less than 1.
        cpu_quota = 1

    cpuset_num = None
    if os.path.exists(cpuset_file_name):
        try:
            with open(cpuset_file_name) as cpuset_file:
                ranges_as_string = cpuset_file.read()
                ranges = ranges_as_string.split(",")
                cpu_ids = []
                for num_or_range in ranges:
                    if "-" in num_or_range:
                        start, end = num_or_range.split("-")
                        cpu_ids.extend(list(range(int(start), int(end) + 1)))
                    else:
                        cpu_ids.append(int(num_or_range))
                cpuset_num = len(cpu_ids)
        except Exception:
            logger.exception("Unexpected error calculating docker cpuset ids.")
    # Possible to-do: Parse cgroups v2's cpuset.cpus.effective for the number
    # of accessible CPUs.

    if cpu_quota and cpuset_num:
        return min(cpu_quota, cpuset_num)
    return cpu_quota or cpuset_num


def get_num_cpus(
    override_docker_cpu_warning: bool = ENV_DISABLE_DOCKER_CPU_WARNING,
) -> int:
    """
    Get the number of CPUs available on this node.
    Depending on the situation, use multiprocessing.cpu_count() or cgroups.

    Args:
        override_docker_cpu_warning: An extra flag to explicitly turn off the Docker
            warning. Setting this flag True has the same effect as setting the env
            RAY_DISABLE_DOCKER_CPU_WARNING. By default, whether or not to log
            the warning is determined by the env variable
            RAY_DISABLE_DOCKER_CPU_WARNING.
    """
    cpu_count = multiprocessing.cpu_count()
    if os.environ.get("RAY_USE_MULTIPROCESSING_CPU_COUNT"):
        logger.info(
            "Detected RAY_USE_MULTIPROCESSING_CPU_COUNT=1: Using "
            "multiprocessing.cpu_count() to detect the number of CPUs. "
            "This may be inconsistent when used inside docker. "
            "To correctly detect CPUs, unset the env var: "
            "`RAY_USE_MULTIPROCESSING_CPU_COUNT`."
        )
        return cpu_count
    try:
        # Not easy to get cpu count in docker, see:
        # https://bugs.python.org/issue36054
        docker_count = _get_docker_cpus()
        if docker_count is not None and docker_count != cpu_count:
            # Don't log this warning if we're on K8s or if the warning is
            # explicitly disabled.
            if (
                "KUBERNETES_SERVICE_HOST" not in os.environ
                and not ENV_DISABLE_DOCKER_CPU_WARNING
                and not override_docker_cpu_warning
            ):
                logger.warning(
                    "Detecting docker specified CPUs. In "
                    "previous versions of Ray, CPU detection in containers "
                    "was incorrect. Please ensure that Ray has enough CPUs "
                    "allocated. As a temporary workaround to revert to the "
                    "prior behavior, set "
                    "`RAY_USE_MULTIPROCESSING_CPU_COUNT=1` as an env var "
                    "before starting Ray. Set the env var: "
                    "`RAY_DISABLE_DOCKER_CPU_WARNING=1` to mute this warning."
                )
            # TODO (Alex): We should probably add support for fractional cpus.
            if int(docker_count) != float(docker_count):
                logger.warning(
                    f"Ray currently does not support initializing Ray "
                    f"with fractional cpus. Your num_cpus will be "
                    f"truncated from {docker_count} to "
                    f"{int(docker_count)}."
                )
            docker_count = int(docker_count)
            cpu_count = docker_count

    except Exception:
        # `nproc` and cgroup are linux-only. If docker only works on linux
        # (will run in a linux VM on other platforms), so this is fine.
        pass

    return cpu_count


def strip_lifespan_events(app):
    """
    handlers/ignores lifespan events from being routed to the given app.
    """
    async def _app(scope, receive, send):
        if scope.get("type") == "lifespan":
            payload = await receive()
            await send({'type': payload['type'] + ".complete"})
            return
        await app(scope, receive, send)
    return _app


container_cpu_count = Gauge(
    name = "container_cpu_count", 
    documentation = "Total CPUs available on container", 
    unit = "cores"
)
container_cpu_count.set_function(get_num_cpus)

container_mem_used = Gauge(
    name = "container_mem_used", 
    documentation = "Memory usage on container", 
    unit = "bytes"
)
container_mem_used.set_function(get_used_memory)

container_mem_total = Gauge(
    name = "container_mem_total", 
    documentation = "Total memory on container", 
    unit = "bytes"
)
container_mem_total.set_function(get_system_memory)


app = make_asgi_app(REGISTRY)
app = strip_lifespan_events(app)


# if __name__ == "__main__":
#     uvicorn.run(app = app, port = 8000, log_level = "info")
#     # uvicorn container_exporter:app