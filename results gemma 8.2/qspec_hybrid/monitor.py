"""Optional sampled NVML process memory and device-level energy, never fabricated."""

import threading


class Monitor:
    def __init__(self, worker_pids):
        self.pids = set(worker_pids)
        self.finished = threading.Event()
        self.thread = None
        self.peak, self.start_energy, self.error, self.foreign = None, None, None, False
        self.initialized = False

    def sample(self):
        try:
            processes = self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle)
            ours = [
                p.usedGpuMemory
                for p in processes
                if p.pid in self.pids
                and isinstance(p.usedGpuMemory, int)
                and p.usedGpuMemory < 2**60
            ]
            self.foreign |= any(p.pid not in self.pids for p in processes)
            if ours:
                self.peak = max(self.peak or 0, sum(ours))
        except Exception as error:
            self.error = str(error)

    def start(self):
        try:
            if not self.pids:
                raise RuntimeError("GPU worker PIDs unavailable")
            import pynvml

            self.nv = pynvml
            self.nv.nvmlInit()
            self.initialized = True
            self.handle = self.nv.nvmlDeviceGetHandleByIndex(0)
            try:
                self.start_energy = self.nv.nvmlDeviceGetTotalEnergyConsumption(
                    self.handle
                )
            except Exception:
                self.start_energy = None
            self.sample()

            def loop():
                while not self.finished.wait(0.05):
                    self.sample()

            self.thread = threading.Thread(target=loop, daemon=True)
            self.thread.start()
        except Exception as error:
            self.error = str(error)
        return self

    def stop(self):
        self.finished.set()
        if self.thread:
            self.thread.join(timeout=2)
            self.sample()
        energy = None
        if self.start_energy is not None:
            try:
                end = self.nv.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                energy = max(0.0, (end - self.start_energy) / 1000)
            except Exception:
                pass
        if self.initialized:
            try:
                self.nv.nvmlShutdown()
            except Exception:
                pass
        return {
            "nvml_worker_process_peak_bytes": self.peak,
            "device_energy_j": energy if not self.foreign else None,
            "other_gpu_processes_present": self.foreign,
            "unavailable_reason": self.error,
            "scope": "50 ms sampled process footprint; device energy counter, not per-process energy; short-window resolution may be insufficient",
        }
