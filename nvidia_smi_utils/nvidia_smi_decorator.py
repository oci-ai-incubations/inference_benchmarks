from contextlib import contextmanager
import functools
import subprocess
import threading


@contextmanager
def run_nvidia_smi():
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,index,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used,compute_mode,clocks.current.sm,clocks.current.memory,power.draw,power.limit,fan.speed,pcie.link.width.current,encoder.stats.sessionCount,encoder.stats.averageFps,encoder.stats.averageLatency,vbios_version,inforom.img,gpu_uuid,gpu_serial",
        "--format=csv",
        "-l", "1"
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    
    def read_output(output, output_list):
        """Reads output from stdout asynchronously."""
        for line in iter(output.readline, ''):
            if line:
                output_list.append(line)
        output.close()
    
    stdout_lines = []
    
    # Start a thread to read the output asynchronously
    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_lines), daemon=True)
    stdout_thread.start()
    
    try:
        yield stdout_lines  # Yield the list capturing stdout incrementally
    finally:
        process.terminate()
        stdout_thread.join()  # Ensure the reading thread finishes
        _, stderr = process.communicate()  # Fetch any remaining error output
        if stderr:
            print(f"nvidia-smi error: {stderr}")

def nvidia_smi_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with run_nvidia_smi() as nvidia_smi_output:
            result = func(*args, **kwargs)
        
        # Collect the nvidia-smi output from the list
        nvidia_smi_output = ''.join(nvidia_smi_output)
        
        return result, nvidia_smi_output
    return wrapper