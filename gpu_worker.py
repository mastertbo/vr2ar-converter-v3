import torch, logging, traceback
from main import process, logger

def worker_main(gpu_id: int, task_q, done_q):
    torch.cuda.set_device(gpu_id)
    logger.info(f"GPU{gpu_id}: worker online")
    while True:
        job = task_q.get()
        if job is None:             # optional sentinel
            logger.info(f"GPU{gpu_id}: shutdown")
            break
        try:
            out = process(
                job["id"], job["video"], job["projection"],
                job["maskL"], job["maskR"], job["crf"], job["erode"],
                gpu_id=gpu_id
            )
            done_q.put(out)
        except Exception as e:
            err = f"ERROR:{job['id']} on GPU{gpu_id}: {e}\n{traceback.format_exc()}"
            done_q.put(err)
            logger.error(err)