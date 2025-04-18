import cv2
import time
import subprocess

import numpy as np

from threading import Thread
from queue import Queue

class ArVideoWriter:

    def __init__(self, filename, fps, crf=16):
        self.initialized = False
        self.finished = False
        self.complete = False
        self.current_frame = 0
        self.filename = filename
        self.fps = fps
        self.crf = crf
        self.mask_img = cv2.imread("mask.png", cv2.IMREAD_UNCHANGED)
        self.process_buffer = Queue(maxsize=256)
        self.thread = Thread(target=self.run, args=())
        self.thread.daemon = True
        self.thread.start()

    def get_current_frame_number(self):
        return self.current_frame

    def add_frame(self, frame, alpha):
        while self.process_buffer.full():
            time.sleep(0.05)
        self.process_buffer.put((frame, alpha))

    def finalize(self):
        self.complete = True

    def is_finished(self):
        return self.finished

    def __setup(self, init_frame):
        self.initialized = True
        self.h, self.w = init_frame.shape[:2]
        self.mask_scaled = cv2.resize(self.mask_img, (int(self.w * 0.4), int(self.h * 0.4)))
        if len(self.mask_scaled .shape) != 2:
            self.mask_scaled  = self.mask_scaled [:, :, 0]

        self.overlay_positions = {
            'left_top': (self.w // 2 - int(0.4 * self.w/2) // 2, self.h - int(0.4 * self.h/2)),
            'left_bottom': (self.w // 2 - int(0.4 * self.w/2) // 2, 0),
            'right_top_left': (self.w - int(0.4 * self.w/4), self.h - int(0.4 * self.h/2)),
            'right_bottom_left': (self.w - int(0.4 * self.w/4), 0),
            'right_top_right': (0, self.h - int(0.4 * self.h/2)),
            'right_bottom_right': (0, 0)
        }

        self.ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.w}x{self.h}',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', 'libx265',
            '-preset', 'veryfast',
            '-pix_fmt', 'yuv420p',
            '-crf', str(self.crf),
            self.filename
        ]
        self.ffmpeg_process = subprocess.Popen(self.ffmpeg_cmd, stdin=subprocess.PIPE)


    def __process_frame(self, frame, alpha):
        if not self.initialized:
            self.__setup(frame)

        # 1. Scale mask video to 40% width (keep aspect ratio)
        alpha = cv2.resize(alpha, (int(self.w * 0.4), int(self.h * 0.4)))

        # 2. Extract alpha channel
        if len(alpha.shape) == 2:
            alpha_channel = alpha
        elif alpha.shape[2] == 4:
            alpha_channel = alpha[:, :, 3]
        else:
            alpha_channel = alpha[:, :, 0]

        # 3. Merge alpha with fisheye mask
        masked_alpha = cv2.merge([alpha_channel, alpha_channel, alpha_channel, self.mask_scaled])

        # 4. Split into left/right eyes
        split_x = masked_alpha.shape[1] // 2
        left_half = masked_alpha[:, :split_x]
        right_half = masked_alpha[:, split_x:]

        # 5. Create overlay regions (top/bottom for left, 4 for right)
        lh = left_half.shape[0]
        left_top = left_half[:lh//2, :]
        left_bottom = left_half[lh//2:, :]
        h_r, w_r = right_half.shape[:2]
        mid_y = h_r // 2
        mid_x = w_r // 2
        top_half = right_half[:mid_y, :]
        bottom_half = right_half[mid_y:, :]
        right_top_left  = top_half[:, :mid_x]
        right_top_right = top_half[:, mid_x:]
        right_bottom_left  = bottom_half[:, :mid_x]
        right_bottom_right = bottom_half[:, mid_x:]

        result = frame.copy()

        overlay_params = [
            (left_top, self.overlay_positions['left_top']),
            (left_bottom, self.overlay_positions['left_bottom']),
            (right_top_left, self.overlay_positions['right_top_left']),
            (right_bottom_left, self.overlay_positions['right_bottom_left']),
            (right_top_right, self.overlay_positions['right_top_right']),
            (right_bottom_right, self.overlay_positions['right_bottom_right'])
        ]

        def overlay_with_alpha(src, dst, x, y):
            h, w = src.shape[:2]
            dst_y1, dst_y2 = y, y + h
            dst_x1, dst_x2 = x, x + w

            dst_crop = dst[dst_y1:dst_y2, dst_x1:dst_x2]

            alpha = src[:, :, 3] / 255.0
            alpha = alpha[..., np.newaxis]

            dst_crop[:] = alpha * src[:, :, :3] + (1 - alpha) * dst_crop

        for overlay_img, (x, y) in overlay_params:
            overlay_with_alpha(overlay_img, result, x, y)
        
        return result


    def run(self):
        while self.process_buffer.qsize() > 0 or not self.complete:
            if self.process_buffer.qsize() < 1:
                time.sleep(0.05)
                continue

            (frame, alpha) = self.process_buffer.get()
            result_frame = self.__process_frame(frame, alpha)
            self.current_frame += 1
            self.ffmpeg_process.stdin.write(result_frame.tobytes())

        try: self.ffmpeg_process.stdin.close()
        except: pass
        try: self.ffmpeg_process.wait()
        except: pass
        self.finished = True


if __name__ == "__main__":
    src_cap = cv2.VideoCapture("a.mp4")
    mask_cap = cv2.VideoCapture("a-alpha.avi")
    writer = ArVideoWriter("out.mp4", src_cap.get(cv2.CAP_PROP_FPS))

    while True:
        ret_output, output_frame = src_cap.read()
        ret_mask, mask_frame = mask_cap.read()
        
        if not (ret_output and ret_mask):
            break

        writer.add_frame(output_frame, mask_frame)

    src_cap.release()
    mask_cap.release()
    writer.finalize()
