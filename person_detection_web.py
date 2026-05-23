"""
=============================================================================
  PERSON DETECTION - Tối ưu tốc độ (GMM contour ROI, không sliding window)
  Chạy: python camera_detect.py
  Q = thoát | SPACE = pause | S = chụp | +/- = confidence
=============================================================================
"""

import cv2
import numpy as np
import joblib
import time
import os
from collections import deque
from skimage.feature import hog as sk_hog


MODEL_PATH        = 'person_detector_pipeline.pkl'
CAMERA_INDEX      = 0
CONFIDENCE_THRESH = 0.88
PROCESS_W         = 320
PROCESS_H         = 240
MIN_ROI_AREA      = 140    # đủ nhạy với chuyển động chậm
SMOOTH_FRAMES     = 5      # số frame để vote số người
# ============================================================


def preprocess_image(img, img_size=(64, 128)):
    img = cv2.resize(img, img_size)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    mean_b = np.mean(gray) / 255.0
    gamma = 0.5 if mean_b < 0.4 else (1.5 if mean_b > 0.7 else 1.0)
    if gamma != 1.0:
        table = np.array([((i/255.0)**(1.0/gamma))*255 for i in range(256)]).astype('uint8')
        gray = cv2.LUT(gray, table)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray.astype(np.float32) / 255.0


def extract_hog(img_proc, cfg):
    return sk_hog(
        img_proc,
        orientations=cfg['hog_orientations'],
        pixels_per_cell=tuple(cfg['hog_pix_per_cell']),
        cells_per_block=tuple(cfg['hog_cells_per_block']),
        block_norm=cfg['hog_block_norm'],
        visualize=False, feature_vector=True
    )


def nms(boxes, scores, thresh=0.3):
    """NMS threshold thấp = merge box tích cực hơn."""
    if not boxes: return [], []
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores)
    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,0]+boxes[:,2], boxes[:,1]+boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)
        iou = inter/(areas[i]+areas[order[1:]]-inter+1e-6)
        order = order[np.where(iou<=thresh)[0]+1]
    return boxes[keep].astype(int).tolist(), scores[keep].tolist()


def merge_rois(rois, overlap=0.2):
    """Gộp ROI chồng nhau trước khi predict."""
    if not rois: return []
    merged = True
    rois = list(rois)
    while merged:
        merged = False
        result = []
        used = [False] * len(rois)
        for i in range(len(rois)):
            if used[i]: continue
            x1, y1, w1, h1 = rois[i]
            for j in range(i+1, len(rois)):
                if used[j]: continue
                x2, y2, w2, h2 = rois[j]
                ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                inter = ix * iy
                union = w1*h1 + w2*h2 - inter
                if union > 0 and inter/union > overlap:
                    nx = min(x1,x2); ny = min(y1,y2)
                    nw = max(x1+w1,x2+w2)-nx
                    nh = max(y1+h1,y2+h2)-ny
                    rois[i] = (nx, ny, nw, nh)
                    used[j] = True
                    merged = True
            result.append(rois[i])
        rois = result
    return rois


class PersonDetector:
    def __init__(self, model_path):
        print(f'🔄 Loading model: {model_path}')
        p = joblib.load(model_path)
        self.model    = p['model']
        self.scaler   = p['scaler']
        self.cfg      = p['config']
        self.img_size = tuple(self.cfg['img_size'])
        self.conf_thresh = CONFIDENCE_THRESH

        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=25, detectShadows=False)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # Buffer để vote số người ổn định
        self.count_buffer = deque(maxlen=SMOOTH_FRAMES)
        self.boxes_buffer = []
        self.scores_buffer = []

        print(f'✅ Ready! Window={self.img_size} | Conf={self.conf_thresh:.0%}')

    def get_rois(self, frame_small):
        fg = self.bg.apply(frame_small)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.kernel)
        fg = cv2.dilate(fg, self.kernel, iterations=4)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rois = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_ROI_AREA:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            pad_x = max(int(w*0.25), 10)
            pad_y = max(int(h*0.25), 10)
            x = max(0, x-pad_x); y = max(0, y-pad_y)
            w = min(frame_small.shape[1]-x, w+2*pad_x)
            h = min(frame_small.shape[0]-y, h+2*pad_y)
            if h > w * 0.4:
                rois.append((x, y, w, h))
        return rois, fg

    def predict_roi(self, frame, box):
        x, y, w, h = box
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0: return 0, 0.0
        proc = preprocess_image(roi, self.img_size)
        feat = extract_hog(proc, self.cfg)
        feat_s = self.scaler.transform(feat.reshape(1, -1))
        label = self.model.predict(feat_s)[0]
        prob  = self.model.predict_proba(feat_s)[0][1]
        return label, prob

    def process(self, frame):
        h_orig, w_orig = frame.shape[:2]
        small = cv2.resize(frame, (PROCESS_W, PROCESS_H))
        sx = w_orig / PROCESS_W
        sy = h_orig / PROCESS_H

        rois, fg_mask = self.get_rois(small)
        rois = merge_rois(rois, overlap=0.2)

        boxes, scores = [], []
        for (x, y, w, h) in rois:
            x0=int(x*sx); y0=int(y*sy)
            w0=int(w*sx); h0=int(h*sy)
            label, prob = self.predict_roi(frame, (x0, y0, w0, h0))
            if label == 1 and prob >= self.conf_thresh:
                boxes.append([x0, y0, w0, h0])
                scores.append(prob)

        boxes, scores = nms(boxes, scores, 0.3)

        # Lọc box không giống người đứng (loại tay, đầu, vật thể nhỏ)
        filtered_boxes, filtered_scores = [], []
        for box, score in zip(boxes, scores):
            x, y, bw, bh = box
            ratio = bh / max(bw, 1)
            if ratio >= 1.0:  # cao hơn rộng ít nhất 1.0 lần
                filtered_boxes.append(box)
                filtered_scores.append(score)
        boxes, scores = filtered_boxes, filtered_scores

        # Vote số người qua SMOOTH_FRAMES frame
        self.count_buffer.append(len(boxes))
        stable_count = int(np.median(self.count_buffer))

        # Nếu số người ổn định thì cập nhật boxes
        if len(boxes) == stable_count:
            self.boxes_buffer  = boxes
            self.scores_buffer = scores

        fg_display = cv2.resize(fg_mask, (160, 120))
        fg_display = cv2.cvtColor(fg_display, cv2.COLOR_GRAY2BGR)
        return self.boxes_buffer, self.scores_buffer, fg_display, len(rois), stable_count


def main():
    if not os.path.exists(MODEL_PATH):
        print(f'❌ Không tìm thấy: {MODEL_PATH}')
        return

    detector = PersonDetector(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print(f'❌ Không mở được camera {CAMERA_INDEX}')
        return

    cv2.namedWindow('Person Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Person Detection', 900, 560)

    boxes, scores = [], []
    stable_count = 0
    fps = 0.0
    frame_count = 0
    t0 = time.time()
    paused = False
    shot_count = 0
    frame = None
    n_rois = 0

    print('\n🎥 Đang chạy...')
    print('   Q=thoát | SPACE=pause | S=chụp | +=tăng conf | -=giảm conf\n')

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: break

            boxes, scores, fg_disp, n_rois, stable_count = detector.process(frame)

            frame_count += 1
            elapsed = time.time() - t0
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                t0 = time.time()

        if frame is None:
            continue

        out = frame.copy()
        h, w = out.shape[:2]

        # Bounding boxes
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x, y, bw, bh = box
            cv2.rectangle(out, (x,y), (x+bw,y+bh), (0,255,80), 2)
            txt = f'#{i+1} {score:.0%}'
            (tw,th),_ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(out, (x,y-th-8), (x+tw+6,y), (0,255,80), -1)
            cv2.putText(out, txt, (x+3,y-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

        # GMM mask góc trên phải
        out[8:128, w-168:w-8] = fg_disp
        cv2.rectangle(out, (w-168,8), (w-8,128), (80,80,80), 1)
        cv2.putText(out, 'GMM Mask', (w-155,143),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150), 1)

        # Header — dùng stable_count để hiển thị ổn định
        cv2.rectangle(out, (0,0), (w,52), (15,15,15), -1)
        color = (0,255,80) if stable_count > 0 else (160,160,160)
        cv2.putText(out, f'People: {stable_count}', (12,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
        cv2.putText(out, f'FPS:{fps:.1f}', (w-120,36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,200,255), 2)

        cv2.putText(out, f'Conf:{detector.conf_thresh:.0%}  ROI:{n_rois}',
                    (12,72), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,0), 1)

        if paused:
            cv2.rectangle(out, (0,0), (w,h), (0,0,180), 3)
            cv2.putText(out, 'PAUSED', (w//2-80,h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,80,255), 3)

        cv2.rectangle(out, (0,h-26), (w,h), (15,15,15), -1)
        cv2.putText(out, 'Q:Thoat  SPACE:Pause  S:Chup  +/-:Confidence',
                    (10,h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100,100,100), 1)

        cv2.imshow('Person Detection', out)

        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), 27]:
            break
        elif key == ord(' '):
            paused = not paused
            print('⏸ Paused' if paused else '▶ Resumed')
        elif key == ord('s'):
            shot_count += 1
            fname = f'shot_{shot_count:03d}.jpg'
            cv2.imwrite(fname, out)
            print(f'📸 {fname}')
        elif key in [ord('+'), ord('=')]:
            detector.conf_thresh = min(detector.conf_thresh+0.05, 0.99)
            print(f'🔼 Confidence: {detector.conf_thresh:.0%}')
        elif key == ord('-'):
            detector.conf_thresh = max(detector.conf_thresh-0.05, 0.50)
            print(f'🔽 Confidence: {detector.conf_thresh:.0%}')

    cap.release()
    cv2.destroyAllWindows()
    print('✅ Thoát!')


if __name__ == '__main__':
    main()