import cv2
from pathlib import Path
from ultralytics import YOLO
import torch

VIDEOS_DIR = Path("videos")
OUTPUT_DIR = Path("output")
FRAME_STEP = 10
CONFIDENCE_THRESHOLD = 0.3
#YOLO_MODEL = "yolov8n.pt"
YOLO_MODEL = "./models/yolov8-fish.pt"
SHOW_PREVIEW = False

model = YOLO(YOLO_MODEL)

def save_yolo_txt(path, detections, img_w, img_h):
    with open(path, "w") as f:
        for det in detections:
            cls_id = det["class_id"]
            x1, y1, x2, y2 = det["box"]
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

def process_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx, saved_idx = 0, 0
    video_name = video_path.stem

    images_dir = OUTPUT_DIR / "images"
    labels_dir = OUTPUT_DIR / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n🎬 Traitement de {video_path.name}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_STEP == 0:
            h, w, _ = frame.shape
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

            detections = []
            detected_labels = []
            for box in results.boxes:
                cls_id = int(box.cls)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({"class_id": cls_id, "box": (x1, y1, x2, y2)})
                detected_labels.append(model.names[cls_id])
                if SHOW_PREVIEW:
                    label = model.names[cls_id]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if detections:
                name = f"{video_name}_i{frame_idx}"
                image_path = images_dir / f"{name}.jpg"
                if image_path.exists():
                    print(f"⏩ {name}.jpg déjà présent, passage à la frame suivante.")
                else:
                    cv2.imwrite(str(image_path), frame)
                    save_yolo_txt(labels_dir / f"{name}.txt", detections, w, h)
                    print(f"✅ {name}.jpg enregistré ({len(detections)} objets) : {', '.join(detected_labels)}")
                    saved_idx += 1
            else:
                print(f"⛔ frame {frame_idx}: aucune détection")

            if SHOW_PREVIEW:
                cv2.imshow("Prévisualisation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("🛑 Interruption par l'utilisateur.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

        frame_idx += 1

    cap.release()
    if SHOW_PREVIEW:
        cv2.destroyAllWindows()

def main():
    videos = list(VIDEOS_DIR.glob("*.MP4")) + list(VIDEOS_DIR.glob("*.mp4")) + list(VIDEOS_DIR.glob("*.avi")) + list(VIDEOS_DIR.glob("*.mov"))
    if not videos:
        print("❌ Aucune vidéo trouvée dans le dossier 'videos/'")
        return

    for video_path in videos:
        process_video(video_path)

    with open(OUTPUT_DIR / "classes.txt", "w") as f:
        for name in model.names.values():
            f.write(f"{name}\n")
        print(f"\n📁 Fichier 'classes.txt' généré avec {len(name)} classes.")

    print("\n✅ Tous les traitements sont terminés.")

if __name__ == "__main__":
    main()
