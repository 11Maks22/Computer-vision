import os
import cv2
import matplotlib.pyplot as plt

VIDEO_PATH = "data/videoplayback.mp4"
NUM_FRAMES = 15


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def create_tracker(name):
    """
    Create OpenCV tracker with compatibility for different versions.
    """
    if name == "KCF":
        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerKCF_create()
        return cv2.TrackerKCF_create()

    elif name == "CSRT":
        if hasattr(cv2, "legacy"):
            return cv2.legacy.TrackerCSRT_create()
        return cv2.TrackerCSRT_create()

    else:
        raise ValueError("Unknown tracker")


def draw_bbox(frame, bbox, color, label):
    """
    Draw bounding box and label on frame.
    """
    x, y, w, h = map(int, bbox)

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.putText(
        frame,
        f"{label}: {x},{y},{w},{h}",
        (x, max(0, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )


def run_tracker(tracker_name, bbox):
    """
    Run tracker and collect bounding box data for analysis.
    """
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = cap.read()
    if not ret:
        raise Exception("Video not found")

    tracker = create_tracker(tracker_name)
    tracker.init(frame, bbox)

    save_dir = f"results/{tracker_name.lower()}"
    ensure_dir(save_dir)

    print(f"\n=== {tracker_name} ===")

    success_count = 0
    results = []  # <-- STORE BBOX DATA

    for i in range(NUM_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        success, bbox = tracker.update(frame)

        if success:
            success_count += 1

            x, y, w, h = map(int, bbox)
            results.append((x, y, w, h))  # <-- SAVE DATA

            color = (0, 255, 0) if tracker_name == "KCF" else (255, 0, 0)
            draw_bbox(frame, bbox, color, tracker_name)

            print(f"{tracker_name} frame {i}: {(x, y, w, h)}")

        else:
            print(f"{tracker_name} frame {i}: LOST")

            cv2.putText(
                frame,
                f"{tracker_name}: LOST",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imwrite(f"{save_dir}/frame_{i:02d}.jpg", frame)

    cap.release()

    print(f"{tracker_name} success: {success_count}/{NUM_FRAMES}")

    return results, success_count  # <-- RETURN DATA


def plot_results(kcf_data, csrt_data):
    """
    Plot tracking comparison (X, Y, Area).
    """
    def extract(data):
        xs = [d[0] for d in data]
        ys = [d[1] for d in data]
        areas = [d[2] * d[3] for d in data]
        return xs, ys, areas

    kcf_x, kcf_y, kcf_area = extract(kcf_data)
    csrt_x, csrt_y, csrt_area = extract(csrt_data)

    frames = list(range(len(kcf_data)))

    plt.figure(figsize=(12, 8))

    # X position
    plt.subplot(3, 1, 1)
    plt.plot(frames, kcf_x, label="KCF X")
    plt.plot(frames, csrt_x, label="CSRT X")
    plt.title("X Position")
    plt.legend()

    # Y position
    plt.subplot(3, 1, 2)
    plt.plot(frames, kcf_y, label="KCF Y")
    plt.plot(frames, csrt_y, label="CSRT Y")
    plt.title("Y Position")
    plt.legend()

    # Area
    plt.subplot(3, 1, 3)
    plt.plot(frames, kcf_area, label="KCF Area")
    plt.plot(frames, csrt_area, label="CSRT Area")
    plt.title("Bounding Box Area")
    plt.legend()

    ensure_dir("results")
    plt.tight_layout()
    plt.savefig("results/comparison_plot.png")
    plt.show()


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise Exception("Cannot read video")

    bbox = cv2.selectROI("Select object", frame, False)
    cv2.destroyAllWindows()

    kcf_data, kcf_success = run_tracker("KCF", bbox)
    csrt_data, csrt_success = run_tracker("CSRT", bbox)

    # 🔥 PLOT RESULTS
    plot_results(kcf_data, csrt_data)

    print("\n=== FINAL COMPARISON ===")
    print(f"KCF success: {kcf_success}/{NUM_FRAMES}")
    print(f"CSRT success: {csrt_success}/{NUM_FRAMES}")


if __name__ == "__main__":
    main()