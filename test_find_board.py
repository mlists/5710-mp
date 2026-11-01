"""Program to annotate images for testing the board detector."""
import argparse
import glob
import json
import os

import cv2
import numpy as np

from find_board import detect_chessboard


class CornerAnnotator:
    def __init__(self, image, window_name="Annotate corners"):
        self.image = image
        self.height, self.width = image.shape[:2]

        # Add space around the image so off-image corners can be selected.
        self.margin_x = max(150, int(self.width * 0.4))
        self.margin_y = max(150, int(self.height * 0.4))

        self.canvas = np.zeros(
            (self.height + 2 * self.margin_y,
             self.width + 2 * self.margin_x, 3),
            dtype=np.uint8
        )

        self.canvas[:] = (45, 45, 45)
        self.canvas[
            self.margin_y:self.margin_y + self.height,
            self.margin_x:self.margin_x + self.width
        ] = image

        self.canvas_height, self.canvas_width = self.canvas.shape[:2]

        self.points = []
        self.window_name = window_name

        # View state. Coordinates are in the padded canvas.
        self.zoom = 1.0
        self.view_x = 0.0
        self.view_y = 0.0

        self.min_zoom = 0.25
        self.max_zoom = 8.0
        self.zoom_step = 1.2

        self.dragging = False
        self.last_mouse = None

        # Start with the entire padded canvas visible.
        self.fit_view()

    def fit_view(self):
        screen_width = 1200
        screen_height = 800

        self.zoom = min(
            screen_width / self.canvas_width,
            screen_height / self.canvas_height,
            1.0
        )

        self.zoom = max(self.zoom, self.min_zoom)

        visible_width = screen_width / self.zoom
        visible_height = screen_height / self.zoom

        self.view_x = max(0, (self.canvas_width - visible_width) / 2)
        self.view_y = max(0, (self.canvas_height - visible_height) / 2)

    def screen_to_canvas(self, x, y):
        canvas_x = self.view_x + x / self.zoom
        canvas_y = self.view_y + y / self.zoom
        return canvas_x, canvas_y

    def canvas_to_image(self, x, y):
        return (
            x - self.margin_x,
            y - self.margin_y
        )

    def zoom_at(self, screen_x, screen_y, factor):
        # Find the canvas position underneath the cursor before zooming.
        canvas_x, canvas_y = self.screen_to_canvas(screen_x, screen_y)

        new_zoom = np.clip(
            self.zoom * factor,
            self.min_zoom,
            self.max_zoom
        )

        if new_zoom == self.zoom:
            return

        self.zoom = float(new_zoom)

        # Keep the same canvas position underneath the cursor.
        self.view_x = canvas_x - screen_x / self.zoom
        self.view_y = canvas_y - screen_y / self.zoom

        self.clamp_view()

    def clamp_view(self):
        visible_width = 1200 / self.zoom
        visible_height = 800 / self.zoom

        max_x = max(0, self.canvas_width - visible_width)
        max_y = max(0, self.canvas_height - visible_height)

        self.view_x = np.clip(self.view_x, 0, max_x)
        self.view_y = np.clip(self.view_y, 0, max_y)

    def redraw(self):
        screen_width = 1200
        screen_height = 800

        visible_width = max(1, int(screen_width / self.zoom))
        visible_height = max(1, int(screen_height / self.zoom))

        x1 = int(self.view_x)
        y1 = int(self.view_y)
        x2 = min(self.canvas_width, x1 + visible_width)
        y2 = min(self.canvas_height, y1 + visible_height)

        view = self.canvas[y1:y2, x1:x2].copy()

        # Convert marked points to coordinates within the current view.
        view_points = []
        for point in self.points:
            canvas_x = point[0] + self.margin_x
            canvas_y = point[1] + self.margin_y
            view_points.append((int(round(canvas_x - x1)), int(round(canvas_y - y1))))

        # Connect the marked corners in the order they were selected.
        # Once all four are marked, close the outline into a quadrilateral.
        if len(view_points) >= 2:
            cv2.polylines(
                view,
                [np.asarray(view_points, dtype=np.int32)],
                len(view_points) == 4,
                (0, 255, 0),
                max(1, int(round(2 / self.zoom))),
                cv2.LINE_AA
            )

        # Draw selected points and their numbers over the connecting lines.
        for index, (local_x, local_y) in enumerate(view_points):
            px = int(local_x * self.zoom)
            py = int(local_y * self.zoom)

            if -20 <= px < screen_width + 20 and -20 <= py < screen_height + 20:
                radius = max(3, int(7 / self.zoom))
                cv2.circle(view, (local_x, local_y), radius, (0, 255, 0), -1)
                cv2.putText(
                    view,
                    str(index + 1),
                    (local_x + radius + 3, local_y - radius - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.4, 0.6 / self.zoom),
                    (0, 255, 0),
                    2
                )

        display = cv2.resize(
            view,
            (max(1, int(view.shape[1] * self.zoom)),
             max(1, int(view.shape[0] * self.zoom))),
            interpolation=cv2.INTER_LINEAR if self.zoom > 1 else cv2.INTER_AREA
        )

        # Keep the display at a consistent size.
        output = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
        output[:] = (25, 25, 25)

        h = min(display.shape[0], screen_height)
        w = min(display.shape[1], screen_width)
        output[:h, :w] = display[:h, :w]

        instructions = (
            f"Points: {len(self.points)}/4 | Zoom: {self.zoom:.2f}x | "
            "Wheel: zoom | Middle drag: pan"
        )

        cv2.putText(
            output, instructions, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )

        cv2.putText(
            output, "Wheel / +/-: zoom | 0: fit | U: undo | R: reset | S: save | N: skip | Q: quit",
            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1
        )

        cv2.imshow(self.window_name, output)

    def mouse_callback(self, event, x, y, flags, param):

        # Scrolling logic, opencv does not expose getMouseWheelDelta for some reason...
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = (flags >> 16) & 0xffff

            if delta >= 32768:
                delta -= 65536

            if delta > 0:
                self.zoom_at(x, y, self.zoom_step)
            elif delta < 0:
                self.zoom_at(x, y, 1 / self.zoom_step)

        elif event == cv2.EVENT_MBUTTONDOWN:
            self.dragging = True
            self.last_mouse = (x, y)

        elif event == cv2.EVENT_MBUTTONUP:
            self.dragging = False
            self.last_mouse = None

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            if self.last_mouse is not None:
                dx = x - self.last_mouse[0]
                dy = y - self.last_mouse[1]

                self.view_x -= dx / self.zoom
                self.view_y -= dy / self.zoom

                self.clamp_view()
                self.last_mouse = (x, y)

        elif event == cv2.EVENT_LBUTTONDOWN:
            canvas_x, canvas_y = self.screen_to_canvas(x, y)
            image_x, image_y = self.canvas_to_image(canvas_x, canvas_y)

            if len(self.points) < 4:
                self.points.append((float(image_x), float(image_y)))

        self.redraw()

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1200, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.redraw()

        while True:
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("+"), ord("=")):
                self.zoom_at(600, 400, self.zoom_step)
                self.redraw()

            elif key in (ord("-"), ord("_")):
                self.zoom_at(600, 400, 1 / self.zoom_step)
                self.redraw()

            elif key == ord("0"):
                self.fit_view()
                self.redraw()

            elif key == ord("u"):
                if self.points:
                    self.points.pop()
                self.redraw()

            elif key == ord("r"):
                self.points = []
                self.redraw()

            elif key == ord("s"):
                if len(self.points) == 4:
                    cv2.destroyWindow(self.window_name)
                    return self.points, "save"

            elif key == ord("n"):
                cv2.destroyWindow(self.window_name)
                return None, "skip"

            elif key == ord("q") or key == 27:
                cv2.destroyWindow(self.window_name)
                return None, "quit"


def load_images(image_dir):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths = []

    for extension in extensions:
        paths.extend(glob.glob(os.path.join(image_dir, extension)))

    return sorted(paths)


def load_annotations(annotation_file):
    if os.path.exists(annotation_file):
        with open(annotation_file, "r") as file:
            return json.load(file)

    return {}


def save_annotations(annotation_file, annotations):
    os.makedirs(os.path.dirname(os.path.abspath(annotation_file)), exist_ok=True)

    with open(annotation_file, "w") as file:
        json.dump(annotations, file, indent=4)


def annotate_images(image_paths, annotation_file):
    annotations = load_annotations(annotation_file)

    for image_path in image_paths:
        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not load {image_path}")
            continue

        print(f"\nAnnotating: {image_path}")

        annotator = CornerAnnotator(image)
        points, action = annotator.run()

        if action == "quit":
            break

        if action == "skip":
            continue

        annotations[os.path.abspath(image_path)] = points
        save_annotations(annotation_file, annotations)

        print(f"Saved annotations for {image_path}")

    save_annotations(annotation_file, annotations)


def align_corners(reference, detected):
    """Find the best alignment across the eight possible board symmetries."""
    reference = np.asarray(reference, dtype=np.float32)
    detected = np.asarray(detected, dtype=np.float32)

    best_detected = None
    best_error = float("inf")

    for candidate in (detected, detected[::-1]):
        for shift in range(4):
            aligned = np.roll(candidate, shift, axis=0)
            error = np.mean(np.linalg.norm(reference - aligned, axis=1))

            if error < best_error:
                best_error = error
                best_detected = aligned.copy()

    return best_detected


def draw_comparison(image, manual, detected, errors, max_distance):
    display = image.copy()

    manual_int = np.round(manual).astype(np.int32)
    detected_int = np.round(detected).astype(np.int32)

    # Draw the manual and detected board boundaries.
    cv2.polylines(display, [manual_int], True, (0, 255, 0), 2)
    cv2.polylines(display, [detected_int], True, (0, 0, 255), 2)

    for index in range(4):
        mx, my = manual_int[index]
        dx, dy = detected_int[index]

        colour = (255, 255, 0) if errors[index] <= max_distance else (0, 165, 255)

        cv2.circle(display, (mx, my), 6, (0, 255, 0), -1)
        cv2.circle(display, (dx, dy), 6, (0, 0, 255), -1)
        cv2.line(display, (mx, my), (dx, dy), colour, 2)

        cv2.putText(
            display,
            f"{errors[index]:.1f}px",
            (mx + 8, my - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1
        )

    return display


def evaluate_images(image_paths, annotation_file, max_distance=5.0):
    annotations = load_annotations(annotation_file)

    results = []
    index = 0

    while 0 <= index < len(image_paths):
        image_path = image_paths[index]
        absolute_path = os.path.abspath(image_path)

        if absolute_path not in annotations:
            print(f"No annotation found for {image_path}")
            index += 1
            continue

        image = cv2.imread(image_path)

        if image is None:
            index += 1
            continue

        manual = np.asarray(annotations[absolute_path], dtype=np.float32)

        try:
            _, detected, _ = detect_chessboard(
                image,
                video_mode=True,
                display_output=False,
                display_debug=False,
                use_standard_detector=False
            )

            detected = align_corners(manual, detected)
            errors = np.linalg.norm(manual - detected, axis=1)

            results.extend(errors.tolist())

            display = draw_comparison(image, manual, detected, errors, max_distance)

            mean_error = np.mean(errors)
            cv2.putText(
                display,
                f"Mean error: {mean_error:.2f}px",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            print(f"\n{image_path}")
            print(f"Corner errors: {np.round(errors, 2)}")
            print(f"Mean error: {mean_error:.2f}px")

        except Exception as error:
            print(f"\nDetection failed for {image_path}: {error}")
            display = image.copy()
            cv2.putText(
                display, "DETECTION FAILED", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        cv2.imshow("Corner evaluation", display)

        print("N: next | P: previous | Q: quit")
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("p"):
            index = max(0, index - 1)
        else:
            index += 1

    cv2.destroyAllWindows()

    if results:
        results = np.asarray(results)

        print("\n===== Overall results =====")
        print(f"Mean error:   {np.mean(results):.2f}px")
        print(f"Median error: {np.median(results):.2f}px")
        print(f"Max error:    {np.max(results):.2f}px")
        print(f"Within {max_distance}px: {np.mean(results <= max_distance) * 100:.1f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["annotate", "evaluate"], required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--annotations", default="corner_annotations.json")
    parser.add_argument("--max-distance", type=float, default=5.0)

    args = parser.parse_args()

    image_paths = load_images(args.images)

    if not image_paths:
        print("No images found.")
        return

    if args.mode == "annotate":
        annotate_images(image_paths, args.annotations)

    elif args.mode == "evaluate":
        evaluate_images(image_paths, args.annotations, args.max_distance)


if __name__ == "__main__":
    main()
