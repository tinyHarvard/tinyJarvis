import cv2
import mediapipe as mp
import numpy as np
import math

# --- Mediapipe Hand Tracking Setup ---
mp_hands = mp.solutions.hands
# Initialize Hands:
# - static_image_mode=False: Process video frames.
# - max_num_hands=1: Track only one hand for simplicity.
# - min_detection_confidence=0.7: Minimum confidence for hand detection.
# - min_tracking_confidence=0.5: Minimum confidence for hand tracking.
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Cube Definition ---
# Define the 8 vertices of a cube centered at (0,0,0) with side length 2
# We'll scale this later.
cube_vertices_3d_normalized = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face (z=-1)
    [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]  # Front face (z=1)
], dtype=np.float32)

# Define the 12 edges of the cube by specifying pairs of vertex indices
cube_edges = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # Back face edges
    (4, 5), (5, 6), (6, 7), (7, 4),  # Front face edges
    (0, 4), (1, 5), (2, 6), (3, 7)  # Connecting edges between front and back
]

# --- Initial Cube State ---
cube_scale = 50.0  # Initial size of the cube
rotation_x = 0.0  # Rotation around X-axis (radians)
rotation_y = 0.0  # Rotation around Y-axis (radians)
rotation_z = 0.0  # Rotation around Z-axis (radians)

# Translation places the cube in the 3D world.
# We'll keep Tx and Ty at 0 to center it on the projection axis.
# Tz moves it away from the camera to make it visible.
translation_x = 0.0
translation_y = 0.0
translation_z = 400.0  # Distance from camera; larger values make it appear smaller/further


# --- Projection Function ---
def project_points_3d_to_2d(points_3d, R_x, R_y, R_z, T_x, T_y, T_z, focal_length, screen_width, screen_height):
    """
    Projects 3D points to 2D screen coordinates.
    Args:
        points_3d: NumPy array of 3D points (N, 3).
        R_x, R_y, R_z: Rotation angles around X, Y, Z axes (radians).
        T_x, T_y, T_z: Translation along X, Y, Z axes.
        focal_length: Focal length of the virtual camera.
        screen_width, screen_height: Dimensions of the output screen.
    Returns:
        NumPy array of 2D projected points (N, 2).
    """
    # Rotation matrices
    rot_mat_x = np.array([
        [1, 0, 0],
        [0, math.cos(R_x), -math.sin(R_x)],
        [0, math.sin(R_x), math.cos(R_x)]
    ])
    rot_mat_y = np.array([
        [math.cos(R_y), 0, math.sin(R_y)],
        [0, 1, 0],
        [-math.sin(R_y), 0, math.cos(R_y)]
    ])
    rot_mat_z = np.array([
        [math.cos(R_z), -math.sin(R_z), 0],
        [math.sin(R_z), math.cos(R_z), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix (Z*Y*X order)
    rotation_matrix = rot_mat_z @ rot_mat_y @ rot_mat_x

    # Apply rotation to each point
    # points_3d is (N,3), rotation_matrix is (3,3). We need (N,3) @ (3,3) -> (N,3)
    # So, we transpose points_3d, multiply, then transpose back. (R @ P.T).T
    rotated_points = (rotation_matrix @ points_3d.T).T

    # Apply translation
    translated_points = rotated_points + np.array([T_x, T_y, T_z])

    # Perspective projection
    projected_points_2d = []
    for p in translated_points:
        # Ensure Z is not zero to avoid division by zero.
        # Points behind the camera (p[2] <= 0) are problematic;
        # ideally, they should be clipped. For simplicity, we'll project them,
        # but they might behave strangely. A robust solution would clip.
        z_val = p[2] if p[2] != 0 else 0.00001

        x_proj = (focal_length * p[0] / z_val) + screen_width / 2
        y_proj = (focal_length * p[1] / z_val) + screen_height / 2  # Y is often inverted in screen coords
        projected_points_2d.append([int(x_proj), int(y_proj)])

    return np.array(projected_points_2d)


# --- Drawing Function ---
def draw_cube_on_image(image, projected_vertices, edges, color=(0, 255, 0), thickness=2):
    """Draws the cube on the image using its projected 2D vertices."""
    if projected_vertices.shape[0] == 0:  # No vertices to draw
        return
    for edge_indices in edges:
        pt1_index, pt2_index = edge_indices
        # Ensure indices are within bounds
        if pt1_index < len(projected_vertices) and pt2_index < len(projected_vertices):
            pt1 = tuple(projected_vertices[pt1_index])
            pt2 = tuple(projected_vertices[pt2_index])
            cv2.line(image, pt1, pt2, color, thickness)


# --- Main Application Loop ---
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Store previous hand landmarks for smoothing or gesture recognition (optional)
prev_hand_landmarks = None
# Store initial pinch distance for relative scaling
initial_pinch_distance = None
pinch_scale_factor = 1.0  # Multiplier for cube_scale based on pinch

# Initial hand position for relative rotation
initial_hand_x_for_rotation = None
initial_hand_y_for_rotation = None
base_rotation_x = 0.0
base_rotation_y = 0.0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Get image dimensions
    img_h, img_w, _ = image.shape

    # Flip the image horizontally for a selfie-view display.
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB before processing.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    rgb_image.flags.writeable = False
    results = hands.process(rgb_image)
    rgb_image.flags.writeable = True  # Back to writeable

    # Convert the RGB image back to BGR for OpenCV display.
    # We do this here because drawing happens on `image` (BGR)
    # image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) # This was already done implicitly by not overwriting `image`

    active_hand_landmarks = None
    if results.multi_hand_landmarks:
        # We are tracking only one hand (max_num_hands=1)
        active_hand_landmarks = results.multi_hand_landmarks[0]

        # --- Gesture Interpretation ---
        # 1. Scaling: Distance between thumb tip (landmark 4) and index finger tip (landmark 8)
        thumb_tip = active_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_finger_tip = active_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Convert normalized coordinates (0.0-1.0) to pixel coordinates
        thumb_px = (thumb_tip.x * img_w, thumb_tip.y * img_h)
        index_px = (index_finger_tip.x * img_w, index_finger_tip.y * img_h)

        current_pinch_distance = math.hypot(index_px[0] - thumb_px[0], index_px[1] - thumb_px[1])

        if initial_pinch_distance is None:
            initial_pinch_distance = current_pinch_distance  # Set on first detection

        if initial_pinch_distance > 1e-6:  # Avoid division by zero
            pinch_scale_factor = current_pinch_distance / initial_pinch_distance

        # Apply scaling factor to the base cube scale. Clamp to avoid extreme sizes.
        cube_scale = 50.0 * pinch_scale_factor
        cube_scale = np.clip(cube_scale, 10, 200)  # Min scale 10, Max scale 200

        # 2. Rotation: Wrist x-position for rotation_y, wrist y-position for rotation_x
        wrist = active_hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]

        # Use relative movement for rotation
        if initial_hand_x_for_rotation is None:
            initial_hand_x_for_rotation = wrist.x
            initial_hand_y_for_rotation = wrist.y
            base_rotation_x = rotation_x  # Store current rotation as base
            base_rotation_y = rotation_y

        # Calculate change in hand position from initial
        delta_hand_x = wrist.x - initial_hand_x_for_rotation
        delta_hand_y = wrist.y - initial_hand_y_for_rotation

        # Map hand movement to rotation angles.
        # Sensitivity factor: how much rotation per unit of hand movement.
        # A larger movement range (e.g., 0.0 to 1.0 for wrist.x) maps to a rotation range (e.g., -pi to pi).
        sensitivity = math.pi  # e.g., moving hand across screen width rotates PI radians

        rotation_y = base_rotation_y + (delta_hand_x * sensitivity)
        rotation_x = base_rotation_x - (delta_hand_y * sensitivity)  # '-' for natural up/down mapping

        # Draw hand landmarks on the image
        mp_drawing.draw_landmarks(
            image,
            active_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

        prev_hand_landmarks = active_hand_landmarks
    else:
        # If hand is lost, reset initial states for gestures
        initial_pinch_distance = None
        initial_hand_x_for_rotation = None
        initial_hand_y_for_rotation = None
        # Optionally, reset cube to a default state or keep last known
        # For now, keep last known rotation, but reset pinch scale factor
        pinch_scale_factor = 1.0
        cube_scale = 50.0 * pinch_scale_factor
        base_rotation_x = rotation_x  # Update base so it doesn't snap when hand reappears
        base_rotation_y = rotation_y

    # --- Cube Transformation and Rendering ---
    # Apply current scale to the normalized cube vertices
    scaled_cube_vertices = cube_vertices_3d_normalized * cube_scale

    # Project the 3D scaled and rotated vertices to 2D screen points
    # Focal length can be tuned: higher values -> less perspective distortion (more orthographic-like)
    # lower values -> more perspective distortion.
    focal_length_k = 300
    projected_2d_vertices = project_points_3d_to_2d(
        scaled_cube_vertices,
        rotation_x, rotation_y, rotation_z,  # Current rotations
        translation_x, translation_y, translation_z,  # Fixed translation
        focal_length_k,
        img_w, img_h
    )

    # Draw the 2D projected cube on the image
    draw_cube_on_image(image, projected_2d_vertices, cube_edges, color=(0, 255, 0), thickness=2)

    # Display the resulting image
    cv2.imshow('Hand Gesture Cube Controller - Press ESC to Exit', image)

    # Exit loop if 'ESC' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# --- Release resources ---
hands.close()
cap.release()
cv2.destroyAllWindows()
