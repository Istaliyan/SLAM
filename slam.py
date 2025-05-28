import cv2
import pygame
import numpy as np
import open3d as o3d
import open3d.core as o3c

W = 640
H = 480
FPS = 30  # Frames per second

def create_point_cloud(keypoints, frame):
    # Extract 2D points from keypoints
    points_2d = np.float32([kp.pt for kp in keypoints])
    
    if len(points_2d) == 0:
        # Return empty point cloud if no keypoints found
        print("No keypoints found in the frame.")
        return o3d.geometry.PointCloud()
    
    # Create dummy depth values (all points at z=1 initially)
    depths = np.ones((len(points_2d), 1))
    
    # Create 3D points by adding the depth dimension
    points_3d = np.hstack((points_2d, depths))
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Add colors from the original frame
    colors = []
    for pt in points_2d:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            color = frame[y, x] / 255.0  # Normalize color values to [0,1]
            colors.append(color)
    
    # If we have colors, add them to the point cloud
    if colors:
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    else:
        # Add default colors (red) if we couldn't get colors from the frame
        default_color = np.array([[1.0, 0.0, 0.0] for _ in range(len(points_3d))])
        pcd.colors = o3d.utility.Vector3dVector(default_color)
    return pcd

def process_frame(frame):
    # Convert frame to grayscale for ORB detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
    
    # Detect keypoints
    keypoints = orb.detect(gray, None)
    
    # Compute descriptors
    keypoints, descriptors = orb.compute(gray, keypoints)
    
    # Create point cloud from keypoints
    point_cloud = create_point_cloud(keypoints, frame)
    
    # Optionally visualize point cloud
    # o3d.visualization.draw_geometries([point_cloud])
    
    # Draw keypoints on the frame
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

    return frame_with_keypoints, point_cloud

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()

    # Set up the display
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SLAM Visualization")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)

    # Initialize Open3D visualizer with rendering options
    vis = o3d.visualization.Visualizer()
    vis.create_window("Point Cloud Viewer", width=W, height=H)
    render_option = vis.get_render_option()
    render_option.point_size = 2.0  # Increase point size for better visibility
    # render_option.background_color = np.array([0, 0, 0])  # Black background
    
    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    vis.add_geometry(coord_frame)
    
    # Initialize video capture
    cap = cv2.VideoCapture('./test.mp4')

    # Initialize point cloud object for updating
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # Set up initial view
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, -1, 0])

    while cap.isOpened():
        clock.tick(FPS)
        current_fps = clock.get_fps()
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (W, H))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.swapaxes(0, 1)

        # Process the frame
        frame_processed, point_cloud = process_frame(frame_rgb)

        # Update point cloud in visualizer only if we have points
        if len(point_cloud.points) > 0:
            # Clear previous points
            vis.remove_geometry(pcd, False)
            # Update with new points
            pcd.points = point_cloud.points
            pcd.colors = point_cloud.colors
            vis.add_geometry(pcd, False)
            vis.poll_events()
            vis.update_renderer()
        
        # Pygame display
        surface = pygame.surfarray.make_surface(frame_processed)
        screen.blit(surface, (0, 0))
        fps_text = font.render(f'FPS: {int(current_fps)}', True, (255, 255, 255))
        screen.blit(fps_text, (10, 10))
        pygame.display.flip()

        # Handle quit event
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                vis.destroy_window()
                pygame.quit()
                exit()

    cap.release()
    vis.destroy_window()
    pygame.quit()
