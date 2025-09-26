"""
Demonstration of the display module components.

This script shows how to use WindowManager and InputHandler together
for basic window management and input processing.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path to import display module
sys.path.insert(0, str(Path(__file__).parent.parent))

from display import WindowManager, InputHandler, CameraState


def main():
    """Main demonstration function."""
    print("Display Module Demo")
    print("==================")
    
    # Create window manager and input handler
    window_manager = WindowManager()
    input_handler = InputHandler()
    
    try:
        # Create window
        print("Creating window...")
        if not window_manager.create_window(800, 600, "Display Module Demo"):
            print("Failed to create window!")
            return
        
        print("Window created successfully!")
        
        # Set up input handler with the window
        window_handle = window_manager.get_window_handle()
        input_handler.set_window(window_handle)
        
        print("Input handler configured!")
        
        # Register some input callbacks
        def on_space_pressed(action, mods):
            if action == 1:  # GLFW_PRESS
                print("Space key pressed! Resetting camera...")
                input_handler.reset_camera()
        
        def on_escape_pressed(action, mods):
            if action == 1:  # GLFW_PRESS
                print("Escape pressed! Exiting...")
                # In a real application, you'd set a flag to exit the main loop
        
        input_handler.register_key_callback(32, on_space_pressed)  # Space
        input_handler.register_key_callback(256, on_escape_pressed)  # Escape
        
        print("\nControls:")
        print("- WASD: Move camera")
        print("- QE: Move up/down")
        print("- Right mouse + drag: Look around")
        print("- Scroll: Zoom")
        print("- Space: Reset camera")
        print("- Escape: Exit")
        print("\nPress Ctrl+C to exit the demo")
        
        # Simple main loop (just for demonstration)
        last_time = time.time()
        frame_count = 0
        
        while not window_manager.should_close():
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            # Poll events
            window_manager.poll_events()
            
            # Update input handler
            input_handler.update_camera_controls(dt)
            
            # Get camera state for demonstration
            camera_state = input_handler.get_camera_state()
            
            # Print camera info every 60 frames (roughly once per second at 60 FPS)
            frame_count += 1
            if frame_count % 60 == 0:
                print(f"Camera - Pos: ({camera_state.position_x:.1f}, "
                      f"{camera_state.position_y:.1f}, {camera_state.position_z:.1f}), "
                      f"Rot: ({camera_state.rotation_x:.2f}, {camera_state.rotation_y:.2f}), "
                      f"Zoom: {camera_state.zoom:.2f}")
            
            # Swap buffers
            window_manager.swap_buffers()
            
            # Simple frame rate limiting (not precise, just for demo)
            time.sleep(0.016)  # ~60 FPS
    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    
    except Exception as e:
        print(f"Error during demo: {e}")
    
    finally:
        # Clean up
        print("Cleaning up...")
        window_manager.destroy()
        print("Demo finished!")


if __name__ == "__main__":
    main()