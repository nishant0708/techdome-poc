import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import gc
torch.cuda.empty_cache()
gc.collect()

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "./videos"

# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# take a look the first video frame
frame_idx = 0
plt.figure(figsize=(9, 6))
plt.title(f"frame {frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[frame_idx])))


inference_state = predictor.init_state(video_path=video_dir)

prompts = {}  # hold all the clicks we add for visualization


# click on the object you want to track

ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

# Let's add a positive click at (x, y) = (200, 300) to get started on the first object
points = np.array([[400, 500]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 3  # give a unique id to each object we interact with (it can be any integers)

# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[950, 400]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels

# `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)


ann_frame_idx = 0  # the frame index we interact with
ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's now move on to the second object we want to track (giving it object id `3`)
# with a positive click at (x, y) = (400, 150)
points = np.array([[1100, 450]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
labels = np.array([1], np.int32)
prompts[ann_obj_id] = points, labels

# `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels,
)

# show the results on the current (interacted) frame on all objects
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
for i, out_obj_id in enumerate(out_obj_ids):
    show_points(*prompts[out_obj_id], plt.gca())
    show_mask((out_mask_logits[i] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_id)


video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
vis_frame_stride=30
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)



import cv2
import numpy as np
import os

def create_masked_video(frame_names, video_segments, output_path, fps=30):
    """
    Create a video with segmentation masks overlaid on the frames with different colors per object.
    """
    # Color mapping for different objects (BGR format)
    color_map = {
        0: [0, 255, 0],    # Green
        1: [0, 0, 255],    # Red
        2: [255, 165, 0],  # Blue
        3: [255, 0, 255],  # Magenta
        4: [0, 255, 255],  # Yellow
        5: [128, 0, 128],  # Purple
        # Add more colors as needed
    }
    
    # Debug information
    print(f"Number of frames: {len(frame_names)}")
    print(f"Number of segments: {len(video_segments)}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_names[0])
    if first_frame is None:
        raise ValueError(f"Could not read frame: {frame_names[0]}")
    
    height, width = first_frame.shape[:2]
    print(f"Frame dimensions: {width}x{height}")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    for frame_idx, frame_path in enumerate(frame_names):
        # Read the original frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"Warning: Could not read frame {frame_path}")
            continue

        # If we have masks for this frame
        if frame_idx in video_segments:
            # Create a blank overlay for this frame
            overlay = np.zeros_like(frame)
            
            # Process each object's mask
            for obj_id, mask in video_segments[frame_idx].items():
                try:
                    # Convert mask to boolean array
                    binary_mask = mask > 0
                    
                    # Ensure mask is the right shape
                    if binary_mask.shape[:2] != (height, width):
                        print(f"Reshaping mask from {binary_mask.shape} to {(height, width)}")
                        binary_mask = np.resize(binary_mask, (height, width))
                    
                    # Get color for this object (use default if not in color_map)
                    color = color_map.get(obj_id, [0, 165, 255])
                    
                    # Apply color to overlay where mask is True
                    overlay[binary_mask] = color
                    
                except Exception as e:
                    print(f"Error processing mask for frame {frame_idx}, object {obj_id}: {str(e)}")
                    print(f"Mask type: {type(mask)}")
                    print(f"Mask shape: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
                    continue

            # Blend the overlay with the original frame
            alpha = 0.5  # Transparency factor
            frame = cv2.addWeighted(frame, 1, overlay, alpha, 0)

        # Write the frame to video
        out.write(frame)

    # Release the video writer
    out.release()
    print(f"Video created successfully with {len(frame_names)} frames at {fps} fps")

# Helper function to examine data structure
def inspect_data(frame_names, video_segments):
    print("\nData Inspection:")
    print("=================")
    print(f"Frame names type: {type(frame_names)}")
    print(f"Video segments type: {type(video_segments)}")
    
    if len(frame_names) > 0:
        print(f"\nFirst frame path: {frame_names[0]}")
        
    if video_segments:
        print("\nFirst frame segment info:")
        first_idx = list(video_segments.keys())[0]
        first_segment = video_segments[first_idx]
        print(f"First segment index: {first_idx}")
        print(f"First segment type: {type(first_segment)}")
        
        if isinstance(first_segment, dict):
            for obj_id, mask in first_segment.items():
                print(f"\nObject {obj_id}:")
                print(f"Mask type: {type(mask)}")
                print(f"Mask shape: {mask.shape if hasattr(mask, 'shape') else 'No shape'}")
                if isinstance(mask, np.ndarray):
                    print(f"Mask dtype: {mask.dtype}")
                    print(f"Mask unique values: {np.unique(mask)}")

frame_paths = [os.path.join(video_dir, name) for name in frame_names]
create_masked_video(frame_paths, video_segments, 'cam2_masked.mp4')