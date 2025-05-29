import os
import torch


def load_checkpoint_metadata(checkpoint_path):
    """
    Load metadata (model name and comment) from a checkpoint file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.

    Returns:
        dict: A dictionary containing the model name and comment, if available.
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        metadata = {
            "model_name": checkpoint.get("model_name", None),
            "comment": checkpoint.get("comment", None),
        }
        return metadata
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None


def inspect_checkpoints(checkpoints_dir):
    """
    Inspects all model checkpoints and prints their names and comments.

    Args:
        checkpoints_dir (str): Path to the directory containing model checkpoints.
    """
    # Inspect checkpoints
    print("Inspecting model checkpoints...")
    for checkpoint_file in os.listdir(checkpoints_dir):
        if checkpoint_file.endswith('.pth'):
            checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
            print(f"Checkpoint: {checkpoint_file}")

            metadata = load_checkpoint_metadata(checkpoint_path)
            if metadata:
                model_name = metadata.get("model_name")
                comment = metadata.get("comment")

                if model_name:
                    print(f"  Model Name: {model_name}")
                else:
                    print(f"  Error: No model name found in checkpoint '{checkpoint_file}'.")

                if comment:
                    print(f"  Comment: {comment}")
                else:
                    print(f"  Warning: No comment provided for model '{model_name}'.")
            else:
                print(f"  Error: Failed to load metadata from checkpoint '{checkpoint_file}'.")

# Example usage
if __name__ == "__main__":
    checkpoints_dir = "/home/panos/dev/hf_seg/checkpoints"
    inspect_checkpoints(checkpoints_dir)