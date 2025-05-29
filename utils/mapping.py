import torch


class kitti360map:
    def __init__(self):
        # List of original class IDs in the dataset
        classes = [0, 6, 7, 8, 9, 10,
                   11, 12, 13, 17, 19,
                   20, 21, 22, 23, 26, 24, 25, 27, 28,
                   30, 32, 33, 34, 35, 36, 37, 38, 
                   39, 40, 41, 42, 44]
        self.num_classes = len(classes)
        max_class_id = max(classes)

        # Create a lookup table with default value -1
        self.mapping = torch.full((max_class_id + 1,), -1, dtype=torch.long)
        for new_id, orig_id in enumerate(classes):
            self.mapping[orig_id] = new_id

        # Assert no new class IDs are -1
        assert (self.mapping[classes] != -1).all(), "Some new class IDs are -1, indicating a mapping error."

    def map(self, masks: torch.Tensor):
        # Map masks using the lookup table
        mapped_masks = self.mapping[masks]
        breakpoint()
        # Assert no mapped masks are not one of the new class IDs
        assert (mapped_masks >= 0 and mapped_masks < self.num_classes).all(), "Some mapped masks are not valid class IDs."

        return mapped_masks

