from core.dataloader import CustomDataLoader
from torch.utils.data import Dataset
import os
from PIL import Image

class iNaturalist_12k(Dataset):
    """
    A custom PyTorch Dataset class to load images from a directory structure
    where subdirectories represent class labels.
    """
    def __new__(cls, root_dir, transform=None):
        assert os.path.isdir(root_dir), f"Directory '{root_dir}' does not exist."
        assert transform is None or callable(transform), "Transform must be a callable or None."
        return super(iNaturalist_12k, cls).__new__(cls)
    
    def __init__(self, root_dir, transform=None):
        """
        Initializes the dataset.

        Args:
            root_dir (str): The root directory of the dataset (e.g., 'root/train').
            transform (callable, optional): A function/transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        self._load_data()

    def _load_data(self):
        """
        Scans the directory structure to find image paths and assign labels.
        """
        # Get the list of all subdirectories, which are our class labels
        class_names = sorted(os.listdir(self.root_dir))
        
        # Create the class-to-index and index-to-class mappings
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name
            
            class_path = os.path.join(self.root_dir, class_name)
            
            # Skip non-directories
            if not os.path.isdir(class_path):
                continue

            # Iterate through all files in the class directory
            for filename in os.listdir(class_path):
                # We only want to add common image file types
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(class_path, filename)
                    self.image_paths.append(image_path)
                    self.labels.append(idx)

    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label by index.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            Tuple[Any, int]: A tuple containing the transformed image and its integer label.
        """
        # Get the image path and label from our pre-compiled lists
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL (Pillow)
        image = Image.open(img_path).convert('RGB')

        # Apply the specified transforms
        if self.transform:
            image = self.transform(image)

        return image, label