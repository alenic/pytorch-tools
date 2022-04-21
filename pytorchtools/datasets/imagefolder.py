import torchvision.datasets as datasets
import os
import json
import numpy as np


class ImageFolderDataset(datasets.vision.VisionDataset):
    """A generic data loader.

    Images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
    root (string): Root directory path.

    class_map_dict (dict, optional): A dict file containing the mappings of class-id
        and class-name. The dict must have the form: {"0": ["person", "woman", "man"],
        "1":["car"], ...}.

    class_map_file (string, optional): A JSON file containing the mappings of
        class-id and class-name. This parameter is ignored if class_map_dict is
        setted. The JSON file must be formed as follows:

        class_map.json
        --------------------------
        {
            "0": ["person", "woman", "man"],
            "1": ["dog", "chihuahua"],
            "2": ["car"],
            ...
        }
        --------------------------
        where the key is the class id (unique) and the value is a list of class
        name (folder).

    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``.

    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.

    loader (callable, optional): A function to load an image given its path.

    is_valid_file (callable, optional): A function that takes path of an Image file
        and check if the file is a valid file (used to check of corrupt files).
    """

    def __init__(
        self,
        root,
        class_map_dict=None,
        class_map_file=None,
        transform=None,
        target_transform=None,
        loader=datasets.folder.default_loader,
        is_valid_file=None,
        max_samples_per_class=None,
        read_paths=False,
        index=None
    ):

        super(ImageFolderDataset, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        self.class_map_dict = class_map_dict
        self.class_map_file = class_map_file
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.max_samples_per_class = max_samples_per_class
        self.read_paths = read_paths

        if class_map_dict is None:
            if class_map_file is not None:
                # Check the existence of the class_map_file
                class_map_ext = class_map_file.split(".")[-1].lower()
                if class_map_ext != "json":
                    assert False, "class_map_file must have .json extension"

                # Check the extension of the class_map_file
                assert os.path.exists(class_map_file), (
                    "Cannot locate class_map_file (%s)" % class_map_file
                )

                # Check the json validity of the class_map_file

                with open(class_map_file, "r") as f:
                    try:
                        class_map_dict = json.load(f)
                    except Exception as e:
                        raise type(e)("class_map_file is an invalid json")
        else:
            assert isinstance(
                class_map_dict, dict
            ), "class_map_dict must be a python dictionary"

        self.extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".ppm",
            ".bmp",
            ".pgm",
            ".tif",
            ".tiff",
            ".webp",
        )

        self.class_map = class_map_dict

        classes, class_to_idx = self._find_classes(self.root)

        if self.class_map is None:
            self.class_map = {k: [c] for k, c in enumerate(classes)}

        self.classes = classes
        self.class_to_idx = class_to_idx

        samples = datasets.folder.make_dataset(
            self.root, class_to_idx, self.extensions, is_valid_file
        )
        if len(samples) == 0:
            raise (
                RuntimeError(
                    "Found 0 files in subfolders of: " + self.root + "\n"
                    "Supported extensions are: " + ",".join(self.extensions)
                )
            )
        if max_samples_per_class is not None:
            target_counter = {}
            limited_samples = []
            for path, target in samples:
                if target not in target_counter.keys():
                    target_counter[target] = 0
                else:
                    if target_counter[target] < max_samples_per_class:
                        limited_samples.append((path, target))
                        target_counter[target] += 1

        if max_samples_per_class is not None:
            self.samples = limited_samples
        else:
            self.samples = samples
        
        self.index = index

    def _find_classes(self, dir):
        """Finds the class folders in a dataset.

        Ensures that no class is a subdirectory of another.

        Args:
        dir (string): Root directory path.

        Returns
        tuple: (classes, class_to_idx) where classes are relative to (dir), and
            class_to_idx is a dictionary.
        """

        if self.class_map is not None:
            # Find all the class names
            class_map_name_list = []
            for class_id in self.class_map:
                class_map_name_list += self.class_map[class_id]

            # Check if a mapped class has any duplicates
            if len(class_map_name_list) != len(set(class_map_name_list)):
                class_duplicates = set(
                    [x for x in class_map_name_list if class_map_name_list.count(x) > 1]
                )

                assert False, "class_map_file has duplicate classes: %s" % (
                    ",".join(class_duplicates)
                )

            num_classes = len(self.class_map)

            # Check if the map ids are contiguos
            class_map_key = [int(x) for x in self.class_map.keys()]
            max_key = max(class_map_key)
            min_key = min(class_map_key)
            n_elem = max_key - min_key + 1
            if n_elem != num_classes:
                assert (
                    False
                ), "class_map_file must have all number in the range [0-%d]" % (
                    num_classes - 1
                )

            folder_classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            # Check if the mapped classes are in the folder classes
            class_to_remove = []
            class_map_name_set = set(class_map_name_list)
            for c in class_map_name_set:
                if c not in folder_classes:
                    print("Warning: '%s' is not a directory of the root" % (c))
                    class_to_remove.append(c)

            # Build class and class_to_idx
            classes = [c for c in class_map_name_set if c not in class_to_remove]
            classes.sort()
            class_to_idx = {}
            for class_id in self.class_map.keys():
                for class_name in self.class_map[class_id]:
                    if class_name not in class_to_remove:
                        class_to_idx[class_name] = int(class_id)
        else:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes.sort()
            class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def force_samples(self, samples):
        """Force samples

        Args:
            samples (list): List of tuples: (image_path, target)
        """
        self.samples = samples

    def get_stats(self):
        labels = np.array([s[1] for s in self.samples])
        unique, counts = np.unique(labels, return_counts=True)
        num_samples = len(self.samples)
        s = ""
        for k in range(len(unique)):
            s += "%s : %d -> %.2f %%\n" % (
                self.class_map[unique[k]],
                counts[k],
                (100 * counts[k] / num_samples),
            )

        return s

    def __getitem__(self, index):
        """
        Args:
        index (int): Index.

        Returns:
        tuple: (sample, target) where target is class_index of the target class.
        """
        if self.index is not None:
            index = self.index[index]
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            if isinstance(self.transform, list):
                sample = self.transform[target](sample)
            else:
                sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.read_paths:
            return sample, target, path

        return sample, target

    def __len__(self):
        if self.index is not None:
            return len(self.index)
        return len(self.samples)
