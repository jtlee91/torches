import torch
import os
import pandas as pd
from PIL import Image


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file_path, image_dir_path, label_dir_path, S=7, B=2, C=20, transform=None):
        self.annotations = pd.read_csv(csv_file_path, header=None)
        self.image_dir_path = image_dir_path
        self.label_dir_path = label_dir_path
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir_path, self.annotations.iloc[index, 1])
        boxes = list()
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [int(x) if float(x) == int(float(x)) else float(x) for x in label.replace("\n", "").split()]
                boxes.append([class_label, x, y, width, height])

        boxes = torch.tensor(boxes)  # (cx, cy, width, height)
        image_path = os.path.join(self.image_dir_path, self.annotations.iloc[index, 0])
        image = Image.open(image_path)

        if self.transform:
            image, boxes = self.transform(image), boxes

        label_matrix = torch.zeros(self.S, self.S, self.C + 5)
        # print(boxes)
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)

            center_x_grid = int(self.S * x)
            center_y_grid = int(self.S * y)
            obj_x = self.S * x - center_x_grid
            obj_y = self.S * y - center_y_grid
            obj_width = width * self.S
            obj_height = height * self.S

            if label_matrix[center_x_grid, center_y_grid, 24] == 0:
                label_matrix[center_x_grid, center_y_grid, 24] = 1
                box_coordinates = torch.tensor([obj_x, obj_y, obj_width, obj_height])
                # print(box_coordinates)
                label_matrix[center_x_grid, center_y_grid, 20:24] = box_coordinates
                label_matrix[center_x_grid, center_y_grid, class_label] = 1
                # print(label_matrix[center_x_grid, center_y_grid, ...])
        return image, label_matrix


if __name__ == "__main__":
    csv_file_path = "D:/Data/PascalVOC_YOLO/test.csv"
    image_dir_path = "D:/Data/PascalVOC_YOLO/images"
    label_dir_path = "D:/Data/PascalVOC_YOLO/labels"
    res = VOCDataset(csv_file_path, image_dir_path, label_dir_path)
    res[2]
