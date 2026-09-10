import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

base_size_x = 8
base_size_y = 32
base_size = base_size_x * base_size_y

boxes = np.empty((0, 4), dtype=int)
pelak_name = "pelak.jpg"


def loadflattenimage(imageadd):
    img = cv2.imread(imageadd, 0)
    img = cv2.resize(img, (base_size_x, base_size_y))
    return img.flatten()


ds_images = np.empty((0, base_size), dtype=int)
ds_nos = np.empty((0), dtype=int)

for dirno in range(1, 10):
    files = os.listdir("pelak/" + str(dirno))

    for filename in files:
        im = loadflattenimage("pelak/" + str(dirno) + "/" + filename)
        ds_images = np.append(ds_images, [im], axis=0)
        ds_nos = np.append(ds_nos, dirno)

x_train, x_test, y_train, y_test = train_test_split(
    ds_images, ds_nos, test_size=0.2
)

model = linear_model.LogisticRegression(max_iter=100000)
model.fit(x_train, y_train)

out = model.predict(x_test)
output = [x == y for x, y in zip(out, y_test)]
accuracy = np.mean(output)

print("Model Accuracy:", accuracy * 100, "%")

pelak_base = cv2.imread(pelak_name)

if pelak_base is None:
    raise FileNotFoundError(
        f"Could not find '{pelak_name}'. Put the plate image next to this Python file."
    )

pelak_base = cv2.resize(pelak_base, (410, 90))
pelak = pelak_base.copy()
pelak = cv2.cvtColor(pelak, cv2.COLOR_BGR2GRAY)
pelak = cv2.blur(pelak, (3, 3))

_, pelak = cv2.threshold(pelak, 64, 255, cv2.THRESH_BINARY)
pelak = cv2.blur(pelak, (3, 3))

plot = 90 - np.sum(pelak, axis=0, keepdims=True) / 255

line_threshold = 5
min_width = 15
max_width = 50

start_index = 0
end_index = 0
prev_value = 0
final_number = ""

for index, plot_value in enumerate(plot[0]):

    if plot_value >= line_threshold and prev_value < line_threshold:
        start_index = index

    if plot_value < line_threshold and prev_value >= line_threshold:
        end_index = index
        width = end_index - start_index

        if width > max_width:
            final_number += " "

        elif min_width < width < max_width:

            cv2.rectangle(
                pelak_base,
                (start_index, 0),
                (end_index, 89),
                (255, 255, 255),
                2
            )

            new_pic = pelak[:, start_index:end_index]

            new_picf = cv2.resize(
                new_pic, (base_size_x, base_size_y)
            ).flatten()

            out = model.predict([new_picf])

            print(out[0], end=" ")
            final_number += str(out[0])

    prev_value = plot_value

zirepelak = pelak_base[0:50].copy()
np.ndarray.fill(zirepelak, 255)

pelak_base = np.append(pelak_base, zirepelak, axis=0)

cv2.putText(
    pelak_base,
    "Plate : " + final_number,
    (10, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.75,
    (255, 0, 255),
    1,
    cv2.LINE_AA
)

print()
print("Detected Plate:", final_number)

cv2.imshow("Detected License Plate", pelak_base)
cv2.waitKey(0)

cv2.imshow("Binary Plate", pelak)
cv2.waitKey(0)

cv2.destroyAllWindows()
