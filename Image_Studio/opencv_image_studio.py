import cv2
import numpy as np

# 1. CARTOONIFY IMAGE

def cartoonify():

    path = input("Enter image path: ")

    image = cv2.imread(path)

    if image is None:
        print("Could not load image.")
        return

    # Resize image for faster processing
    image = cv2.resize(image, (800, 600))

    # 1. Smooth the colors
    color = image.copy()

    for i in range(2):
        color = cv2.bilateralFilter(
            color,
            9,
            75,
            75
        )

    # 2. Convert to grayscale

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth grayscale image
    gray = cv2.medianBlur(gray, 7)

    # 3. Detect cartoon outlines

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    # 4. Reduce colors

    data = np.float32(color)

    # Number of colors
    K = 8

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0
    )

    _, labels, centers = cv2.kmeans(
        data.reshape((-1, 3)),
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)

    quantized = centers[labels.flatten()]

    quantized = quantized.reshape(color.shape)

    # 5. Combine colors + outlines

    cartoon = cv2.bitwise_and(
        quantized,
        quantized,
        mask=edges
    )

    # 6. Display result

    cv2.imshow("Original", image)
    cv2.imshow("Cartoon", cartoon)

    cv2.imwrite("cartoon.png", cartoon)

    print("Cartoon image saved as cartoon.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 2. CREATE IMAGE COLLAGE

def create_collage():

    images = []

    for i in range(1, 5):
        path = input(f"Enter path for image {i}: ")

        image = cv2.imread(path)

        if image is None:
            print(f"Could not load image {i}.")
            return

        # Resize every image to same size
        image = cv2.resize(image, (400, 300))

        images.append(image)

    # Arrange images
    top = np.hstack((images[0], images[1]))
    bottom = np.hstack((images[2], images[3]))

    collage = np.vstack((top, bottom))

    cv2.imshow("Image Collage", collage)
    cv2.imwrite("collage.png", collage)

    print("Collage saved as collage.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()



# MAIN MENU

while True:

    print("\nOpenCV Image Studio")

    print("1. Cartoonify Image")
    print("2. Create Image Collage")
    print("3. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":
        cartoonify()

    elif choice == "2":
        create_collage()

    elif choice == "3":
        print("Exiting Image Studio...")
        break

    else:
        print("Invalid choice. Please try again.")