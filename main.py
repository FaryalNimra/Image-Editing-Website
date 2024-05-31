import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tkinter import filedialog

# Initialize global variables
img = None
original_img = None


def convert_to_cartoon(image):
    # Convert image to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply median blur to smoothen the image
    img_blur = cv2.medianBlur(img_gray, 7)

    # Retrieve the edges for cartoon effect
    edges = cv2.Laplacian(img_blur, cv2.CV_8U, ksize=5)
    _, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY_INV)

    # Apply the cartoon effect by masking the original image
    img_cartoon = cv2.bitwise_and(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), image)

    return img_cartoon


def open_home():
    print("Home button clicked")


def open_about():
    print("About button clicked")


def open_filters():
    print("Filters button clicked")


def open_contact():
    print("Contact button clicked")


def open_image():
    global image_label, img, original_img
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
    print("Selected Image:", file_path)
    if file_path:
        original_img = cv2.imread(file_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img = original_img.copy()
        image = Image.fromarray(img)
        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.image = photo


def apply_filter():
    global image_label, img, original_img
    selected_filter = filter_var.get()

    if selected_filter == "Cartoon filter":
        if original_img is not None:
            img = convert_to_cartoon(original_img)
    elif selected_filter == "Brightness":
        brightness_value = 50
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        v = np.clip(v + brightness_value, 0, 255)
        hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    elif selected_filter == "Edge Detection":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 100, 200)
        img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        img = img_edges
    elif selected_filter == "Sharpness":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_sharpened = cv2.filter2D(img, -1, kernel)
        img = img_sharpened
    elif selected_filter == "Laplacian":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        # Convert the Laplacian output to an image format with appropriate values for visualization
        img_laplacian = np.uint8(np.absolute(laplacian))
        # Stack the Laplacian output with three identical channels to convert it to RGB format
        img_laplacian = cv2.merge((img_laplacian, img_laplacian, img_laplacian))
        img = img_laplacian
    elif selected_filter == "Canny edge detection":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img_gray, 50, 150)
        img_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        img = img_edges
    elif selected_filter == "Harris edge detection":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply Harris corner detection
        dst = cv2.cornerHarris(img_gray, 2, 3, 0.04)
        # Threshold to obtain the corners
        dst = cv2.dilate(dst, None)
        img[dst > 0.01 * dst.max()] = [0, 0, 255]  # Mark the detected corners in red
    elif selected_filter == "Emboss":
        kernel_emboss_1 = np.array([[0, -1, -1],
                                    [1, 0, -1],
                                    [1, 1, 0]])
        img_emboss = cv2.filter2D(img, -1, kernel_emboss_1)
        img = img_emboss
    elif selected_filter == "Sepia":
        kernel_sepia = np.array([[0.393, 0.769, 0.189],
                                 [0.349, 0.686, 0.168],
                                 [0.272, 0.534, 0.131]])
        img_sepia = cv2.transform(img, kernel_sepia)
        img = img_sepia
    elif selected_filter == "Gaussian filter":
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif selected_filter == "Median filter":
        img = cv2.medianBlur(img, 5)
    elif selected_filter == "Mean filter":
        img = cv2.blur(img, (5, 5))
    elif selected_filter == "Bilateral filter":
        img = cv2.bilateralFilter(img, 9, 75, 75)
    elif selected_filter == "Sobel filter":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=5)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=5)
        sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel = np.clip(sobel, 0, 255)
        img = np.uint8(sobel)
    elif selected_filter == "Prewitt filter":
        kernel_prewitt_x = np.array([[1, 1, 1],
                                     [0, 0, 0],
                                     [-1, -1, -1]])
        kernel_prewitt_y = np.array([[-1, 0, 1],
                                     [-1, 0, 1],
                                     [-1, 0, 1]])
        img_prewitt_x = cv2.filter2D(img, -1, kernel_prewitt_x)
        img_prewitt_y = cv2.filter2D(img, -1, kernel_prewitt_y)
        img = img_prewitt_x + img_prewitt_y
    elif selected_filter == "High-pass filter":
        kernel_highpass = np.array([[-1, -1, -1],
                                    [-1, 9, -1],
                                    [-1, -1, -1]])
        img_highpass = cv2.filter2D(img, -1, kernel_highpass)
        img = img_highpass
    elif selected_filter == "Roberts Cross edge detector":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        roberts_x = np.array([[1, 0], [0, -1]])
        roberts_y = np.array([[0, 1], [-1, 0]])
        roberts_x_edges = cv2.filter2D(img_gray, -1, roberts_x)
        roberts_y_edges = cv2.filter2D(img_gray, -1, roberts_y)
        img_edges = np.sqrt(roberts_x_edges ** 2 + roberts_y_edges ** 2)
        img_edges = np.clip(img_edges, 0, 255)
        img_edges = np.uint8(img_edges)
        img = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
    elif selected_filter == "Gaussian noise filter":
        noise = np.random.normal(0, 50, img.shape)  # Adjust the standard deviation (sigma) as needed
        noisy_img = img + noise
        img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    elif selected_filter == "Salt-and-pepper noise filter":
        noise = np.random.randint(0, 2, img.shape[:2])
        salt = noise == 1
        pepper = noise == 0
        img[salt] = 255
        img[pepper] = 0
    elif selected_filter == "Wiener filter":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.medianBlur(img_gray, 5)
        img = cv2.warpAffine(img, np.float32([[1, 0, 0], [0, 1, 0]]), (0, 0), img, cv2.INTER_LINEAR)
    elif selected_filter == "Adaptive filter":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.bilateralFilter(img_gray, 9, 75, 75)
    elif selected_filter == "Fourier Transform filters":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift))
        img = magnitude_spectrum
    elif selected_filter == "Butterworth filter":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        D = 30  # cutoff radius
        n = 2  # order of the filter
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                mask[i, j] = 1 / (1 + (distance / D) ** (2 * n))
        f_shift_butterworth = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_butterworth)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img = np.uint8(img_back)
    elif selected_filter == "Chebyshev filter":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        D = 30  # cutoff radius
        n = 2  # order of the filter
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                mask[i, j] = 1 / (1 + ((distance * D) / ((distance ** 2) - D ** 2)) ** (2 * n))
        f_shift_chebyshev = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_chebyshev)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img = np.uint8(img_back)
    elif selected_filter == "Band-pass filter":
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        f_transform = np.fft.fft2(img_gray)
        f_shift = np.fft.fftshift(f_transform)
        rows, cols = img_gray.shape
        crow, ccol = rows // 2, cols // 2
        D_low = 20  # lower cutoff radius
        D_high = 60  # higher cutoff radius
        n = 2  # order of the filter
        mask = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                distance = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
                mask[i, j] = 1 / (1 + ((D_low * D_high) / ((distance ** 2) - D_low * D_high)) ** (2 * n))
        f_shift_bandpass = f_shift * mask
        f_ishift = np.fft.ifftshift(f_shift_bandpass)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        img = np.uint8(img_back)
    elif selected_filter == "Erosion filter":
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
    elif selected_filter == "Dilation filter":
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
    elif selected_filter == "Opening filter":
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif selected_filter == "Closing filter":
        kernel = np.ones((5, 5), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        print("No filter selected")

    # Display the modified image
    photo = ImageTk.PhotoImage(Image.fromarray(img))
    image_label.config(image=photo)
    image_label.image = photo


def undo_filter():
    global image_label, img, original_img
    if original_img is not None:
        img = original_img.copy()
        photo = ImageTk.PhotoImage(Image.fromarray(img))
        image_label.config(image=photo)
        image_label.image = photo


root = tk.Tk()
root.title("Main Page")
root.configure(bg="black")  # Set background color to black

# Header
header_frame = tk.Frame(root, bg="#1A1A1D", pady=20)
header_frame.pack(fill=tk.X)

logo_label = tk.Label(header_frame, text="Filters", font=("Arial", 16, "bold"), fg="#FF5733", bg="#1A1A1D")
logo_label.pack(side=tk.LEFT, padx=(20, 0))

nav_frame = tk.Frame(header_frame, bg="#080808")
nav_frame.pack(side=tk.RIGHT, padx=(0, 20))

nav_buttons = [
    ("Smoothing", open_home),
    ("Sharpening", open_about),
    ("Detection", open_filters),
    ("Reduction", open_contact)
]

for text, command in nav_buttons:
    button = tk.Button(nav_frame, text=text, font=("Arial", 16, "bold"), fg="#FF5733", bg="#080808", bd=0,
                       activebackground="#FF5733", activeforeground="white", command=command)  # Changed button colors
    button.pack(side=tk.LEFT, padx=10)

# Pink line below navbar
pink_line = tk.Canvas(root, bg="#FF5733", height=5, width=root.winfo_width())
pink_line.create_line(0, 0, root.winfo_width(), 0, fill="#FF5733", width=10)  # Changed line color
pink_line.pack(fill=tk.X)

# Instructions Frame
instructions_frame = tk.Frame(root, bg="#1A1A1D", pady=20, width=250)  # Changed background color to match header
instructions_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20), pady=20, anchor="e")

instructions_text = """Smoothing Filters
1.Gaussian       2.Median 
3.Mean           4.Bilateral 

Sharpening Filters
1.Laplacian         2.Sobel 
3.Prewitt           4.High-pass 

Edge Detection Filters
1.Canny             2.Sobel 
3.Prewitt           4.Roberts 

Noise Reduction Filters
1.Gaussian   2.Salt-and-pepper 
3.Wiener     4.Adaptive 

Frequency Domain Filters
1.Fourier    2.Butterworth 
3.Chebyshev  4.Band-pass 
 """
# Create a frame to contain the heading, line, and instructions label
heading_frame = tk.Frame(instructions_frame, bg="#1A1A1D")
heading_frame.pack(padx=40, pady=(5, 20), anchor="w")

# Add the heading label
heading_label = tk.Label(heading_frame, text="Apply What You Like", font=("Arial", 15, "bold"), fg="#FF5733",
                         bg="#1A1A1D")
heading_label.pack(side="top")

# Add a canvas for the line below the heading
line_canvas = tk.Canvas(heading_frame, bg="#FF5733", height=1, width=250)  # Adjust width as needed
line_canvas.pack(side="top", fill="x")

# Add the instructions label
instructions_label = tk.Label(instructions_frame, text=instructions_text, font=("Arial", 12), fg="#FF5733",
                              bg="#1A1A1D",
                              wraplength=250, justify=tk.LEFT)
instructions_label.pack(padx=40, pady=(5, 20))

# Add the button
open_image_button = tk.Button(instructions_frame, text="Open Image", font=("Arial", 16, "bold"), fg="#FF5733",
                              bg="#1A1A1D", bd=0, activebackground="#FF5733", activeforeground="white",
                              command=open_image)
open_image_button.pack(pady=10)

# Filter Selection Frame
filter_frame = tk.Frame(root, bg="#1A1A1D", pady=20, width=250)  # Changed background color to match header
filter_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20, anchor="e")

# Drop-down menu for filters
filter_var = tk.StringVar(root)
filter_var.set("Select Filter")

filters = ["Gaussian filter", "Mean filter", "Median filter", "Bilateral filter", "Laplacian",
           "Prewitt filter", "Sobel filter", "High-pass filter", "Canny edge detection",
           "Roberts Cross edge detector", "Gaussian noise filter", "Salt-and-pepper noise filter", "Wiener filter",
           "Adaptive filter", "Fourier Transform filters", "Chebyshev filter", "Band-pass filter", "Butterworth filter",
           "Erosion filter", "Opening filter", "Closing filter", "Dilation filter",
           "Cartoon filter", "Brightness", "Harris edge detection", "Edge Detection", "Sharpness", "Emboss", "Sepia",
           ]

filter_menu = tk.OptionMenu(filter_frame, filter_var, *filters)
filter_menu.config(font=("Arial", 16), bg="#1A1A1D", fg="white", width=15)  # Changed bg color and text color
filter_menu["menu"].config(bg="#1A1A1D", fg="white")  # Changed menu bg color and text color
filter_menu.pack()
filter_menu.configure(fg="#FF5733")

# Button to apply selected filter
apply_button = tk.Button(filter_frame, text="Apply Filter", font=("Arial", 16, "bold"), fg="#FF5733", bg="#1A1A1D",
                         bd=0,
                         activebackground="#FF5733", activeforeground="#1A1A1D",
                         command=apply_filter)  # Changed button colors
apply_button.pack(pady=10)

# Button to undo filter
undo_button = tk.Button(filter_frame, text="Undo", font=("Arial", 16, "bold"), fg="#FF5733", bg="#1A1A1D", bd=0,
                        activebackground="#FF5733", activeforeground="#1A1A1D",
                        command=undo_filter)  # Changed button colors
undo_button.pack(pady=10)


# Button to save the filtered image
def save_image():
    global img
    file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"),
                                                        ("All files", "*.*")])
    if file_path:
        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


save_button = tk.Button(filter_frame, text="Save", font=("Arial", 16, "bold"), fg="#FF5733", bg="#1A1A1D", bd=0,
                        activebackground="#FF5733", activeforeground="#1A1A1D",
                        command=save_image)  # Changed button colors
save_button.pack(pady=10)

# Create a frame to hold the image label
image_frame = tk.Frame(root, bg="#1A1A1D")
image_frame.pack(expand=True, fill=tk.BOTH, padx=20)

# Create a label to display the image
image_label = tk.Label(image_frame)
image_label.pack()

root.mainloop()
