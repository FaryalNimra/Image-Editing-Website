import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
import os
from tkinter import filedialog


class LandingPage:
    def __init__(self, master):
        self.master = master
        master.title("Welcome to Website")
        master.geometry("800x600")

        self.create_widgets()

    def create_widgets(self):
        # Create a frame for the navigation bar
        self.nav_frame = tk.Frame(self.master, bg="#1A1A1D", pady=10)
        self.nav_frame.pack(side="top", fill="x")

        # Navigation bar content
        self.nav_label = tk.Label(self.nav_frame, text="Filtering Website", font=("Arial", 20, "bold"), fg="#FF5733",
                                  bg="#1A1A1D")
        self.nav_label.pack(side="left", padx=20)

        # Create a frame for the welcome message and images
        self.content_frame = tk.Frame(self.master, bg="#1A1A1D")
        self.content_frame.pack(expand=True, fill="both")

        # Create a label for decorative line with increased thickness
        self.line_label = tk.Label(self.content_frame, bg="#FF5733", height=4)
        self.line_label.pack(fill="x")

        # Create a scrolled text widget to display HTML content
        self.html_display = scrolledtext.ScrolledText(self.content_frame, wrap=tk.WORD, width=80, height=5,
                                                      bg="#1A1A1D", fg="white", font=("Arial", 18))
        self.html_display.pack(expand=True, fill="both", padx=20, pady=(20, 0))

        # Welcome message
        welcome_message = """
        Welcome to Image Filtering Website
        Experience the thrill and excitement of the Best Editing Website of the year!"""

        # Load the HTML content into the scrolled text widget
        self.html_display.insert(tk.END, welcome_message)

        # Add more content
        additional_content_label = tk.Label(self.content_frame, text="Explore our latest features:", fg="#FF5733",
                                             bg="#1A1A1D", font=("Arial", 14, "bold"))
        additional_content_label.pack(pady=(20, 10))

        feature1_label = tk.Label(self.content_frame, text="1. Advanced filters", fg="white", bg="#1A1A1D",
                                   font=("Arial", 12))
        feature1_label.pack()

        feature2_label = tk.Label(self.content_frame, text="2. Your Choice Matters!", fg="white", bg="#1A1A1D",
                                   font=("Arial", 12))
        feature2_label.pack()

        # Add images
        self.image_frame = tk.Frame(self.content_frame, bg="#1A1A1D")
        self.image_frame.pack(expand=True, fill="both", pady=(20, 0))

        # Previous images
        pumpkin_path = "C:/Users/PMLS/Pictures/H5.PNG"
        ghost_path = "C:/Users/PMLS/Pictures/H.PNG"

        self.add_image(pumpkin_path)
        self.add_image(ghost_path)

        # New images
        self.add_image("C:/Users/PMLS/Pictures/H1.PNG")
        self.add_image("C:/Users/PMLS/Pictures/H2.PNG")
        self.add_image("C:/Users/PMLS/Pictures/H3.PNG")
        self.add_image("C:/Users/PMLS/Pictures/H4.PNG")
        self.add_image("C:/Users/PMLS/Pictures/A.PNG")

        # Center align the welcome message
        self.html_display.tag_configure("center", justify="center")
        self.html_display.tag_add("center", "1.0", "end")

        # Add a button below the welcome message
        try_now_button = tk.Button(self.content_frame, text="Try Now", font=("Arial", 12, "bold"), fg="white",
                                   bg="#FF5733", bd=0, command=self.next_stage)
        try_now_button.pack(pady=20)

    def add_image(self, path):
        image = Image.open(path)
        image = image.resize((150, 150))
        image = ImageTk.PhotoImage(image)
        image_label = tk.Label(self.image_frame, image=image, bg="#1A1A1D")
        image_label.image = image
        image_label.pack(side="left", padx=20)

    def next_stage(self):
        # Execute the main.py file
        file_path = "C:/Users/PMLS/PycharmProjects/pythonProject1/.venv/main.py"
        os.system("python {}".format(file_path))


def main():
    root = tk.Tk()
    app = LandingPage(root)
    root.mainloop()


if __name__ == "__main__":
    main()
