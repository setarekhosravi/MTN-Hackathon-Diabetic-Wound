import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk  # Import for handling images
import time

def show_next_page():
    # Destroy the current window and open the next page
    root.destroy()
    next_page()

def next_page():
    # Initialize the next page window
    next_window = tk.Tk()
    next_window.title("Choose Image and Model")
    next_window.geometry("600x400")
    next_window.configure(bg="#FCBC14")  # Set background color to #FCBC14

    def browse_image():
        # Open file dialog to choose an image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
        if file_path:
            image_label.config(text=f"Selected Image: {file_path}")

    def save_selection():
        # Save the selected image path and model
        selected_image = image_label.cget('text').split(': ', 1)[1]
        selected_model = model_var.get()
        if selected_image == "No Image Selected":
            print("No image selected. Please choose an image.")
        else:
            print(f"Image Path: {selected_image}")
            print(f"Selected Model: {selected_model}")
            next_window.destroy()

    # Add widgets to the next page
    instruction_label = tk.Label(next_window, text="Choose an image and a model", font=("Comic Sans MS", 14, "bold"), bg="#FCBC14", fg="black")
    instruction_label.pack(pady=10)

    browse_button = tk.Button(next_window, text="Browse Image", command=browse_image, font=("Comic Sans MS", 12), bg="black", fg="#FCBC14")
    browse_button.pack(pady=10)

    image_label = tk.Label(next_window, text="No Image Selected", font=("Comic Sans MS", 12), bg="#FCBC14", fg="black")
    image_label.pack(pady=10)

    # Add radio buttons for model selection
    model_var = tk.StringVar(value="u-net")  # Default value is "u-net"
    u_net_radio = tk.Radiobutton(next_window, text="U-Net", variable=model_var, value="u-net", font=("Comic Sans MS", 12), bg="#FCBC14", fg="black")
    u_net_radio.pack(pady=5)

    deep_skin_radio = tk.Radiobutton(next_window, text="DeepSkin", variable=model_var, value="deepskin", font=("Comic Sans MS", 12), bg="#FCBC14", fg="black")
    deep_skin_radio.pack(pady=5)

    save_button = tk.Button(next_window, text="Save Selection", command=save_selection, font=("Comic Sans MS", 12), bg="black", fg="#FCBC14")
    save_button.pack(pady=20)

    next_window.mainloop()

# Initialize the main window
root = tk.Tk()
root.title("Diabetic Wound Segmentation and Clustering")
root.geometry("600x400")  # Set the window size

# Load and set the background image
background_image = Image.open("irancell-ehsanmobile.blog.ir.jpg")  # Replace with your image path
background_photo = ImageTk.PhotoImage(background_image)

background_label = tk.Label(root, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Add content to the first page
frame = tk.Frame(root, bg="#FCBC14", padx=25, pady=10)  # Create a yellow frame
frame.place(relx=0.5, rely=0.15, anchor=tk.CENTER)  # Place frame at the top of the background image

# Add labels for the project name, sponsor, and welcome message
project_name_label = tk.Label(frame, text="Diabetic Wound Segmentation and Clustering", bg="#FCBC14", fg="black", font=("Comic Sans MS", 16, "bold"))
project_name_label.pack(pady=1)

welcome_label = tk.Label(frame, text="Welcome!", bg="#FCBC14", fg="black", font=("Comic Sans MS", 12, "bold"))
welcome_label.pack(pady=1)

# Schedule the transition to the next page after 5 seconds
root.after(5000, show_next_page)

# Run the application
root.mainloop()
