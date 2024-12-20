import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Import for handling images
import time

def show_next_page():
    root.destroy()  # Close the current window for now; next page logic will be added later.

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
