import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from main import (show_wound_parts, analyze_colors, plot_color_histogram, segment_wound_parts,
                  display_segmented_parts, predict_cluster_for_wound)
from utils.crop_images import find_contour, crop
from utils.preprocess import preprocess_image
import openai


class WoundAnalysisApp:
    def __init__(self):
        self.current_window = None
        self.selected_image = None
        self.selected_model = None
        self.preprocessed_image = None
        self.color_percentages = {}
        openai.api_key = "API_KEY"
        self.show_welcome_page()

    def show_welcome_page(self):
        if self.current_window:
            self.current_window.destroy()
        
        self.current_window = tk.Tk()
        self.current_window.title("Diabetic Wound Segmentation and Clustering")
        self.current_window.geometry("600x400")

        # Load and set background image
        background_image = Image.open("Hackathon Official Data/irancell-ehsanmobile.blog.ir.jpg")
        background_photo = ImageTk.PhotoImage(background_image)
        
        background_label = tk.Label(self.current_window, image=background_photo)
        background_label.image = background_photo
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        frame = tk.Frame(self.current_window, bg="#FCBC14", padx=25, pady=10)
        frame.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

        project_name_label = tk.Label(frame, text="Diabetic Wound Segmentation and Clustering",
                                      bg="#FCBC14", fg="black", font=('courier 10 pitch', 16, "bold"))
        project_name_label.pack(pady=1)

        welcome_label = tk.Label(frame, text="Welcome!", 
                                 bg="#FCBC14", fg="black", font=("courier 10 pitch", 12, "bold"))
        welcome_label.pack(pady=1)

        self.current_window.after(5000, self.show_selection_page)
        self.current_window.mainloop()

    def show_selection_page(self):
        if self.current_window:
            self.current_window.destroy()
        
        self.current_window = tk.Tk()
        self.current_window.title("Choose Image and Model")
        self.current_window.geometry("600x400")
        self.current_window.configure(bg="#FCBC14")

        instruction_label = tk.Label(self.current_window, text="Choose an image and a model",
                                     font=("courier 10 pitch", 14, "bold"), bg="#FCBC14", fg="black")
        instruction_label.pack(pady=10)

        browse_button = tk.Button(self.current_window, text="Browse Image",
                                  command=self.browse_image, font=("courier 10 pitch", 12),
                                  bg="black", fg="#FCBC14")
        browse_button.pack(pady=10)

        self.image_label = tk.Label(self.current_window, text="No Image Selected",
                                    font=("courier 10 pitch", 12), bg="#FCBC14", fg="black")
        self.image_label.pack(pady=10)

        self.model_var = tk.StringVar(value="u-net")
        u_net_radio = tk.Radiobutton(self.current_window, text="U-Net",
                                     variable=self.model_var, value="u-net",
                                     font=("courier 10 pitch", 12), bg="#FCBC14", fg="black")
        u_net_radio.pack(pady=5)

        deep_skin_radio = tk.Radiobutton(self.current_window, text="DeepSkin",
                                         variable=self.model_var, value="deepskin",
                                         font=("courier 10 pitch", 12), bg="#FCBC14", fg="black")
        deep_skin_radio.pack(pady=5)

        next_button = tk.Button(self.current_window, text="Next",
                                command=self.show_preprocessing_page,
                                font=("courier 10 pitch", 12), bg="black", fg="#FCBC14")
        next_button.pack(pady=20)

    def browse_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif *.bmp *.ico")])
        if file_path:
            self.selected_image = file_path
            self.image_label.config(text=f"Selected Image: {file_path}")

    def show_preprocessing_page(self):
        if not self.selected_image:
            messagebox.showerror("Error", "Please select an image first")
            return
            
        self.selected_model = self.model_var.get()
        if self.current_window:
            self.current_window.destroy()

        # Load and preprocess image
        original_img = cv2.imread(self.selected_image)
        bounding_box = find_contour(original_img)
        self.preprocessed_image = crop(image=original_img, bbox=bounding_box)
        
        self.current_window = tk.Tk()
        self.current_window.title("Preprocessed Image")
        self.current_window.geometry("800x600")
        self.current_window.configure(bg="#FCBC14")

        # Display preprocessed image
        img_display = cv2.cvtColor(self.preprocessed_image, cv2.COLOR_BGR2RGB)
        img_display = Image.fromarray(img_display)
        img_display = ImageTk.PhotoImage(img_display.resize((400, 400)))

        img_label = tk.Label(self.current_window, image=img_display)
        img_label.image = img_display
        img_label.pack(pady=20)

        next_button = tk.Button(self.current_window, text="Next",
                                command=self.show_clustering_page, font=("courier 10 pitch", 12),
                                bg="black", fg="#FCBC14")
        next_button.pack(pady=20)

    def show_clustering_page(self):
        if self.current_window:
            self.current_window.destroy()

        self.current_window = tk.Tk()
        self.current_window.title("Clustering Results")
        self.current_window.geometry("800x600")
        self.current_window.configure(bg="#FCBC14")

        # Predict cluster
        img_for_clustering = preprocess_image(self.preprocessed_image)
        cluster_id = predict_cluster_for_wound(img_for_clustering,
                                               kmeans_model_path="Hackathon Official Data/Results/kmeans_model.joblib",
                                               umap_model_path="Hackathon Official Data/Results/umap_model.joblib")
        tk.Label(self.current_window, text=f"Cluster ID: {cluster_id}", font=("courier 10 pitch", 14), bg="#FCBC14").pack(pady=20)

        next_button = tk.Button(self.current_window, text="Next",
                                command=self.show_wound_parts_page, font=("courier 10 pitch", 12),
                                bg="black", fg="#FCBC14")
        next_button.pack(pady=20)

    def show_wound_parts_page(self):
        if self.current_window:
            self.current_window.destroy()

        self.current_window = tk.Tk()
        self.current_window.title("Wound Parts")
        self.current_window.geometry("800x600")
        self.current_window.configure(bg="#FCBC14")

        show_wound_parts(self.selected_model, self.preprocessed_image)

        self.color_percentages = analyze_colors(self.preprocessed_image)
        plot_color_histogram(self.color_percentages)

        # Segment and display parts
        parts = segment_wound_parts(self.preprocessed_image)
        display_segmented_parts(parts)

        next_button = tk.Button(self.current_window, text="Next",
                              command=self.show_results_page,
                              font=("courier 10 pitch", 12),
                              bg="black", fg="#FCBC14")
        next_button.pack(pady=20)

    def generate_llm_report(self):
        """
        Generate a detailed wound analysis report using OpenAI GPT-4.
        """
        # Format the color percentages into a readable string
        color_info = "\n".join([f"{color}: {percentage:.2f}%" for color, percentage in self.color_percentages.items()])

        # Create the prompt for GPT-4
        prompt = f"""
        As an experienced wound care specialist nurse, analyze this diabetic wound based on the following color percentages and provide a detailed assessment and recommendations:

        {color_info}

        Please provide:
        1. An assessment of the wound condition based on these colors.
        2. The stage of healing the wound appears to be in.
        3. Specific treatment recommendations.
        4. Any warning signs or concerns.
        5. Follow-up care instructions.

        Please format your response in clear sections with headers.
        """

        try:
            # Call the GPT-4 API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an experienced wound care specialist nurse."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0.7
            )

            # Extract the generated response
            return response["choices"][0]["message"]["content"]

        except Exception as e:
            # Handle errors gracefully
            return f"Error generating report: {str(e)}"



    def show_results_page(self):
        if self.current_window:
            self.current_window.destroy()

        self.current_window = tk.Tk()
        self.current_window.title("Analysis Results")
        self.current_window.geometry("1000x800")
        self.current_window.configure(bg="#FCBC14")

        # Add title
        title_label = tk.Label(self.current_window,
                             text="Wound Analysis Report",
                             font=("courier 10 pitch", 18, "bold"),
                             bg="#FCBC14",
                             fg="black")
        title_label.pack(pady=20)

        # Create a frame for the report
        report_frame = tk.Frame(self.current_window, bg="#FCBC14", padx=20, pady=20)
        report_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Add scrolled text widget for the report
        report_text = scrolledtext.ScrolledText(report_frame,
                                              wrap=tk.WORD,
                                              font=("Arial", 12),
                                              width=80,
                                              height=25)
        report_text.pack(padx=10, pady=10)

        # Generate and display the report
        report_content = self.generate_llm_report()
        report_text.insert(tk.END, report_content)
        report_text.configure(state='disabled')  # Make text read-only

        # Add buttons
        button_frame = tk.Frame(self.current_window, bg="#FCBC14")
        button_frame.pack(pady=20)

        restart_button = tk.Button(button_frame,
                                 text="Start New Analysis",
                                 command=self.show_welcome_page,
                                 font=("courier 10 pitch", 12),
                                 bg="black",
                                 fg="#FCBC14")
        restart_button.pack(side=tk.LEFT, padx=10)

        exit_button = tk.Button(button_frame,
                              text="Exit",
                              command=self.current_window.destroy,
                              font=("courier 10 pitch", 12),
                              bg="black",
                              fg="#FCBC14")
        exit_button.pack(side=tk.LEFT, padx=10)

if __name__ == "__main__":
    app = WoundAnalysisApp()
