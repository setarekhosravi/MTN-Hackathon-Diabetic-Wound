import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
from test_for_app import (show_wound_parts, analyze_colors, plot_color_histogram, segment_wound_parts,
                  display_segmented_parts, predict_cluster_for_wound)
from crop_images import find_contour, crop
from preprocess import preprocess_image
# from groq import Groq
# from transformers import pipeline
from transformers import pipeline
import torch

class WoundAnalysisApp:
    def __init__(self):
        self.current_window = None
        self.selected_image = None
        self.selected_model = None
        self.preprocessed_image = None
        self.color_percentages = None
        # self.api_key = "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"
        # self.groq_client = Groq(api_key=self.api_key)
        self.text_generator = pipeline("text-generation", model="gpt2")
        # self.pipe = pipeline(
        #     "text-generation", 
        #     model="meta-llama/Llama-3.2-1B", 
        #     torch_dtype=torch.bfloat16, 
        #     device_map="auto"
        # )
        self.show_welcome_page()

    def show_welcome_page(self):
        if self.current_window:
            self.current_window.destroy()
        
        self.current_window = tk.Tk()
        self.current_window.title("Diabetic Wound Segmentation and Clustering")
        self.current_window.geometry("600x400")

        # Load and set background image
        background_image = Image.open("irancell-ehsanmobile.blog.ir.jpg")
        background_photo = ImageTk.PhotoImage(background_image)
        
        background_label = tk.Label(self.current_window, image=background_photo)
        background_label.image = background_photo
        background_label.place(x=0, y=0, relwidth=1, relheight=1)

        frame = tk.Frame(self.current_window, bg="#FCBC14", padx=25, pady=10)
        frame.place(relx=0.5, rely=0.15, anchor=tk.CENTER)

        project_name_label = tk.Label(frame, text="Diabetic Wound Segmentation and Clustering",
                                      bg="#FCBC14", fg="black", font=("Comic Sans MS", 16, "bold"))
        project_name_label.pack(pady=1)

        welcome_label = tk.Label(frame, text="Welcome!", 
                                 bg="#FCBC14", fg="black", font=("Comic Sans MS", 12, "bold"))
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
                                     font=("Comic Sans MS", 14, "bold"), bg="#FCBC14", fg="black")
        instruction_label.pack(pady=10)

        browse_button = tk.Button(self.current_window, text="Browse Image",
                                  command=self.browse_image, font=("Comic Sans MS", 12),
                                  bg="black", fg="#FCBC14")
        browse_button.pack(pady=10)

        self.image_label = tk.Label(self.current_window, text="No Image Selected",
                                    font=("Comic Sans MS", 12), bg="#FCBC14", fg="black")
        self.image_label.pack(pady=10)

        self.model_var = tk.StringVar(value="u-net")
        u_net_radio = tk.Radiobutton(self.current_window, text="U-Net",
                                     variable=self.model_var, value="u-net",
                                     font=("Comic Sans MS", 12), bg="#FCBC14", fg="black")
        u_net_radio.pack(pady=5)

        deep_skin_radio = tk.Radiobutton(self.current_window, text="DeepSkin",
                                         variable=self.model_var, value="deepskin",
                                         font=("Comic Sans MS", 12), bg="#FCBC14", fg="black")
        deep_skin_radio.pack(pady=5)

        next_button = tk.Button(self.current_window, text="Next",
                                command=self.show_preprocessing_page,
                                font=("Comic Sans MS", 12), bg="black", fg="#FCBC14")
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
                                command=self.show_clustering_page, font=("Comic Sans MS", 12),
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
        tk.Label(self.current_window, text=f"Cluster ID: {cluster_id}", font=("Comic Sans MS", 14), bg="#FCBC14").pack(pady=20)

        next_button = tk.Button(self.current_window, text="Next",
                                command=self.show_wound_parts_page, font=("Comic Sans MS", 12),
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
                              font=("Comic Sans MS", 12),
                              bg="black", fg="#FCBC14")
        next_button.pack(pady=20)

    # def generate_llm_report(self):
    #     # Create a detailed prompt for the LLM
    #     color_info = "\n".join([f"{color}: {percentage:.2f}%" for color, percentage in self.color_percentages.items()])
        
    #     prompt = f"""As an experienced wound care specialist nurse, please analyze this diabetic wound based on the following color percentages and provide a detailed assessment and recommendations:

    #             {color_info}

    #             Please provide:
    #             1. An assessment of the wound condition based on these colors
    #             2. What stage of healing the wound appears to be in
    #             3. Specific treatment recommendations
    #             4. Any warning signs or concerns
    #             5. Follow-up care instructions

    #             Please format your response in clear sections with headers."""

    #     try:
    #         chat_completion = self.groq_client.chat.completions.create(
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": "You are an experienced wound care specialist nurse with expertise in diabetic wound assessment and treatment."
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": prompt,
    #                 }
    #             ],
    #             model="llama3-8b-8192",
    #             temperature=0.5,
    #             max_tokens=1024,
    #             top_p=1,
    #             stop=", 6",
    #             stream=False,
    #         )
    #         return chat_completion.choices[0].message.content
    #     except Exception as e:
    #         return f"Error generating report: {str(e)}\n\nPlease check your API key and internet connection."

    def generate_llm_report(self):
        # Create a detailed prompt for the LLM with examples
        color_info = "\n".join([f"{color}: {percentage:.2f}%" for color, percentage in self.color_percentages.items()])
        
        prompt = f"""You are an experienced wound care specialist nurse with expertise in diabetic wound assessment and treatment.

        Here are the color percentages of a diabetic wound:
        {color_info}

        Based on these colors, provide a detailed report:

        Example:
        - Colors: Red: 50%, Yellow: 30%, Black: 20%
        - Assessment: The wound shows 50% red, indicating active inflammation. The 30% yellow area suggests slough, which may delay healing, and 20% black indicates necrosis.
        - Stage of Healing: The wound appears to be in the inflammatory stage.
        - Recommendations: Debridement of necrotic tissue is advised. Apply moist dressings to promote granulation.
        - Warning Signs: Risk of infection in the sloughy and necrotic areas. Monitor for fever or increased redness around the wound.
        - Follow-up Care: Weekly wound evaluations and possible antibiotics if infection occurs.

        Now provide the analysis for the given wound:
        - Colors: {color_info}
        - Assessment:"""

        # Generate the report using the Hugging Face text generation pipeline
        try:
            response = self.text_generator(prompt, max_length=500, num_return_sequences=1)
            return response[0]['generated_text']
        except Exception as e:
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
                             font=("Comic Sans MS", 18, "bold"),
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
                                 font=("Comic Sans MS", 12),
                                 bg="black",
                                 fg="#FCBC14")
        restart_button.pack(side=tk.LEFT, padx=10)

        exit_button = tk.Button(button_frame,
                              text="Exit",
                              command=self.current_window.destroy,
                              font=("Comic Sans MS", 12),
                              bg="black",
                              fg="#FCBC14")
        exit_button.pack(side=tk.LEFT, padx=10)

if __name__ == "__main__":
    app = WoundAnalysisApp()
