import os
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pandas as pd
from mlp import MLP  # Assuming you have the MLP class in a file named "mlp.py"

class TrainedModelGUI:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()

        self.root = tk.Tk()
        self.root.title("Trained Model GUI")

        self.data = None
        self.data_cols = None
        self.prediction_col = None
        self.error_col = None

        self.table = None
        self.prediction_var = tk.StringVar()
        self.error_var = tk.StringVar()

        self.setup_gui()
        self.root.mainloop()

    def setup_gui(self):
        self.create_menu()

        frame = ttk.Frame(self.root)
        frame.pack(padx=10, pady=10)

        self.load_data_button = ttk.Button(frame, text="Load Data", command=self.load_data)
        self.load_data_button.grid(row=0, column=0, columnspan=2, pady=5)

        self.table = ttk.Treeview(frame)
        self.table["columns"] = ("Prediction", "Error")
        self.table.heading("#0", text="Data")
        self.table.heading("Prediction", text="Prediction")
        self.table.heading("Error", text="Error")
        self.table.column("#0", width=150)
        self.table.column("Prediction", width=100)
        self.table.column("Error", width=100)
        self.table.grid(row=1, column=0, columnspan=2)

        self.prediction_label = ttk.Label(frame, text="Prediction:")
        self.prediction_label.grid(row=2, column=0, pady=5)

        self.prediction_entry = ttk.Entry(frame, textvariable=self.prediction_var, state="readonly")
        self.prediction_entry.grid(row=2, column=1, pady=5)

        self.error_label = ttk.Label(frame, text="Error:")
        self.error_label.grid(row=3, column=0, pady=5)

        self.error_entry = ttk.Entry(frame, textvariable=self.error_var, state="readonly")
        self.error_entry.grid(row=3, column=1, pady=5)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model", command=self.load_model)

    def load_model(self):
        model = MLP()  
        if os.path.exists(self.model_path):
            model.load_state_dict(torch.load(self.model_path))
            model.eval()
            return model
        else:
            print("Model file not found.")
            return None

    def load_data(self):
        file_path = filedialog.askopenfilename(title="Select Data File", filetypes=[("Excel Files", "*.xlsx")])
        if file_path:
            data = pd.read_excel(file_path)
            self.data_cols = data.columns.tolist()

            # Adding two columns for prediction and error
            data["Prediction"] = ""
            data["Error"] = ""

            self.data = data
            self.display_data()

    def display_data(self):
        if self.data is not None:
            for index, row in self.data.iterrows():
                self.table.insert("", "end", text=index, values=(row["Prediction"], row["Error"]))

    def update_prediction_and_error(self):
        if self.data is not None:
            # Assuming that the input variables are from columns 0 to n-3, and the target variable is at n-2
            input_cols = self.data_cols[:-2]
            target_col = self.data_cols[-2]

            for index, row in self.data.iterrows():
                inputs = torch.tensor(row[input_cols].values, dtype=torch.float).unsqueeze(0)
                with torch.no_grad():
                    prediction = self.model(inputs)
                    error = abs(prediction.item() - row[target_col])
                self.data.at[index, "Prediction"] = prediction.item()
                self.data.at[index, "Error"] = error

            self.display_data()

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    model_path = "models/trained_model.pt"  # Update with the correct path to the trained model file
    gui = TrainedModelGUI(model_path)
