import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import logging
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from tkinter import ttk
import threading
from PIL import Image, ImageTk
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s [line %(lineno)d]')

class LOAD:
    def __init__(self, file_path):
        self.filepath = file_path


class EDA(LOAD):
    def cleanfile(self):
        self.filee = None 
        
        if self.filepath is not None:
            try:
                if os.path.isfile(self.filepath):
                    self.filee = pd.read_csv(self.filepath)
                    
                    # Clean duplicates
                    if self.filee.duplicated().sum() > 0:
                        self.filee.drop_duplicates(keep="first", inplace=True)
                        self.filee.reset_index(drop=True, inplace=True)

                    # Clean missing values
                    if self.filee.isnull().sum().sum() > 0:
                        self.filee.dropna(inplace=True)
                    
                    return self.filee
                
                else:
                    logging.error("The specified path is not a file.")
                    return None
                    
            except Exception as e:
                logging.error(e)
                return None
        else:
            logging.error("File path is None or invalid.")       
            return None
            
    def analysis(self):
        if self.filee is not None:
            try:
                # Convert Date to datetime and extract year
                if 'Date' in self.filee.columns:
                    self.filee['Date'] = pd.to_datetime(self.filee['Date']).dt.year
                else:
                    raise Exception("Date column not found")

                # Create grouped data
                gr1 = self.filee.groupby("Date").agg({"High": "max", "Low": "min"})
                gr2 = self.filee.groupby("Date").agg({"Open": "max", "Close": "min"})
                gr3 = self.filee.groupby("Date")["Volume"].mean().reset_index()
                
                return gr1, gr2, gr3
        
            except Exception as e:
                logging.error(e)
                return None, None, None
        else:
            logging.error("File is Empty.......")
            return None, None, None
    
    def savefile(self):
        if self.filee is not None:
            try:
                self.filee.to_csv("bitcoin_mod.csv", index=False)
                return True
            except Exception as e:
                logging.error(e)
                return False
        else:
            logging.error("file not found.........")
            return False
            
    def visualize1(self, gr1, gr2, plot_type):
        try:
            fig = plt.figure(figsize=(10, 6), dpi=100)  # Fixed size for all plots
            
            if plot_type == 0:  # High/Low prices
                df_melted1 = gr1.reset_index().melt(id_vars='Date', value_vars=['High', 'Low'], 
                                            var_name='Price Type', value_name='Value')
                sns.barplot(x='Date', y='Value', hue='Price Type', data=df_melted1, palette="viridis")
                plt.title("High and Low of Bitcoin (2014-2024)", fontsize=14)
                plt.xticks(rotation=45)
                plt.ylabel("Price")
                plt.xlabel("Year")
                
            elif plot_type == 1:  # Open/Close prices
                df_melted2 = gr2.reset_index().melt(id_vars='Date', value_vars=['Open', 'Close'], 
                                            var_name='Price Type', value_name='Value')
                sns.barplot(x='Date', y='Value', hue='Price Type', data=df_melted2, palette="winter")
                plt.title("Open and Close of Bitcoin (2014-2024)", fontsize=14)
                plt.xticks(rotation=45)
                plt.ylabel("Price")
                plt.xlabel("Year")
                
            elif plot_type == 2:  # Volume line plot
                sns.lineplot(x=self.filee["Date"], y=self.filee["Volume"], palette="winter", linewidth=2)
                plt.title("Average Volume (2014-2024)", fontsize=14)
                plt.xlabel("Year")
                plt.ylabel("Volume")
                
            elif plot_type == 3:  # Heatmap
                sns.heatmap(self.filee.corr(numeric_only=True), annot=True, cmap="coolwarm", linewidths=0.5)
                plt.title("Correlation Heatmap", fontsize=14)
                
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logging.error(f"Visualization error: {e}")
            return None


class ML(EDA):
    def ml(self):
        try:
            df = self.filee.copy()

            # Prepare data for High price prediction
            X1 = df.drop(columns=["Date", "High"], axis=1)
            Y1 = df["High"]
            x_train1, x_test1, y_train1, y_test1 = train_test_split(X1, Y1, random_state=101, train_size=0.7)

            # Prepare data for Low price prediction
            X2 = df.drop(columns=["Date", "Low"], axis=1)
            Y2 = df["Low"]
            x_train2, x_test2, y_train2, y_test2 = train_test_split(X2, Y2, random_state=42, train_size=0.7)

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor()
            }

            results = []
            
            for name, model in models.items():
                # Train and predict for High prices
                model.fit(x_train1, y_train1)
                preds1 = model.predict(x_test1)
                mae1 = mean_absolute_error(y_test1, preds1)
                mse1 = mean_squared_error(y_test1, preds1)
                r2_1 = r2_score(y_test1, preds1)

                # Train and predict for Low prices
                model.fit(x_train2, y_train2)
                preds2 = model.predict(x_test2)
                mae2 = mean_absolute_error(y_test2, preds2)
                mse2 = mean_squared_error(y_test2, preds2)
                r2_2 = r2_score(y_test2, preds2)

                # Create prediction comparison
                comparison = pd.DataFrame({
                    'Adj Close': x_test1['Adj Close'].values,
                    'Close': x_test1['Close'].values,
                    'Open': x_test1['Open'].values,
                    'Volume': x_test1['Volume'].values,
                    'High(actual)': y_test1.values,
                    'High(predicted)': preds1,
                    'Low(actual)': y_test2.values,
                    'Low(predicted)': preds2
                })

                # Create prediction plots
                fig1 = self._create_prediction_plot(y_test1.values, preds1, f"{name} - High Price Prediction")
                fig2 = self._create_prediction_plot(y_test2.values, preds2, f"{name} - Low Price Prediction")

                # Store results
                results.append({
                    'model': name,
                    'high_mae': mae1,
                    'high_mse': mse1,
                    'high_r2': r2_1,
                    'low_mae': mae2,
                    'low_mse': mse2,
                    'low_r2': r2_2,
                    'comparison': comparison,
                    'high_plot': fig1,
                    'low_plot': fig2
                })

            return results
            
        except Exception as e:
            logging.error(f"ML error: {e}")
            return None

    def _create_prediction_plot(self, actual, predicted, title):
        fig = plt.figure(figsize=(12, 5))
        plt.plot(actual, label='Actual', marker='o', markersize=4)
        plt.plot(predicted, label='Predicted', marker='x', markersize=4)
        plt.title(title, fontsize=12)
        plt.xlabel("Samples", fontsize=10)
        plt.ylabel("Price", fontsize=10)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


class BitcoinApp:
    def __init__(self, root):
        self.visualization_index = 0  # To track which visualization to show
        self.root = root
        self.root.title("Bitcoin Analysis & Prediction")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize variables
        self.file_path = None
        self.model = None
        self.gr1 = None
        self.gr2 = None
        self.gr3 = None
        self.ml_results = None
        self.current_plot = None
        self.current_canvas = None
        
        # Configure styles
        self._configure_styles()
        
        # Setup UI
        self._setup_ui()
        
        # Center the window
        self._center_window()

    def _configure_styles(self):
        self.style = ttk.Style()
        
        # Theme configuration
        self.style.theme_use('clam')
        
        # Colors
        bg_color = '#2d2d3d'
        fg_color = '#ffffff'
        accent_color = '#4e8cff'
        secondary_color = '#3d3d4d'
        text_color = '#e0e0e0'
        
        # Configure styles
        self.style.configure('.', background=bg_color, foreground=fg_color)
        self.style.configure('TFrame', background=bg_color)
        self.style.configure('TLabel', background=bg_color, foreground=fg_color, font=('Segoe UI', 10))
        self.style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=6, 
                           background=secondary_color, foreground=fg_color)
        self.style.map('TButton', 
                      background=[('active', accent_color), ('pressed', accent_color)],
                      foreground=[('active', fg_color), ('pressed', fg_color)])
        self.style.configure('TNotebook', background=bg_color, borderwidth=0)
        self.style.configure('TNotebook.Tab', background=secondary_color, foreground=fg_color,
                            padding=[10, 5], font=('Segoe UI', 10, 'bold'))
        self.style.map('TNotebook.Tab', 
                      background=[('selected', accent_color), ('active', accent_color)],
                      foreground=[('selected', fg_color), ('active', fg_color)])
        self.style.configure('Treeview', background=secondary_color, fieldbackground=secondary_color, 
                           foreground=fg_color, rowheight=25)
        self.style.configure('Treeview.Heading', background=accent_color, foreground=fg_color)
        self.style.configure('Vertical.TScrollbar', background=secondary_color, 
                           arrowcolor=fg_color, troughcolor=bg_color)
        self.style.configure('Horizontal.TScrollbar', background=secondary_color, 
                           arrowcolor=fg_color, troughcolor=bg_color)
        
        # Custom styles
        self.style.configure('Title.TLabel', font=('Segoe UI', 18, 'bold'), foreground=accent_color)
        self.style.configure('Subtitle.TLabel', font=('Segoe UI', 12), foreground='#aaaaaa')
        self.style.configure('Status.TLabel', font=('Segoe UI', 9), foreground='#888888')
        self.style.configure('Accent.TButton', background=accent_color, foreground=fg_color)
        self.style.configure('Secondary.TButton', background=secondary_color, foreground=fg_color)

    def _setup_ui(self):
        # Main container
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header
        self.header_frame = ttk.Frame(self.main_frame)
        self.header_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.title_label = ttk.Label(self.header_frame, text="Bitcoin Price Analysis & Prediction", 
                                   style='Title.TLabel')
        self.title_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(self.header_frame, text="No file loaded", style='Status.TLabel')
        self.status_label.pack(side=tk.RIGHT)
        
        # Body - using Notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # File Tab
        self.file_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.file_tab, text="File Operations")
        
        # EDA Tab
        self.eda_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.eda_tab, text="Exploratory Analysis")
        
        # ML Tab
        self.ml_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.ml_tab, text="Machine Learning")
        
        # Results Tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Results")
        
        # Initialize tabs
        self._setup_file_tab()
        self._setup_eda_tab()
        self._setup_ml_tab()
        self._setup_results_tab()
        
        # Progress bar
        self.progress = ttk.Progressbar(self.main_frame, mode='determinate')
        self.progress.pack(fill=tk.X, pady=(5, 0))
        
        # Console output
        self.console_frame = ttk.LabelFrame(self.main_frame, text="Console Output")
        self.console_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.console = scrolledtext.ScrolledText(self.console_frame, wrap=tk.WORD, 
                                               bg='#1e1e2d', fg='#e0e0e0',
                                               insertbackground='white',
                                               font=('Consolas', 9))
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Redirect stdout to console
        import sys
        sys.stdout = TextRedirector(self.console, "stdout")
        sys.stderr = TextRedirector(self.console, "stderr")

    def _setup_file_tab(self):
        # File selection section
        file_select_frame = ttk.LabelFrame(self.file_tab, text="File Selection")
        file_select_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.path_label = ttk.Label(file_select_frame, text="No file selected", 
                                  style='Subtitle.TLabel')
        self.path_label.pack(side=tk.LEFT, padx=5, pady=5)
        
        select_btn = ttk.Button(file_select_frame, text="Select CSV File", 
                               command=self.select_file, style='Accent.TButton')
        select_btn.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # File operations section
        file_ops_frame = ttk.LabelFrame(self.file_tab, text="File Operations")
        file_ops_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        btn_config = {
            'style': 'Secondary.TButton',
            'width': 20
        }
        
        clean_btn = ttk.Button(file_ops_frame, text="Clean File", 
                              command=self.clean_file, **btn_config)
        clean_btn.grid(row=0, column=0, padx=10, pady=10)
        
        save_btn = ttk.Button(file_ops_frame, text="Save Cleaned File", 
                             command=self.save_file, **btn_config)
        save_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # File preview section
        preview_frame = ttk.LabelFrame(self.file_tab, text="File Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.file_preview = ttk.Treeview(preview_frame)
        self.file_preview.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, 
                                command=self.file_preview.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_preview.configure(yscrollcommand=scroll_y.set)
        
        scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, 
                                command=self.file_preview.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.file_preview.configure(xscrollcommand=scroll_x.set)

    def _setup_eda_tab(self):
        # EDA buttons section
        eda_btn_frame = ttk.Frame(self.eda_tab)
        eda_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        eda_btn = ttk.Button(eda_btn_frame, text="Perform EDA", 
                            command=self.perform_eda, style='Accent.TButton')
        eda_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        visualize_btn = ttk.Button(eda_btn_frame, text="Visualize Data", 
                                  command=self.visualize_data, style='Secondary.TButton')
        visualize_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # EDA results section
        eda_results_frame = ttk.LabelFrame(self.eda_tab, text="EDA Results")
        eda_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.eda_text = scrolledtext.ScrolledText(eda_results_frame, wrap=tk.WORD, 
                                                bg='#1e1e2d', fg='#e0e0e0',
                                                insertbackground='white',
                                                font=('Consolas', 9))
        self.eda_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Plot display section
        self.plot_frame = ttk.Frame(eda_results_frame)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def _setup_ml_tab(self):
        # ML buttons section
        ml_btn_frame = ttk.Frame(self.ml_tab)
        ml_btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ml_btn = ttk.Button(ml_btn_frame, text="Run ML Models", 
                           command=self.run_models, style='Accent.TButton')
        ml_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Model selection
        model_frame = ttk.LabelFrame(self.ml_tab, text="Model Selection")
        model_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.model_var = tk.StringVar(value="Random Forest")
        
        ttk.Radiobutton(model_frame, text="Random Forest", variable=self.model_var, 
                       value="Random Forest").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(model_frame, text="Linear Regression", variable=self.model_var, 
                       value="Linear Regression").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(model_frame, text="Decision Tree", variable=self.model_var, 
                       value="Decision Tree").pack(side=tk.LEFT, padx=10)
        
        # ML results section
        ml_results_frame = ttk.LabelFrame(self.ml_tab, text="ML Results")
        ml_results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.ml_text = scrolledtext.ScrolledText(ml_results_frame, wrap=tk.WORD, 
                                               bg='#1e1e2d', fg='#e0e0e0',
                                               insertbackground='white',
                                               font=('Consolas', 9))
        self.ml_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _setup_results_tab(self):
        # Results display section
        results_frame = ttk.Frame(self.results_tab)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Placeholder for results visualization
        self.results_canvas = tk.Canvas(results_frame, bg='#2d2d3d', highlightthickness=0)
        self.results_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.results_label = ttk.Label(self.results_canvas, text="Results will be displayed here", 
                                     style='Subtitle.TLabel')
        self.results_label.pack(pady=50)

    def _center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def _update_status(self, message):
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def _update_progress(self, value):
        self.progress['value'] = value
        self.root.update_idletasks()

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path = file_path
            self.path_label.config(text=file_path)
            self.model = ML(file_path=self.file_path)
            self._update_status(f"Loaded: {os.path.basename(file_path)}")
            
            # Preview the file
            self._preview_file()

    def _preview_file(self):
        try:
            # Clear existing preview
            for item in self.file_preview.get_children():
                self.file_preview.delete(item)
            
            # Read just the first few rows for preview
            df = pd.read_csv(self.file_path, nrows=10)
            
            # Set up columns
            self.file_preview['columns'] = list(df.columns)
            
            # Create headings
            for col in df.columns:
                self.file_preview.heading(col, text=col)
                self.file_preview.column(col, width=100, anchor=tk.CENTER)
            
            # Add data
            for _, row in df.iterrows():
                self.file_preview.insert('', tk.END, values=list(row))
                
        except Exception as e:
            self._log_error(f"Error previewing file: {str(e)}")

    def clean_file(self):
        if self.model:
            self._update_status("Cleaning file...")
            self._update_progress(0)
            
            def _clean():
                try:
                    start_time = time.time()
                    cleaned_data = self.model.cleanfile()
                    self._update_progress(50)
                    
                    if cleaned_data is not None:
                        # Update preview with cleaned data
                        for item in self.file_preview.get_children():
                            self.file_preview.delete(item)
                        
                        # Show first 10 rows of cleaned data
                        for _, row in cleaned_data.head(10).iterrows():
                            self.file_preview.insert('', tk.END, values=list(row))
                        
                        self._update_progress(100)
                        self._update_status("File cleaned successfully")
                        self._log_message(f"File cleaned in {time.time() - start_time:.2f} seconds")
                    else:
                        self._update_progress(0)
                        self._update_status("Error cleaning file")
                except Exception as e:
                    self._update_progress(0)
                    self._update_status("Error cleaning file")
                    self._log_error(f"Error cleaning file: {str(e)}")
            
            threading.Thread(target=_clean, daemon=True).start()
        else:
            self._show_error("Please select a file first.")

    def save_file(self):
        if self.model:
            self._update_status("Saving file...")
            self._update_progress(0)
            
            def _save():
                try:
                    start_time = time.time()
                    success = self.model.savefile()
                    self._update_progress(100 if success else 0)
                    
                    if success:
                        self._update_status("File saved as bitcoin_mod.csv")
                        self._log_message(f"File saved in {time.time() - start_time:.2f} seconds")
                    else:
                        self._update_status("Error saving file")
                except Exception as e:
                    self._update_progress(0)
                    self._update_status("Error saving file")
                    self._log_error(f"Error saving file: {str(e)}")
            
            threading.Thread(target=_save, daemon=True).start()
        else:
            self._show_error("Please select and clean a file first.")

    def perform_eda(self):
        if self.model:
            self._update_status("Performing EDA...")
            self._update_progress(0)
            self.eda_text.delete(1.0, tk.END)  # Clear previous results
            
            def _perform_eda():
                try:
                    start_time = time.time()
                    self.gr1, self.gr2, self.gr3 = self.model.analysis()
                    self._update_progress(50)
                    
                    if self.gr1 is not None and self.gr2 is not None:
                        # Display EDA results in text widget
                        self.eda_text.insert(tk.END, "=== High and Low Prices by Year ===\n")
                        self.eda_text.insert(tk.END, str(self.gr1) + "\n\n")
                        
                        self.eda_text.insert(tk.END, "=== Open and Close Prices by Year ===\n")
                        self.eda_text.insert(tk.END, str(self.gr2) + "\n\n")
                        
                        self.eda_text.insert(tk.END, "=== Average Volume by Year ===\n")
                        self.eda_text.insert(tk.END, str(self.gr3) + "\n\n")
                        
                        self._update_progress(100)
                        self._update_status("EDA completed successfully")
                        self._log_message(f"EDA completed in {time.time() - start_time:.2f} seconds")
                    else:
                        self._update_progress(0)
                        self._update_status("Error performing EDA")
                except Exception as e:
                    self._update_progress(0)
                    self._update_status("Error performing EDA")
                    self._log_error(f"Error performing EDA: {str(e)}")
            
            threading.Thread(target=_perform_eda, daemon=True).start()
        else:
            self._show_error("Please select and clean a file first.")

    def visualize_data(self):
        if self.model and self.gr1 is not None and self.gr2 is not None:
            self._update_status("Generating visualizations...")
            self._update_progress(0)
            
            def _visualize():
                try:
                    start_time = time.time()
                    
                    # Clear previous plot
                    for widget in self.plot_frame.winfo_children():
                        widget.destroy()
                    
                    # Get the current visualization index and cycle it
                    plot_type = self.visualization_index % 4  # We have 4 plot types
                    self.visualization_index += 1
                    
                    # Create and display the plot
                    fig = self.model.visualize1(self.gr1, self.gr2, plot_type)
                    self._update_progress(50)
                    
                    if fig is not None:
                        # Create a container frame with scrollbars
                        container = ttk.Frame(self.plot_frame)
                        container.pack(fill=tk.BOTH, expand=True)
                        
                        # Create canvas with scrollbars
                        canvas = tk.Canvas(container)
                        scroll_y = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
                        scroll_x = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
                        
                        # Configure canvas scrolling
                        canvas.configure(
                            yscrollcommand=scroll_y.set,
                            xscrollcommand=scroll_x.set,
                            bg='#2d2d3d',
                            highlightthickness=0
                        )
                        
                        # Pack scrollbars and canvas
                        scroll_y.pack(side="right", fill="y")
                        scroll_x.pack(side="bottom", fill="x")
                        canvas.pack(side="left", fill="both", expand=True)
                        
                        # Create frame for the plot inside canvas
                        plot_frame = ttk.Frame(canvas)
                        canvas.create_window((0, 0), window=plot_frame, anchor="nw")
                        
                        # Display the matplotlib figure
                        canvas_fig = FigureCanvasTkAgg(fig, master=plot_frame)
                        canvas_fig.draw()
                        canvas_fig.get_tk_widget().pack(padx=10, pady=10)
                        
                        # Add navigation toolbar below the plot
                        toolbar = NavigationToolbar2Tk(canvas_fig, plot_frame)
                        toolbar.update()
                        
                        # Configure scroll region after widgets are drawn
                        def _configure_scrollregion(event):
                            canvas.configure(scrollregion=canvas.bbox("all"))
                            # Limit minimum canvas size to figure size
                            canvas.config(width=fig.get_size_inches()[0]*100, 
                                        height=fig.get_size_inches()[1]*100)
                        
                        plot_frame.bind("<Configure>", _configure_scrollregion)
                        
                        self._update_progress(100)
                        self._update_status(f"Showing visualization {self.visualization_index % 4 + 1}/4")
                        self._log_message(f"Visualization generated in {time.time() - start_time:.2f} seconds")
                    else:
                        self._update_progress(0)
                        self._update_status("Error generating visualization")
                except Exception as e:
                    self._update_progress(0)
                    self._update_status("Error generating visualization")
                    self._log_error(f"Error generating visualization: {str(e)}")
            
            threading.Thread(target=_visualize, daemon=True).start()
        else:
            self._show_error("Please perform EDA first.")
        
    def run_models(self):
        if self.model:
            selected_model = self.model_var.get()  # Get the selected model
            
            self._update_status(f"Running {selected_model} model...")
            self._update_progress(0)
            self.ml_text.delete(1.0, tk.END)  # Clear previous results
            
            def _run_models():
                try:
                    start_time = time.time()
                    ml_results = self.model.ml()  # Get all results
                    self._update_progress(50)
                    
                    if ml_results is not None:
                        # Find the selected model's results
                        model_result = next((r for r in ml_results if r['model'] == selected_model), None)
                        
                        if model_result:
                            self.ml_text.insert(tk.END, f"\n=== {model_result['model']} ===\n")
                            self.ml_text.insert(tk.END, "High Price Prediction:\n")
                            self.ml_text.insert(tk.END, f"MAE: {model_result['high_mae']:.4f}\n")
                            self.ml_text.insert(tk.END, f"MSE: {model_result['high_mse']:.2e}\n")
                            self.ml_text.insert(tk.END, f"R²: {model_result['high_r2']:.4f}\n\n")
                            
                            self.ml_text.insert(tk.END, "Low Price Prediction:\n")
                            self.ml_text.insert(tk.END, f"MAE: {model_result['low_mae']:.4f}\n")
                            self.ml_text.insert(tk.END, f"MSE: {model_result['low_mse']:.2e}\n")
                            self.ml_text.insert(tk.END, f"R²: {model_result['low_r2']:.4f}\n\n")
                            
                            self.ml_text.insert(tk.END, "Sample Predictions:\n")
                            self.ml_text.insert(tk.END, str(model_result['comparison'].head()) + "\n\n")
                            
                            # Show the selected model's results
                            self._show_model_results(model_result)
                            
                            self._update_progress(100)
                            self._update_status(f"{selected_model} completed")
                            self._log_message(f"{selected_model} completed in {time.time() - start_time:.2f} seconds")
                        else:
                            self._update_progress(0)
                            self._update_status("Error: Model results not found")
                    else:
                        self._update_progress(0)
                        self._update_status("Error running ML models")
                except Exception as e:
                    self._update_progress(0)
                    self._update_status("Error running ML models")
                    self._log_error(f"Error running ML models: {str(e)}")
            
            threading.Thread(target=_run_models, daemon=True).start()
        else:
            self._show_error("Please select and clean a file first.")

    def _show_model_results(self, model_result):
        # Clear previous results
        for widget in self.results_canvas.winfo_children():
            widget.destroy()
        
        # Create a scrollable frame
        canvas = tk.Canvas(self.results_canvas, bg='#2d2d3d', highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.results_canvas, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Model title
        title = ttk.Label(scrollable_frame, text=f"{model_result['model']} Results", 
                        style='Title.TLabel')
        title.pack(pady=(0, 20))
        
        # Metrics frame
        metrics_frame = ttk.Frame(scrollable_frame)
        metrics_frame.pack(fill=tk.X, pady=5)
        
        # High price metrics
        high_frame = ttk.LabelFrame(metrics_frame, text="High Price Prediction")
        high_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(high_frame, text=f"MAE: {model_result['high_mae']:.4f}", 
                font=('Segoe UI', 10)).pack(anchor=tk.W, pady=2)
        ttk.Label(high_frame, text=f"MSE: {model_result['high_mse']:.2e}", 
                font=('Segoe UI', 10)).pack(anchor=tk.W, pady=2)
        ttk.Label(high_frame, text=f"R²: {model_result['high_r2']:.4f}", 
                font=('Segoe UI', 10)).pack(anchor=tk.W, pady=2)
        
        # Low price metrics
        low_frame = ttk.LabelFrame(metrics_frame, text="Low Price Prediction")
        low_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        ttk.Label(low_frame, text=f"MAE: {model_result['low_mae']:.4f}", 
                font=('Segoe UI', 10)).pack(anchor=tk.W, pady=2)
        ttk.Label(low_frame, text=f"MSE: {model_result['low_mse']:.2e}", 
                font=('Segoe UI', 10)).pack(anchor=tk.W, pady=2)
        ttk.Label(low_frame, text=f"R²: {model_result['low_r2']:.4f}", 
                font=('Segoe UI', 10)).pack(anchor=tk.W, pady=2)
        
        # Plots frame
        plots_frame = ttk.Frame(scrollable_frame)
        plots_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Display plots vertically
        if model_result['high_plot']:
            high_canvas = FigureCanvasTkAgg(model_result['high_plot'], master=plots_frame)
            high_canvas.draw()
            high_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        if model_result['low_plot']:
            low_canvas = FigureCanvasTkAgg(model_result['low_plot'], master=plots_frame)
            low_canvas.draw()
            low_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    def _log_message(self, message):
        self.console.insert(tk.END, f"[INFO] {message}\n")
        self.console.see(tk.END)

    def _log_error(self, message):
        self.console.insert(tk.END, f"[ERROR] {message}\n", 'error')
        self.console.see(tk.END)

    def _show_error(self, message):
        messagebox.showerror("Error", message)
        self._log_error(message)


class TextRedirector:
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag
        
    def write(self, text):
        self.widget.configure(state="normal")
        self.widget.insert(tk.END, text, (self.tag,))
        self.widget.configure(state="disabled")
        self.widget.see(tk.END)
        
    def flush(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = BitcoinApp(root)
    
    # Configure console tags for colors
    app.console.tag_config('stdout', foreground='#e0e0e0')
    app.console.tag_config('stderr', foreground='#ff6b6b')
    app.console.tag_config('error', foreground='#ff6b6b')
    
    root.mainloop()
