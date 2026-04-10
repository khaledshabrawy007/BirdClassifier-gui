#Tkinter GUI
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import os

from data_loader import (
    load_dataset, train_test_split_by_class, normalize, FEATURE_COLUMNS, CLASS_LABELS
)
from classifiers import Perceptron, Adaline
from evaluation import confusion_matrix, overall_accuracy
from visualization import build_result_figure


DATASET_PATH = os.path.join(os.path.dirname(__file__), 'birds.csv')


class BirdsClassifierApp(tk.Tk):
    

    def __init__(self):
        super().__init__()
        self.title('Birds Neural Network Classifier')
        self.geometry('1100x820')
        self.resizable(True, True)
        self.configure(bg='#1E1E2E')

        self.df = load_dataset(DATASET_PATH)

        self.canvas_widget = None
        self.fig = None

        self._build_ui()



    def _build_ui(self):
        header = tk.Frame(self, bg='#313244', pady=10)
        header.pack(fill='x')
        tk.Label(header, text='🐦  Birds Neural Network Classifier',
                 font=('Courier New', 16, 'bold'),
                 bg='#313244', fg='#CBA6F7').pack()

        #Main layout: left panel + right canvas
        main = tk.Frame(self, bg='#1E1E2E')
        main.pack(fill='both', expand=True, padx=10, pady=10)

        left = tk.Frame(main, bg='#181825', width=280, relief='ridge', bd=1)
        left.pack(side='left', fill='y', padx=(0, 10))
        left.pack_propagate(False)

        right = tk.Frame(main, bg='#1E1E2E')
        right.pack(side='left', fill='both', expand=True)

        self._build_controls(left)
        self._build_output(right)

    def _section(self, parent, title):
        f = tk.LabelFrame(parent, text=title,
                          font=('Courier New', 9, 'bold'),
                          bg='#181825', fg='#89B4FA',
                          padx=8, pady=6, relief='groove', bd=1)
        f.pack(fill='x', padx=8, pady=5)
        return f

    def _label_entry(self, parent, label, default):
        row = tk.Frame(parent, bg='#181825')
        row.pack(fill='x', pady=2)
        tk.Label(row, text=label, width=16, anchor='w',
                 font=('Courier New', 9), bg='#181825', fg='#CDD6F4').pack(side='left')
        var = tk.StringVar(value=str(default))
        e = tk.Entry(row, textvariable=var, width=10,
                     font=('Courier New', 9), bg='#313244', fg='#CDD6F4',
                     insertbackground='white', relief='flat', bd=2)
        e.pack(side='left')
        return var

    def _build_controls(self, parent):

        #feature selection
        f_sec = self._section(parent, ' Features ')
        self.feat1_var = tk.StringVar(value='body_mass')
        self.feat2_var = tk.StringVar(value='beak_length')
        tk.Label(f_sec, text='Feature 1:', font=('Courier New', 9),
                 bg='#181825', fg='#CDD6F4').pack(anchor='w')
        ttk.Combobox(f_sec, textvariable=self.feat1_var,
                     values=FEATURE_COLUMNS, state='readonly',
                     font=('Courier New', 9), width=18).pack(fill='x', pady=2)
        tk.Label(f_sec, text='Feature 2:', font=('Courier New', 9),
                 bg='#181825', fg='#CDD6F4').pack(anchor='w')
        ttk.Combobox(f_sec, textvariable=self.feat2_var,
                     values=FEATURE_COLUMNS, state='readonly',
                     font=('Courier New', 9), width=18).pack(fill='x', pady=2)

        #class selection
        c_sec = self._section(parent, ' Classes ')
        tk.Label(c_sec, text='Class 1:', font=('Courier New', 9),
                 bg='#181825', fg='#CDD6F4').pack(anchor='w')
        self.class1_var = tk.StringVar(value='A')
        ttk.Combobox(c_sec, textvariable=self.class1_var,
                     values=CLASS_LABELS, state='readonly',
                     font=('Courier New', 9), width=18).pack(fill='x', pady=2)
        tk.Label(c_sec, text='Class 2:', font=('Courier New', 9),
                 bg='#181825', fg='#CDD6F4').pack(anchor='w')
        self.class2_var = tk.StringVar(value='B')
        ttk.Combobox(c_sec, textvariable=self.class2_var,
                     values=CLASS_LABELS, state='readonly',
                     font=('Courier New', 9), width=18).pack(fill='x', pady=2)

        #Hyperparameters
        h_sec = self._section(parent, ' Hyperparameters ')
        self.eta_var     = self._label_entry(h_sec, 'Learning rate:', 0.01)
        self.epochs_var  = self._label_entry(h_sec, 'Epochs:', 100)
        self.mse_var     = self._label_entry(h_sec, 'MSE threshold:', 0.001)

        #Options
        o_sec = self._section(parent, ' Options ')
        self.bias_var = tk.BooleanVar(value=True)
        tk.Checkbutton(o_sec, text='Add Bias', variable=self.bias_var,
                       font=('Courier New', 9),
                       bg='#181825', fg='#A6E3A1',
                       activebackground='#181825',
                       selectcolor='#313244').pack(anchor='w')

        self.algo_var = tk.StringVar(value='Perceptron')
        for algo in ['Perceptron', 'Adaline']:
            tk.Radiobutton(o_sec, text=algo, variable=self.algo_var, value=algo,
                           font=('Courier New', 9),
                           bg='#181825', fg='#89DCEB',
                           activebackground='#181825',
                           selectcolor='#313244').pack(anchor='w')

        #Classify single sample
        s_sec = self._section(parent, ' Classify Sample ')
        tk.Label(s_sec, text='x1 value:', font=('Courier New', 9),
                 bg='#181825', fg='#CDD6F4').pack(anchor='w')
        self.sample_x1 = tk.Entry(s_sec, font=('Courier New', 9),
                                  bg='#313244', fg='#CDD6F4',
                                  insertbackground='white', relief='flat', bd=2)
        self.sample_x1.pack(fill='x', pady=2)
        tk.Label(s_sec, text='x2 value:', font=('Courier New', 9),
                 bg='#181825', fg='#CDD6F4').pack(anchor='w')
        self.sample_x2 = tk.Entry(s_sec, font=('Courier New', 9),
                                  bg='#313244', fg='#CDD6F4',
                                  insertbackground='white', relief='flat', bd=2)
        self.sample_x2.pack(fill='x', pady=2)
        tk.Button(s_sec, text='Classify Sample',
                  command=self._classify_sample,
                  font=('Courier New', 9, 'bold'),
                  bg='#F38BA8', fg='#1E1E2E',
                  activebackground='#eba0ac',
                  relief='flat', pady=4).pack(fill='x', pady=4)

        #Train button
        tk.Button(parent, text='▶  TRAIN & EVALUATE',
                  command=self._run,
                  font=('Courier New', 11, 'bold'),
                  bg='#A6E3A1', fg='#1E1E2E',
                  activebackground='#94e2d5',
                  relief='flat', pady=8).pack(fill='x', padx=8, pady=10)

    def _build_output(self, parent):
        # Log 
        self.log = scrolledtext.ScrolledText(
            parent, height=7,
            font=('Courier New', 9),
            bg='#181825', fg='#A6E3A1',
            insertbackground='white', relief='flat', bd=1
        )
        self.log.pack(fill='x', padx=5, pady=(0, 5))

        # Plot frame
        self.plot_frame = tk.Frame(parent, bg='#1E1E2E')
        self.plot_frame.pack(fill='both', expand=True)

  

    def _log(self, msg: str):
        self.log.insert(tk.END, msg + '\n')
        self.log.see(tk.END)

    def _parse_params(self):
        try:
            eta = float(self.eta_var.get())
            epochs = int(self.epochs_var.get())
            mse_threshold = float(self.mse_var.get())
        except ValueError:
            messagebox.showerror('Input Error',
                                 'Please enter valid numeric values for hyperparameters.')
            return None
        if eta <= 0:
            messagebox.showerror('Input Error', 'Learning rate must be positive.')
            return None
        if epochs < 1:
            messagebox.showerror('Input Error', 'Epochs must be at least 1.')
            return None
        return eta, epochs, mse_threshold

    
    def _store_classifier(self, clf):
        self._trained_clf = clf
        self._trained_feat1 = self.feat1_var.get()
        self._trained_feat2 = self.feat2_var.get()
        self._trained_class1 = self.class1_var.get()
        self._trained_class2 = self.class2_var.get()


    def _run(self):
        params = self._parse_params()
        if params is None:
            return
        eta, epochs, mse_threshold = params

        feat1 = self.feat1_var.get()
        feat2 = self.feat2_var.get()
        class1 = self.class1_var.get()
        class2 = self.class2_var.get()
        use_bias = self.bias_var.get()
        algorithm = self.algo_var.get()

        if feat1 == feat2:
            messagebox.showerror('Input Error', 'Please select two different features.')
            return
        if class1 == class2:
            messagebox.showerror('Input Error', 'Please select two different classes.')
            return

        self.log.delete('1.0', tk.END)
        self._log(f'Algorithm : {algorithm}')
        self._log(f'Features  : {feat1}, {feat2}')
        self._log(f'Classes   : {class1} (+1) vs {class2} (-1)')
        self._log(f'eta={eta}  epochs={epochs}  mse_thresh={mse_threshold}  bias={use_bias}')
        self._log('─' * 55)

        #Split data
        X_train, y_train, X_test, y_test = train_test_split_by_class(
            self.df, class1, class2, feat1, feat2)
        X_train, X_test = normalize(X_train, X_test)

        #Train
        if algorithm == 'Perceptron':
            clf = Perceptron(eta=eta, epochs=epochs, use_bias=use_bias)
            clf.fit(X_train, y_train)
            history = None
        else:
            clf = Adaline(eta=eta, epochs=epochs,
                          mse_threshold=mse_threshold, use_bias=use_bias)
            clf.fit(X_train, y_train)
            history = None

        self._store_classifier(clf)

        #Evaluate
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[1, -1])
        acc = overall_accuracy(cm)

        self._log(f'Confusion Matrix (rows=true, cols=pred):')
        self._log(f'         {class1}(+1)  {class2}(-1)')
        self._log(f'{class1}(+1)  {cm[0][0]:5d}   {cm[0][1]:5d}')
        self._log(f'{class2}(-1)  {cm[1][0]:5d}   {cm[1][1]:5d}')
        self._log(f'Overall Accuracy: {acc * 100:.1f}%')

        #Plot
        if self.fig is not None:
            plt.close(self.fig)
        if self.canvas_widget is not None:
            self.canvas_widget.get_tk_widget().destroy()

        self.fig = build_result_figure(
            weights=clf.weights,
            bias=clf.bias,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            y_pred=y_pred,
            cm=cm,
            accuracy=acc,
            feature1=feat1, feature2=feat2,
            class1=class1, class2=class2,
            algorithm=algorithm,
            training_history=history,
            use_bias=use_bias
        )

        self.canvas_widget = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack(fill='both', expand=True)



    def _classify_sample(self):
        if not hasattr(self, '_trained_clf') or self._trained_clf is None:
            messagebox.showwarning('Not Trained', 'Please train the model first.')
            return
        try:
            x1 = float(self.sample_x1.get())
            x2 = float(self.sample_x2.get())
        except ValueError:
            messagebox.showerror('Input Error', 'Please enter valid numeric feature values.')
            return

        sample = np.array([x1, x2])
        prediction = self._trained_clf.predict_single(sample)
        label = self._trained_class1 if prediction == 1 else self._trained_class2
        self._log(f'\n🔍 Sample [{x1}, {x2}] → Predicted Class: {label} (raw={prediction})')


def main():
    app = BirdsClassifierApp()
    app.mainloop()


if __name__ == '__main__':
    main()
