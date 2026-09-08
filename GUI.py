import sys
import os
import PySide6
import torch
import torch.nn.functional as F
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QProgressBar,
    QMessageBox,
    QFrame,
)
from PIL import Image
from PIL.ImageQt import ImageQt
import cv2

from data_handler import ToTensor
from Model_architecture import ResNet50_ViT
import numpy as np
import tifffile as tiff

# device check
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# choose models
MODEL_CONFIGS = {

    "COS-7 widefield": {

        "model_path": "Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/Code_for_publication/trained_models/COS7_widefield.pt",

        "architecture": ResNet50_ViT,

        "architecture_kwargs": {
            "img_size": 224,
            "patch_size": 4,
            "embed_dim": 768,
            "num_heads": 4,
            "depth": 3,
            "num_classes": 4,
        },

        "class_names": [
            "Control",
            "FCCP",
            "H2O2",
            "Heat Shock",
        ],

        "class_colors": {
                'Control':   '#B8B8B8',  # light gray
                'FCCP':      '#6F8FBF',  # soft blue
                'H2O2':      '#8C79B8',  # soft purple
                'Heat Shock':'#F0C060',  # yellow
            },
    },

    "COS-7 Confocal": {
    
            "model_path": "Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/Code_for_publication/trained_models/COS7_Confocal.pt",
    
            "architecture": ResNet50_ViT,
    
            "architecture_kwargs": {
                "img_size": 224,
                "patch_size": 4,
                "embed_dim": 768,
                "num_heads": 4,
                "depth": 3,
                "num_classes": 4,
            },
    
            "class_names": [
                "Control",
                "FCCP",
                "H2O2",
                "Heat Shock",
            ],

            "class_colors": {
                'Control':   '#B8B8B8',  # light gray
                'FCCP':      '#6F8FBF',  # soft blue
                'H2O2':      '#8C79B8',  # soft purple
                'Heat Shock':'#F0C060',  # yellow
            },
        },


    "COS-7 SIM": {
    
            "model_path": "Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/Code_for_publication/trained_models/COS7_SIM.pt",
    
            "architecture": ResNet50_ViT,
    
            "architecture_kwargs": {
                "img_size": 224,
                "patch_size": 4,
                "embed_dim": 768,
                "num_heads": 4,
                "depth": 3,
                "num_classes": 4,
            },
    
            "class_names": [
                "Control",
                "FCCP",
                "H2O2",
                "Heat Shock",
            ],

            "class_colors": {
                'Control':   '#B8B8B8',  # light gray
                'FCCP':      '#6F8FBF',  # soft blue
                'H2O2':      '#8C79B8',  # soft purple
                'Heat Shock':'#F0C060',  # yellow
            },

        },
    

    "SH-SY5Y": {

        "model_path": "Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/Code_for_publication/trained_models/SH_4classes.pt",

        "architecture": ResNet50_ViT,

        "architecture_kwargs": {
            "img_size": 224,
            "patch_size": 4,
            "embed_dim": 768,
            "num_heads": 4,
            "depth": 3,
            "num_classes": 4,
        },

        "class_names": [
            "Control",
            "FCCP",
            "H2O2",
            "Rotenone",
        ],
        "class_colors": {
            "Control": "#B8B8B8",
            "FCCP": "#6F8FBF",
            "H2O2": "#8C79B8",
            "Rotenone": "#6FAF9F"
        },
    },

"SH-SY5Y + non-mitochondrial stress (6 classes)": {

        "model_path": "Z:/Users/yj368/ROI isentification/Health_state_prediction/organised training code/work for mito and lyso identification task/Code_for_publication/trained_models/SH_6classes.pt",

                "architecture": ResNet50_ViT,
        
                "architecture_kwargs": {
                    "img_size": 224,
                    "patch_size": 4,
                    "embed_dim": 768,
                    "num_heads": 4,
                    "depth": 3,
                    "num_classes": 6,
                },
        
                "class_names": [
                    "Control",
                    "FCCP",
                    "H2O2",
                    'Rotenone', 
                    'Tunicamycin', 
                    'Thapsigargin'
                ],

                "class_colors": {
                    "Control": "#B8B8B8",
                    "FCCP": "#6F8FBF",
                    "H2O2": "#8C79B8",
                    "Rotenone": "#6FAF9F",
                    "Tunicamycin": "#F0A0C0",
                    "Thapsigargin": "#F0C080"
                },
            },


    "Customised Model": {

        "model_path": "models/Customised Model.pt",

        "architecture": ResNet50_ViT,

        "architecture_kwargs": {
            "img_size": 224,
            "patch_size": 4,
            "embed_dim": 768,
            "num_heads": 4,
            "depth": 3,
            "num_classes": 4,
        },

        "class_names": [
            "Class 1",
            "Class 2",
            "Class 3",
            "Class 4"
        ],
    },
}


# data handler for model input
def preprocess_image(image_path):
    transform = ToTensor()

    #give a random label, since the transform expects a tuple of (image_path, label)
    image, _ = transform((image_path, 0))

    return image

# display image in the GUI
def prepare_display_image(image_path):
    img = tiff.imread(image_path)

    if len(img.shape) == 2:    # this setting is based on my won image acquisition way, change it if needed for the user's image data
        img = img.astype(np.float32)

    else:
        img = img[1].astype(np.float32)

    # contrast adjustment
    low = np.percentile(img,1)
    high = np.percentile(img,99)

    if high > low:
        img = ((img - low)/ (high - low))

    img = np.clip(img,0,1)
    img = (img * 255).astype(np.uint8)

    img = cv2.applyColorMap(img,cv2.COLORMAP_HOT)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    return img

def numpy_to_qpixmap(img):
    # for RGB images, just for display. The model input image is 2D mitochondrial channel
    if len(img.shape) == 3:
        height, width, channels = img.shape
        qimage = QImage(
            img.data,
            width,
            height,
            width * 3,
            QImage.Format_RGB888
        )

    else:
        # grayscale 
        height, width = img.shape
        qimage = QImage(
            img.data,
            width,
            height,
            width,
            QImage.Format_Grayscale8
        )

    pixmap = QPixmap.fromImage(qimage.copy())

    return pixmap


# load model configuration and weights
def load_model(config):
    architecture = config["architecture"]
    kwargs = config["architecture_kwargs"]
    model = architecture(**kwargs)

    checkpoint = torch.load(config["model_path"],map_location=DEVICE)
    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model

# generate prediction for a single image
def predict(model,image_path,class_names):
    image = preprocess_image(image_path)

    image = image.unsqueeze(0)
    image = image.to(DEVICE)

    with torch.no_grad():
        output = model(image)
        probabilities = F.softmax(output,dim=1)

    probabilities = (probabilities[0].cpu())

    predicted_index = torch.argmax(probabilities).item()

    predicted_class = (class_names[predicted_index])

    confidence = (probabilities[predicted_index].item())

    return (predicted_class,confidence,probabilities)


# interface to drag an image into the GUI
class DropArea(QGroupBox):
    """
    This is just used to display the image for user visualisation of mitochondria structure.

    The DropArea only:
    1. Accepts .tif / .tiff files
    2. Displays the selected image
    3. Gives the image path to the main GUI for prediction

    Image preprocessing and model prediction are handled separately.
    """

    imageDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(450,400)
        self.image_label = QLabel("Drop image")
        self.setAlignment(Qt.AlignCenter)

        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #999999;
                border-radius: 12px;
                background: #fafafa;
                color: #666666;
                font-size: 18px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.image_label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path.lower().endswith((".tif", ".tiff")): # accept only .tif and .tiff files
                    event.acceptProposedAction()
                    return

        event.ignore()

    def dropEvent(self, event):
        if not event.mimeData().hasUrls():
            event.ignore()
            return

        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith((".tif", ".tiff")):
                print("Dropped image:",path)
                self.imageDropped.emit(path)
                event.acceptProposedAction()
                return

        event.ignore()

    def set_image(self, pixmap):
        if pixmap.isNull():
            return
        
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def clear_image(self):
        self.image_label.clear()
        self.image_label.setText("Drop image")

    def resizeEvent(self, event):
        super().resizeEvent(event)

# show the prediction results in the GUI
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mitochondrial Stress Predictor")

        self.resize(1200,800)
        self.current_image = None
        self.current_model = None
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        title = QLabel("MitoDecode: Mitochondrial Stress Profiler")
        title.setFont(QFont("Arial",24,QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        subtitle = QLabel("Deep-learning based mitochondrial phenotyping")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #666666;")
        main_layout.addWidget(subtitle)

        # model selection
        model_group = QGroupBox("Model")
        model_layout = QGridLayout(model_group)
        model_layout.addWidget(QLabel("Model:"),0,0)

        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_CONFIGS.keys())
        self.model_combo.currentIndexChanged.connect(self.model_changed)
        model_layout.addWidget(self.model_combo,0,1)

        self.class_info = QLabel()
        self.class_info.setWordWrap(True)
        model_layout.addWidget(self.class_info,1,0,1,2)

        main_layout.addWidget(model_group)
        content = QHBoxLayout()
        image_group = QGroupBox("Input Image")
        image_layout = QVBoxLayout(image_group)

        self.drop_area = DropArea()
        self.drop_area.imageDropped.connect(self.image_selected)
        image_layout.addWidget(self.drop_area)

        open_button = QPushButton("Open Image")
        open_button.setMinimumHeight(40)
        open_button.clicked.connect(self.open_image)
        image_layout.addWidget(open_button)

        content.addWidget(image_group)

        result_group = QGroupBox("Phenotypic profiling")
        result_layout = QVBoxLayout(result_group)

        # Section title
        phenotype_title = QLabel("Predicted phenotype")
        phenotype_title.setAlignment(Qt.AlignCenter)
        phenotype_title.setFont(QFont("Arial", 16, QFont.Bold))
        phenotype_title.setStyleSheet("""
            QLabel {
                color: #4A5A61;
                padding-top: 2px;
            }
        """)
        result_layout.addWidget(phenotype_title)

        # Predicted class
        self.prediction = QLabel("No prediction")
        self.prediction.setAlignment(Qt.AlignCenter)
        self.prediction.setFont(QFont("Arial", 16, QFont.Bold))
        self.prediction.setMinimumHeight(50)

        self.prediction.setStyleSheet("""
            QLabel {
                border: none;
                border-radius: 10px;
                padding: 6px;
                color: #4A5A61;
            }
        """)
        result_layout.addWidget(self.prediction)
        

        self.confidence = QLabel("Model score: --")
        self.confidence.setAlignment(Qt.AlignCenter)
        self.confidence.setFont(QFont("Arial",12))
        result_layout.addWidget(self.confidence)

        probability_group = QGroupBox("Phenotypic probability distribution")

        probability_layout = QVBoxLayout(probability_group)

        self.probability_rows = []

        for _ in range(6):
            row = QHBoxLayout()
            name = QLabel()
            name.setMinimumWidth(110)
            name.setFont(QFont("Arial", 12))

            bar = QProgressBar()
            bar.setRange(0,100) # probability
            bar.setTextVisible(False)
            bar.setMinimumHeight(20)

            value = QLabel()
            value.setMinimumWidth(65)
            value.setFont(QFont("Arial", 12))

            row.addWidget(name)
            row.addWidget(bar)
            row.addWidget(value)

            probability_layout.addLayout(row)
            self.probability_rows.append((name,bar,value))

        result_layout.addWidget(probability_group)
        content.addWidget(result_group)
        main_layout.addLayout(content)

        # click run
        self.run_button = QPushButton("RUN PREDICTION")
        self.run_button.setMinimumHeight(55)
        self.run_button.setFont(QFont("Arial",14,QFont.Bold))
        self.run_button.clicked.connect(self.run_prediction)
        main_layout.addWidget(self.run_button)

        self.status = QLabel(f"Device: {DEVICE}")
        self.status.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status)

        # initialise the model and class info
        self.model_changed(0)

    def model_changed(self,index):
        model_name = (self.model_combo.currentText())
        config = MODEL_CONFIGS[model_name]
        class_names = config["class_names"]

        self.class_info.setText(
            f"<b>Number of classes:</b> "
            f"{len(class_names)}<br>"
            f"<b>Classes:</b> "
            f"{', '.join(class_names)}"
        )

        self.current_model = None
        self.reset_results()
        self.status.setText(f"Selected model: {model_name}")

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self,"Open TIFF Image","","TIFF Images (*.tif *.tiff)")
        if not path:
            return

        self.image_selected(path)

    def image_selected(self,path):
        try:
            if not path.lower().endswith((".tif", ".tiff")):
                QMessageBox.warning(
                    self,
                    "Invalid image",
                    "Please select a TIFF image."
                )
                return
            
            # Preview
            display_img = prepare_display_image(path)
            pixmap = numpy_to_qpixmap(display_img)

            pixmap = pixmap.scaled(
                450,
                400,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )

            self.drop_area.set_image(pixmap)

            self.current_image = path
            self.status.setText("Image loaded: "+ os.path.basename(path))
            # self.reset_results()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Could not open image:\n\n{e}"
            )

    def set_probability_bar_style(self, bar, class_name, class_colors):
        color = class_colors.get(class_name, "#808080")

        bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #E9EDF0;
            }}

            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

    def run_prediction(self):
        if self.current_image is None:
            QMessageBox.warning(
                self,
                "No image",
                "Please select an image first."
            )

            return

        model_name = (self.model_combo.currentText())
        config = MODEL_CONFIGS[model_name]
        self.run_button.setEnabled(False)
        self.status.setText("Loading model...")

        QApplication.processEvents()

        try:
            if self.current_model is None:
                self.current_model = load_model(config)

            self.status.setText("Running prediction...")

            QApplication.processEvents()
            (predicted_class,confidence,probabilities) = predict(self.current_model,self.current_image,config["class_names"])

            self.prediction.setText(predicted_class)

            self.confidence.setText(
                f"Severity score: "
                f"{confidence * 100:.2f}%"
            )

            class_names = config["class_names"]
            class_colors = config.get("class_colors",{})

            for i in range(len(self.probability_rows)):
                name, bar, value = (self.probability_rows[i])

                if i < len(class_names):
                    probability = (probabilities[i].item())

                    name.setText(class_names[i])

                    bar.setValue(int(probability * 100))
                    self.set_probability_bar_style(bar,class_names[i],class_colors)

                    value.setText(f"{probability * 100:.2f}%")

                else:
                    name.setText("")
                    bar.setValue(0)
                    value.setText("")

            self.status.setText("Prediction completed")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Prediction Error",
                str(e)
            )

            self.status.setText("Prediction failed")

        finally:
            self.run_button.setEnabled(True)

    # reset
    def reset_results(self):
        self.prediction.setText("No prediction")
        self.confidence.setText("Model score: --")

        for name, bar, value in (self.probability_rows):
            name.setText("")
            bar.setValue(0)
            value.setText("")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    font = app.font()
    font.setPointSize(12)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
