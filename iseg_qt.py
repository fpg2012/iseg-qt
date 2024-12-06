from typing import List, Optional
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QFileDialog, QGraphicsEllipseItem,
    QWidget, QVBoxLayout, QPushButton, QHBoxLayout, QGraphicsRectItem, QMessageBox, QGraphicsPixmapItem
)
from PySide6.QtGui import QPixmap, QAction, QColor, QPen, QBrush, QImage
from PySide6.QtCore import Qt, QRectF, Signal, Slot, QObject, QPointF
import numpy as np
import sys
import cv2

class Predictor:
    
    def set_image(self, image: np.ndarray):
        pass
    
    def predict(self, point_coords: List[QPointF], point_labels: List[int]) -> np.ndarray:
        pass

    def reset(self):
        pass

class RandomPredictor(Predictor):

    def __init__(self):
        self.w = 0
        self.h = 0
    
    def set_image(self, image):
        h, w, _ = image.shape
        self.w = w
        self.h = h

    def predict(self, point_coords, point_labels):
        w1 = np.random.randint(0, self.w)
        h1 = np.random.randint(0, self.h)
        w2 = np.random.randint(w1, self.w)
        h2 = np.random.randint(h1, self.h)
        mask = np.zeros((self.w, self.h), np.uint8)
        mask[h1:h2, w1:w2] = 1
        return mask

class DataModel(QObject):
    # Defining the signals
    image_changed = Signal()
    clicks_changed = Signal()
    mask_changed = Signal()

    def __init__(self, predictor: Predictor):
        super().__init__()  # Initialize QObject
        self.image_path = ""  # Store the image file path
        self.click_positions = []  # List of clicked positions (as tuples of x, y)
        self.click_types = []  # List of click types (1 for left click, 0 for right click)
        self.active_mask = None
        self.predictor: Predictor = predictor
    
    def reset(self):
        self.set_mask(None)
        self.clear_clicks()
        self.predictor.reset()

    @Slot()
    def set_image(self, file_path):
        self.reset()
        self.image_path = file_path
        if self.predictor:
            image = cv2.imread(self.image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.predictor.set_image(image)
        self.image_changed.emit()  # Emit signal when the image changes

    @Slot()
    def add_click(self, position: QPointF, click_type: int):
        print(position, click_type)
        self.click_positions.append(position)
        self.click_types.append(click_type)
        self.clicks_changed.emit()  # Emit signal when clicks are updated
        if self.predictor:
            mask = self.predictor.predict(self.click_positions, self.click_types)
            self.set_mask(mask)
    
    @Slot()
    def set_mask(self, mask: np.ndarray):
        self.active_mask = mask
        self.mask_changed.emit()

    def get_image(self) -> str:
        return self.image_path

    def get_click_data(self):
        return self.click_positions, self.click_types

    @Slot()
    def undo_last_click(self):
        if self.click_positions:
            self.click_positions.pop()
            self.click_types.pop()
            self.clicks_changed.emit()  # Emit signal when clicks are updated

    @Slot()
    def clear_clicks(self):
        self.click_positions.clear()
        self.click_types.clear()
        self.clicks_changed.emit()  # Emit signal when clicks are cleared


class ImageView(QGraphicsView):
    def __init__(self, scene, model):
        super().__init__(scene)
        self.model: DataModel = model  # Reference to the model
        self.click_items = []  # List to store click circle items
        self.drag_start = None
        self.drag_rect = None
        self.active_mask = None

        # Connect signals to slots
        self.model.image_changed.connect(self.render_image)
        self.model.clicks_changed.connect(self.update_clicks)
        self.model.mask_changed.connect(self.update_mask)

    @Slot()
    def render_image(self):
        # Render the image in the scene
        if self.model.get_image():
            pixmap = QPixmap(self.model.get_image())
            self.scene().clear()  # Clear the scene completely
            self.click_items.clear()
            pixmap_item = self.scene().addPixmap(pixmap)  # Add the pixmap to the scene

            # Set the scene rect to match the pixmap dimensions
            self.scene().setSceneRect(pixmap_item.boundingRect())

            # Fit the view to the pixmap dimensions
            self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

            # Update the clicks after rendering the image
            self.update_clicks()

    @Slot()
    def update_clicks(self):
        # Remove all existing click circles but keep the image intact
        for item in self.click_items:
            self.scene().removeItem(item)
        self.click_items.clear()  # Clear the list of click circles

        # Redraw the clicks
        click_positions, click_types = self.model.get_click_data()

        # Add the new click circles
        for i, pos in enumerate(click_positions):
            color = QColor(0, 255, 0) if click_types[i] == 1 else QColor(255, 0, 0)
            self.add_click_circle(pos, color)

    @Slot()
    def update_mask(self):
        if self.model.active_mask is not None:
            if self.active_mask:
                self.scene().removeItem(self.active_mask)
            h, w = self.model.active_mask.shape
            color = np.array([30/255, 144/255, 255/255, 0.6])
            mask_image = self.model.active_mask.reshape(h, w, 1) * color.reshape(1, 1, 4)
            mask_image = (mask_image * 255).astype(np.uint8)
            image = QImage(mask_image.data, w, h, QImage.Format_RGBA8888)
            self.active_mask = QGraphicsPixmapItem(QPixmap.fromImage(image))

            self.scene().addItem(self.active_mask)
        elif self.active_mask:
            self.scene().removeItem(self.active_mask)

    def add_click_circle(self, pos, color):
        # Create a circle at the clicked position
        radius = 5
        ellipse = QGraphicsEllipseItem(
            QRectF(pos[0] - radius, pos[1] - radius, 2 * radius, 2 * radius)
        )
        ellipse.setPen(QPen(QColor(255, 255, 255), 2))  # White border
        ellipse.setBrush(QBrush(color))  # Set the circle color
        self.scene().addItem(ellipse)
        self.click_items.append(ellipse)  # Keep track of the click circles

    @Slot()
    def global_view(self):
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)

    def mousePressEvent(self, event):
        # Handle mouse press within the QGraphicsView
        if event.button() == Qt.LeftButton:
            click_type = 1  # Left click
        elif event.button() == Qt.RightButton:
            click_type = 0  # Right click
        elif event.button() == Qt.MiddleButton:
            self.drag_start = self.mapToScene(event.position().toPoint())
            return
        else:
            return

        # Map the click position to the scene coordinates
        scene_pos = self.mapToScene(event.position().toPoint())

        # Check if the click is within the image's bounding rectangle
        if not self.scene().sceneRect().contains(scene_pos):
            return  # Ignore clicks outside the image

        # Add the click data to the model
        self.model.add_click((scene_pos.x(), scene_pos.y()), click_type)

        # Call the parent implementation
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self.drag_start:
            drag_end = self.mapToScene(event.position().toPoint())
            rect = QRectF(self.drag_start, drag_end).normalized()
            if self.drag_rect:
                self.scene().removeItem(self.drag_rect)
            self.drag_rect = QGraphicsRectItem(rect)
            self.drag_rect.setPen(QPen(QColor(255, 255, 0), 3))
            self.scene().addItem(self.drag_rect)
            
        return super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self.drag_start:
            drag_end = self.mapToScene(event.position().toPoint())
            if self.drag_rect:
                self.scene().removeItem(self.drag_rect)
            rect = QRectF(self.drag_start, drag_end).normalized()
            self.fitInView(rect, Qt.KeepAspectRatio)
            self.drag_start = None
        return super().mouseReleaseEvent(event)


class SidePanel(QWidget):
    # Defining the signals
    finished = Signal()
    undid = Signal()
    cleared = Signal()
    global_viewed = Signal()

    def __init__(self):
        super().__init__()

        # Create buttons
        self.finish_button = QPushButton("Finish")
        self.undo_button = QPushButton("Undo")
        self.clear_button = QPushButton("Clear")
        self.global_view_button = QPushButton("Global View")

        # Connect buttons to corresponding actions
        self.finish_button.clicked.connect(self.finish)
        self.undo_button.clicked.connect(self.undo)
        self.clear_button.clicked.connect(self.clear)
        self.global_view_button.clicked.connect(self.global_view)

        # Create layout and add buttons
        layout = QVBoxLayout()
        layout.addWidget(self.finish_button)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.global_view_button)

        self.setLayout(layout)


    def finish(self):
        # Handle finish action (for now, just print a message)
        print("Finish clicked")
        self.finished.emit()

    def undo(self):
        # Undo last click in the model
        self.undid.emit()

    def clear(self):
        # Clear all clicks in the model
        self.cleared.emit()
    
    def global_view(self):
        self.global_viewed.emit()


class MainWindow(QMainWindow):
    def __init__(self, predictor: Predictor):
        super().__init__()
        self.setWindowTitle("Model-View Design Pattern with Signals")

        # Create the model and view
        self.model = DataModel(predictor)
        self.scene = QGraphicsScene()
        self.view = ImageView(self.scene, self.model)
        self.setCentralWidget(self.view)

        # Create the side panel and layout
        self.side_panel = SidePanel()

        # Create a horizontal layout to hold the view and the side panel
        layout = QHBoxLayout()
        layout.addWidget(self.side_panel)
        layout.addWidget(self.view)

        # Create a central widget and set the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Set up the menu bar and actions
        self.create_menu_bar()

        # Initial window size
        self.resize(800, 600)

        self.side_panel.undid.connect(self.model.undo_last_click)
        self.side_panel.cleared.connect(self.model.clear_clicks)
        self.side_panel.global_viewed.connect(self.view.global_view)

    def create_menu_bar(self):
        # Create a menu bar
        menu_bar = self.menuBar()

        # Add a File menu
        file_menu = menu_bar.addMenu("File")
        about_menu = menu_bar.addMenu("Help")

        # Add 'Load Image' action
        load_image_action = QAction("Load Image", self)
        load_image_action.triggered.connect(self.load_image)
        file_menu.addAction(load_image_action)

        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about_dialog)
        about_menu.addAction(about_action)

    def load_image(self):
        # Open a file dialog to choose an image
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
        )
        if file_path:
            # Update the model with the new image path
            self.model.set_image(file_path)
    
    def show_about_dialog(self):
        QMessageBox.about(self, "About", "Image Viewer with Clicks\nVersion 1.0")

def run_application(predictor: Predictor = None):
    app = QApplication(sys.argv)
    window = MainWindow(predictor=predictor)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    predictor = RandomPredictor()
    run_application(predictor=predictor)
