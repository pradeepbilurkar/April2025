import sys
import random
import numpy as np
import json
import openai
from shapely.geometry import box, MultiPoint
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,QScrollArea,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGroupBox, QFormLayout,QMessageBox, QCheckBox

)
from PyQt5.QtWidgets import QComboBox
from PyQt5.QtCore import Qt, QObject
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib.pyplot as plt
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtCore import Qt, QSize, QThreadPool, QRunnable, pyqtSlot, pyqtSignal, QObject
from PyQt5.QtGui import QMovie, QPixmap, QFont , QIcon

from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from geometry_utils import compute_drive_lane
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import Polygon
from datetime import datetime
import matplotlib
from shapely.geometry import Polygon
import fitz  # PyMuPDF
import re
from PyQt5.QtWidgets import QFileDialog




matplotlib.rcParams['font.family'] = 'Segoe UI Emoji'  # or 'Noto Color Emoji' on Linux

class WorkerSignals(QObject):
    finished = pyqtSignal(dict)  # Carries optimization results to GUI

# ── Layout Computation Worker ──
class LayoutWorker(QRunnable):
    def __init__(self, params, callback):
        super().__init__()
        self.params = params
        self.signals = WorkerSignals()
        self.signals.finished.connect(callback)



# 🧠 Replace with your actual API key

class OptimizerWorker(QThread):
    finished = pyqtSignal(object)
    def __init__(self, parent, config, trucks):
        super().__init__()
        self.parent = parent
        self.config = config
        self.trucks = trucks
    def run(self):
        result = self.parent._compute_optimization(self.config, self.trucks)
        self.finished.emit(result)


class AgenticOptimizerGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.trucks = []
        self.config = {}

        self.setWindowTitle("🚚 SAMSAN Layout Optimizer")
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("C:/PythonProject/logo6.png"))

        # ── Global Styling ──
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f9ff;
                font-family: 'Segoe UI';
                font-size: 10pt;
            }
            QLabel {
                color: #333;
            }
            QLineEdit, QTextEdit {
                background-color: #ffffff;
                border: 1px solid #ccd6e0;
                padding: 4px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 6px 10px;          /* Add breathing room */
                margin-top: 2px;            /* Prevent vertical clipping */
                color: #005FA1;
                font-weight: bold;
                font-size: 11pt;
            }

        """)

        # ── Logo & Title ──
        header_layout = QHBoxLayout()
        logo_label = QLabel()
        try:
            logo_pixmap = QPixmap("C:/PythonProject/logo6.png").scaled(50, 50, Qt.KeepAspectRatio,
                                                                       Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        except:
            logo_label.setText("🧠")
        title_label = QLabel("SAMSAN Layout Optimizer")
        title_label.setFont(QFont("Segoe UI", 16, QFont.Bold))
        header_layout.addWidget(logo_label)
        header_layout.addWidget(title_label)
        header_layout.addStretch()

        # ── Layout Parameters ──
        form_layout = QFormLayout()
        self.warehouse_width_input = QLineEdit("50")
        self.warehouse_length_input = QLineEdit("50")
        self.truck_count_input = QLineEdit("10")
        self.clearance_input = QLineEdit("2.0")
        self.shape_points_input = QLineEdit("8")

        form_layout.addRow("Warehouse Width (m)", self.warehouse_width_input)
        form_layout.addRow("Warehouse Length (m)", self.warehouse_length_input)
        form_layout.addRow("Truck Count", self.truck_count_input)
        form_layout.addRow("Clearance (m)", self.clearance_input)
        form_layout.addRow("Shape Complexity (Points)", self.shape_points_input)

        self.optimize_mode = QComboBox()
        self.optimize_mode.addItems(["Maximize Truck Count", "Minimize Warehouse Area"])
        form_layout.addRow("Optimization Goal", self.optimize_mode)

        form_group = QGroupBox("📥 Layout Parameters")
        form_group.setLayout(form_layout)

        # ── PDF Import Group ──
        self.load_from_pdf_checkbox = QCheckBox("📄 Use Polygon from PDF")
        self.upload_pdf_btn = QPushButton("📄 Upload Layout PDF")
        self.upload_pdf_btn.clicked.connect(self.handle_pdf_upload)

        pdf_tools_layout = QVBoxLayout()
        pdf_tools_layout.addWidget(self.load_from_pdf_checkbox)
        pdf_tools_layout.addWidget(self.upload_pdf_btn)

        pdf_tools_group = QGroupBox("📐 Import Layout")
        pdf_tools_group.setLayout(pdf_tools_layout)
        pdf_tools_group.setStyleSheet("QGroupBox { font-weight: bold; font-size: 10.5pt; color: #005FA1; }")

        # ── Optimization Controls ──
        self.density_toggle = QCheckBox("Show Truck Density")

        self.run_button = QPushButton("🧠 Run Optimization")
        self.run_button.setStyleSheet("""
            QPushButton {
                background-color: #0078D7;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005FA1;
            }
        """)
        self.run_button.clicked.connect(self.start_optimization_thread)

        # ── Natural Language UI ──
        #self.nl_query_input = QLineEdit()
        #self.nl_query_input.setPlaceholderText("💬 Describe your layout goal in natural language...")

        self.nl_query_input = QTextEdit()
        self.nl_query_input.setFixedHeight(60)  # Optional: limit vertical size
        #self.nl_query_input.setPlaceholderText("💬 Describe your layout goal in natural language...")

        self.nl_query_input.setText(
            "Maximize truck placement inside warehouse polygon, ensuring clearance and no overlaps.")
        self.query_button = QPushButton("🎯 Run via AI")
        self.query_button.clicked.connect(self.process_nl_query)

        # ── Spinner ──
        self.spinner = QLabel()
        self.spinner.setAlignment(Qt.AlignCenter)
        self.spinner_movie = QMovie("C:/PythonProject/spinner-3.gif")
        self.spinner.setMovie(self.spinner_movie)
        self.spinner_movie.setSpeed(100)
        self.spinner.setVisible(False)

        # ── Agent Summary ──
        self.agent_output_display = QTextEdit()
        self.agent_output_display.setReadOnly(True)
        self.agent_output_display.setFont(QFont("Consolas", 9))
        self.agent_output_display.setFixedHeight(180)
        self.agent_output_display.setText("📋 Agent strategy summary will appear here...")

        summary_label = QLabel("📊 Summary Output")
        summary_label.setStyleSheet("QLabel { color: #0078D7; font-weight: bold; font-size: 11pt; }")

        # ── RHS Layout ──
        rhs_layout = QVBoxLayout()
        rhs_layout.addLayout(header_layout)
        rhs_layout.addWidget(form_group)
        rhs_layout.addWidget(pdf_tools_group)
        rhs_layout.addWidget(self.run_button)
        rhs_layout.addWidget(self.spinner)
        rhs_layout.addWidget(self.density_toggle)
        rhs_layout.addWidget(self.nl_query_input)
        rhs_layout.addWidget(self.query_button)
        rhs_layout.addWidget(summary_label)
        rhs_layout.addWidget(self.agent_output_display)
        rhs_layout.addStretch()

        rhs_widget = QWidget()
        rhs_widget.setLayout(rhs_layout)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(rhs_widget)

        # ── Plot Canvas ──
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = Canvas(self.figure)

        # ── Narration ──
        self.narration_box = QTextEdit()
        self.narration_box.setReadOnly(True)
        self.narration_box.setLineWrapMode(QTextEdit.WidgetWidth)
        self.narration_box.setFixedHeight(140)
        self.narration_box.setStyleSheet("background-color: #fdfdfd; font-family: Consolas; font-size: 12px;")

        narration_group = QGroupBox("📣 Layout Feedback")
        narration_layout = QVBoxLayout()
        narration_layout.addWidget(self.narration_box)
        narration_group.setLayout(narration_layout)

        # ── LHS Layout ──
        lhs_layout = QVBoxLayout()
        lhs_layout.addWidget(self.canvas)
        lhs_layout.addWidget(narration_group)
        lhs_layout.addStretch()

        # ── Main Layout ──
        main_layout = QHBoxLayout()
        main_layout.addLayout(lhs_layout)
        main_layout.addWidget(scroll_area)
        self.setLayout(main_layout)

#######################################################################
    def handle_pdf_upload(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF Layout", "", "PDF Files (*.pdf)")
        if file_path:
            polygon = self.load_polygon_from_pdf(file_path)
            if polygon:
                self.loaded_pdf_polygon = polygon
                QMessageBox.information(self, "PDF Loaded", f"✅ Polygon extracted from:\n{file_path}")
            else:
                QMessageBox.warning(self, "PDF Error", "❌ Failed to extract polygon from selected PDF.")



    def load_polygon_from_pdf(self,pdf_path):
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text() for page in doc)

        # Match side dimensions: "15.00 m"
        matches = re.findall(r"(\d+\.\d+)\s*m", text)
        if len(matches) >= 4:
            try:
                sides = [float(m) for m in matches[:4]]
                # Reconstruct rectangle for now — refine later for irregular shapes
                coords = [(0, 0), (sides[0], 0), (sides[0], sides[1]), (0, sides[1])]
                return Polygon(coords)
            except Exception as e:
                print("❌ Failed to parse dimensions:", e)
        else:
            print("⚠️ Insufficient side data found in PDF text.")

        return None

    def save_polygon_to_pdf(self, polygon, filename="optimized_layout.pdf"):
        fig, ax = plt.subplots(figsize=(8, 8))
        x, y = polygon.exterior.xy
        ax.plot(x, y, color='blue', linewidth=2)
        ax.fill(x, y, alpha=0.05, color='lightblue', label='Polygon Area')

        # Set aspect ratio and grid
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', linewidth=0.4, color='gray')
        ax.set_axisbelow(True)

        # Annotate side dimensions with arrows and labels
        coords = list(polygon.exterior.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            # Side length
            dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            # Midpoint for label
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

            # Arrow annotation
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle='<->', mutation_scale=10,
                                    color='black', linewidth=0.8)
            ax.add_patch(arrow)

            # Label
            ax.text(mid_x, mid_y, f"{dist:.2f} m",
                    fontsize=8, fontfamily='monospace',
                    color='black', ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.7))

        # Title block metadata
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
        ax.text(0, -10, f"Layout: Optimized Polygon\nDate: {timestamp}\nUnits: meters",
                fontsize=8, fontfamily='monospace', va='top',
                bbox=dict(boxstyle='round', fc='lightgrey', alpha=0.5))

        # Save as high-resolution PDF
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, format='pdf', dpi=300)
        plt.close()
        print(f"✅ Polygon layout exported to PDF: {filename}")

    def process_nl_query(self):
        query = self.nl_query_input.toPlainText().strip()

        if not query:
            return

        from nl_parser import interpret_layout_query
        api_key = openai.api_key

        params = interpret_layout_query(query, api_key)
        if not params:
            self.agent_output_display.setText("⚠️ Failed to interpret layout query.")
            return

        # Apply extracted parameters
        self.truck_count_input.setText(str(params.get("truck_count", 10)))
        self.clearance_input.setText(str(params.get("clearance", 2.0)))
        self.optimize_mode.setCurrentText(
            "Maximize Truck Count" if params.get("optimize_goal") == "maximize_trucks" else "Minimize Warehouse Area"
        )

        # You may optionally use lane_width in compute_drive_lane() or render hint
        self.agent_output_display.setText(f"🎯 AI interpreted config:\n{json.dumps(params, indent=2)}")

        if self.load_from_pdf_checkbox.isChecked() and hasattr(self, "loaded_pdf_polygon"):
            polygon = self.loaded_pdf_polygon
            self.config["polygon"] = polygon
            self.config["warehouse_width"] = polygon.bounds[2] - polygon.bounds[0]
            self.config["warehouse_length"] = polygon.bounds[3] - polygon.bounds[1]
            self.config["source"] = "pdf_layout"

        # Fire optimization!
        self.start_optimization_thread()



    def process_nl_query1(self):
        user_prompt = self.nl_query_input.text()
        if not user_prompt.strip():
            return

        from nl_parser import interpret_layout_query
        api_key = openai.api_key

        result_json = interpret_layout_query(user_prompt, api_key)
        try:
            params = json.loads(result_json)
        except:
            self.agent_output_display.setText("⚠️ AI response parsing failed.")
            return

        # Apply values to form inputs
        self.truck_count_input.setText(str(params.get("truck_count", 10)))
        self.clearance_input.setText(str(params.get("clearance", 2.0)))
        self.optimize_mode.setCurrentText(
            "Maximize Truck Count" if params.get("optimize_goal") == "maximize_trucks" else "Minimize Warehouse Area")

        # Kick off optimization!
        self.start_optimization_thread()

    def start_optimization_thread(self):
        self.spinner.setVisible(True)
        self.spinner_movie.start()
        try:
            config = {
                "warehouse_width": float(self.warehouse_width_input.text()),
                "warehouse_length": float(self.warehouse_length_input.text()),
                "truck_count": int(self.truck_count_input.text()),
                "clearance": float(self.clearance_input.text()),
                "shape_points": int(self.shape_points_input.text()),
                "goal": self.optimize_mode.currentText()
            }
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid numbers.")
            self.spinner_movie.stop()
            self.spinner.setVisible(False)
            return

        if hasattr(self, "loaded_pdf_polygon"):
            config["polygon"] = self.loaded_pdf_polygon
            config["warehouse_width"] = self.loaded_pdf_polygon.bounds[2] - self.loaded_pdf_polygon.bounds[0]
            config["warehouse_length"] = self.loaded_pdf_polygon.bounds[3] - self.loaded_pdf_polygon.bounds[1]
            config["source"] = "pdf_layout"

        self.worker = OptimizerWorker(self, config, self.trucks)
        self.worker.finished.connect(self._complete_optimization)
        self.worker.start()

    def _compute_optimization(self, cfg, trucks):
        res = 1.0  # Resolution in meters
        heatmap_grid = np.zeros((int(cfg["warehouse_length"] / res), int(cfg["warehouse_width"] / res)))
        density_grid = np.zeros((int(cfg["warehouse_length"] / res), int(cfg["warehouse_width"] / res)))


        warehouse_width = cfg["warehouse_width"]
        warehouse_length = cfg["warehouse_length"]
        truck_count = cfg["truck_count"]
        clearance = cfg["clearance"]
        shape_points = cfg["shape_points"]
        optimization_goal = cfg["goal"]

        TRUCK_TYPES = [
            {"type": "Tata Ace", "length": 8.5, "width": 5.5, "priority": 3},
            {"type": "Tata 407", "length": 15.0, "width": 6.5, "priority": 2},
            {"type": "Container 32ft", "length": 32.0, "width": 8.0, "priority": 1}
        ]

        # GPT strategy (optional)
        try:
            prompt = f"""You are a logistics expert optimizing a warehouse of {warehouse_width}m x {warehouse_length}m.
    It must accommodate {truck_count} trucks with {clearance}m clearance each.
    Suggest a smart truck placement strategy optimizing maneuver zones and dock accessibility."""
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150
            )
            strategy = response.choices[0].message["content"].strip()
        except Exception:
            strategy = "Fallback: corner placement with maneuver priority"

        placed, outline, polygon = [], [], None
        area = 0
        start_size = int(max(warehouse_width, warehouse_length) * 1.5)

        if optimization_goal == "Minimize Warehouse Area":
            success, final_layout = False, None

            for size in range(start_size, 30, -2):
                shape = self.generate_irregular_shape(shape_points, size, size)
                if not shape or "polygon" not in shape or "outline" not in shape:
                    continue  # or break, depending on your retry strategy
                outline = shape["outline"]
                polygon = shape["polygon"]

                try:
                    entry_x = polygon.centroid.x
                    bottom_y = polygon.bounds[1]
                    top_y = polygon.bounds[3]

                    try:
                        lane_width = max(truck["width"] for truck in trucks)
                    except:
                        lane_width = 1.5

                    drive_lane = box(entry_x - lane_width / 2, bottom_y, entry_x + lane_width / 2, top_y)

                    # Dynamically size lane width from truck data
                    #lane_width = max(truck["width"] for truck in trucks)
                    drive_lane = box(entry_x - lane_width / 2, bottom_y,
                                     entry_x + lane_width / 2, top_y)
                except Exception as e:
                    print("⚠️ Could not create drive lane:", e)
                    drive_lane = None


                #trucks = [{**random.choice(TRUCK_TYPES), "id": f"T{i + 1}"} for i in range(truck_count)]

                ##################################################################
                # 👇 Accept user-defined truck type from cfg
                requested_type = cfg.get("truck_type", "mixed")  # fallback if not provided

                if requested_type == "mixed":
                    trucks = [{**random.choice(TRUCK_TYPES), "id": f"T{i + 1}"} for i in range(truck_count)]
                else:
                    matching_types = [t for t in TRUCK_TYPES if requested_type.lower() in t["type"].lower()]
                    if not matching_types:
                        print(f"⚠️ Unknown truck type: {requested_type}. Using mixed fallback.")
                        trucks = [{**random.choice(TRUCK_TYPES), "id": f"T{i + 1}"} for i in range(truck_count)]
                    else:
                        trucks = [{**matching_types[0], "id": f"T{i + 1}"} for i in range(truck_count)]


                ###############################################################





                placed, occupied = [], []# 📍 Setup protected entry lane
                entry_x = size / 2  # or dynamically computed via candidate scan
                bottom_y = min(p[1] for p in outline)
                entry_lane = box(entry_x - 0.75, bottom_y, entry_x + 0.75, size)


                for truck in sorted(trucks, key=lambda x: x["priority"]):
                    maneuver = truck["length"] * 0.6
                    placed_flag = False
                    for x in np.arange(0, size, 0.5):
                        for y in np.arange(0, size, 0.5):
                            full_box = box(x, y, x + truck["width"] + clearance,
                                           y + truck["length"] + clearance + maneuver)
                            # 🚧 Skip locations that intersect with reserved entry lane

                            from geometry_utils import compute_drive_lane

                            drive_lane = compute_drive_lane(polygon, trucks)
                            self.drive_lane = drive_lane  # ✅ Make it available globally

                            if drive_lane and drive_lane.intersects(full_box):
                                continue  # 🚫 Skip — would block the central truck flow lane

                            if entry_lane.intersects(full_box):
                                continue
                            if polygon.contains(full_box) and not any(full_box.intersects(p) for p in occupied):
                                placed.append((truck["id"], truck["type"], truck["width"], truck["length"],
                                               x + clearance / 2, y + clearance / 2))

                                occupied.append(full_box)
                                x_start, y_start = int(y / res), int(x / res)
                                x_end = int((y + truck["length"]) / res)
                                y_end = int((x + truck["width"]) / res)

                                for i in range(x_start, x_end):
                                    for j in range(y_start, y_end):
                                        if 0 <= i < density_grid.shape[0] and 0 <= j < density_grid.shape[1]:
                                            density_grid[i, j] += 1

                                placed_flag = True
                                break
                            else:
                                ix, iy = int(y / res), int(x / res)
                                if 0 <= ix < heatmap_grid.shape[0] and 0 <= iy < heatmap_grid.shape[1]:
                                    heatmap_grid[ix, iy] += 1

                        if placed_flag:
                            break
                if len(placed) == truck_count:
                    warehouse_width = warehouse_length = size
                    final_layout = (outline, polygon, placed)
                    success = True
                else:
                    break
            if not success:
                return {"error": "Unable to place all trucks in minimized area."}
            outline, polygon, placed = final_layout
        else:
            for _ in range(5):
                shape = self.generate_irregular_shape(shape_points, warehouse_width, warehouse_length)
                entry_x = warehouse_width / 2

                if not shape or "polygon" not in shape or shape["polygon"] is None:
                    print("❌ No valid shape returned — skipping placement attempt.")
                    return {"error": "Failed to generate warehouse shape."}
                polygon = shape["polygon"]
                outline = shape["outline"]

                try:
                    entry_x = polygon.centroid.x
                    bottom_y = polygon.bounds[1]
                    top_y = polygon.bounds[3]

                    # Dynamically size lane width from truck data
                    #lane_width = max(truck["width"] for truck in trucks)

                    try:
                        lane_width = max(truck["width"] for truck in trucks)
                    except:
                        lane_width = 1.5

                    drive_lane = box(entry_x - lane_width / 2, bottom_y, entry_x + lane_width / 2, top_y)

                    #drive_lane = box(entry_x - lane_width / 2, bottom_y,
                                    # entry_x + lane_width / 2, top_y)
                except Exception as e:
                    print("⚠️ Could not create drive lane:", e)
                    drive_lane = None

                bottom_y = min(p[1] for p in outline)
                entry_lane = box(entry_x - 0.75, bottom_y, entry_x + 0.75, warehouse_length)


                trucks = [{**random.choice(TRUCK_TYPES), "id": f"T{i + 1}"} for i in range(truck_count)]
                placed, occupied = [], []
                for truck in sorted(trucks, key=lambda x: x["priority"]):
                    maneuver = truck["length"] * 0.6
                    placed_flag = False
                    for x in np.arange(0, warehouse_width, 0.5):
                        for y in np.arange(0, warehouse_length, 0.5):
                            full_box = box(x, y, x + truck["width"] + clearance,
                                           y + truck["length"] + clearance + maneuver)

                            from geometry_utils import compute_drive_lane
                            drive_lane = compute_drive_lane(polygon, trucks)
                            self.drive_lane = drive_lane  # ✅ Make it available globally

                            if drive_lane and drive_lane.intersects(full_box):
                               # print(f"⛔ Truck skipped due to flow lane conflict: {tx}, {ty}, size {w}x{l}")
                                continue

                            if entry_lane.intersects(full_box):
                                continue  # 🛑 Skip location to preserve entry access
                            if polygon.contains(full_box) and not any(full_box.intersects(p) for p in occupied):
                                placed.append((truck["id"], truck["type"], truck["width"], truck["length"],
                                               x + clearance / 2, y + clearance / 2))
                                occupied.append(full_box)
                                ix, iy = int(y / res), int(x / res)
                                if 0 <= ix < density_grid.shape[0] and 0 <= iy < density_grid.shape[1]:
                                    density_grid[ix, iy] += 1

                                placed_flag = True
                                break
                            else:
                                x_start, y_start = int(y / res), int(x / res)
                                x_end = int((y + truck["length"]) / res)
                                y_end = int((x + truck["width"]) / res)

                                for i in range(x_start, x_end):
                                    for j in range(y_start, y_end):
                                        if 0 <= i < density_grid.shape[0] and 0 <= j < density_grid.shape[1]:
                                            density_grid[i, j] += 1

                        if placed_flag:
                            break
                if placed:
                    break

        if not placed or len(placed) < truck_count:
            print(f"⚠️ Only {len(placed)} out of {truck_count} trucks placed. Retrying with expanded warehouse...")

            expansion_factor = 1.3  # Or a smarter scaling based on how many trucks were missed
            new_width = int(warehouse_width * expansion_factor)
            new_length = int(warehouse_length * expansion_factor)

            shape = self.generate_irregular_shape(shape_points, new_width, new_length)
            if not shape or "polygon" not in shape:
                return {"error": "Failed to generate expanded shape."}

            polygon = shape["polygon"]
            outline = shape["outline"]
            self.drive_lane = compute_drive_lane(polygon, trucks)

            # Re-run placement loop exactly as before but with updated polygon dimensions
            # You could even wrap your current placement logic in a helper: `place_trucks(...)`
            # For brevity, you can rerun similar logic or call back into `_compute_optimization` recursively if structured right

            retry_cfg = cfg.copy()
            retry_cfg["warehouse_width"] = new_width
            retry_cfg["warehouse_length"] = new_length
            retry_cfg["polygon"] = polygon
            retry_cfg["outline"] = outline

            # 👇 Either rerun the same placement loop here, or modularize as:
            result = self._compute_optimization(retry_cfg, trucks)
            result["expanded"] = True
            return result
        area = round(polygon.area, 2)

        return {
            "placed": placed,
            "strategy": strategy,
            "outline": outline,
            "polygon": polygon,
            "warehouse_width": warehouse_width,
            "warehouse_length": warehouse_length,
            "area": area,
            "goal": optimization_goal,
            "truck_count": truck_count,
            "heatmap": heatmap_grid.tolist(),
            "density": density_grid.tolist()
        }

        # ── Auto-select entry point


    def _complete_optimization(self, result):
        self.ax.clear()
        if "error" in result:
            QMessageBox.warning(self, "Optimization Failed", result["error"])
            self.spinner_movie.stop()
            self.spinner.setVisible(False)
            return
        outline = result.get("outline", [])

        polygon = result.get("polygon")
        if not polygon or not outline or len(outline) < 3:
            print("❌ Missing polygon or invalid outline")
            return

        # 🚩 Core layout variables
        entry_x = polygon.centroid.x
        exit_x = polygon.centroid.x

        bottom_y = polygon.bounds[1]
        top_y = polygon.bounds[3]
        entry_y = bottom_y - 1.5
        exit_y = top_y + 1.5

        # 🚚 Dynamically determine lane width from placed trucks
        try:
            lane_width = max(w for _, _, w, _, _, _ in result["placed"])
        except Exception as e:
            print("⚠️ Unable to compute lane width:", e)
            lane_width = 1.5  # fallback

        # 🟥 Reserve central traversal path visually
        drive_path = Rectangle(
            (entry_x - lane_width / 2, bottom_y),
            lane_width,
            top_y - bottom_y,
            facecolor='none',
            edgecolor='red',
            linestyle='dashdot',
            linewidth=2
        )
        self.ax.add_patch(drive_path)
        self.ax.text(entry_x+3, bottom_y - 2,
                     f"TRUCK FLOW (Width: {lane_width}m)", fontsize=8,
                     color='red')
################################################################### GENAI

        from dignostics import generate_layout_feedback
        from layout_narrator import narrate_layout_feedback


        placed_trucks = result["placed"]
        reserved_lane = self.drive_lane
        feedback_lines = generate_layout_feedback(polygon, placed_trucks, reserved_lane)
        feedback_summary = "\n".join(feedback_lines)
        api_key = openai.api_key
        narration = narrate_layout_feedback(feedback_summary, api_key)
        print(narration)
        self.narration_box.clear()
        self.narration_box.setPlainText(narration)

        ##############################################################################
        # self.ax.text(entry_x + 5, bottom_y - 2,
        #              f"TRUCK FLOW (Width: {lane_width}m)", fontsize=8,
        #              ha='center', color='red')

        try:
            assert outline and len(outline) >= 3
            assert all(isinstance(p, tuple) and len(p) == 2 for p in outline)
            assert all(isinstance(coord, (int, float)) for pt in outline for coord in pt)

            warehouse_patch = MplPolygon(outline, closed=True, edgecolor='black',
                                         facecolor='#f0f0f0', alpha=0.5)
            self.ax.add_patch(warehouse_patch)
        except Exception as e:
            print("🚫 Skipped polygon due to invalid outline:", e)

        clearance = float(self.clearance_input.text())

        self.ax.set_xlim(-1, result["warehouse_width"] + 1)
        self.ax.set_ylim(-3, result["warehouse_length"] + 1)
        #entry_x = result["warehouse_width"] / 2

        # entry_lane = box(entry_x - 0.75, bottom_y, entry_x + 0.75, warehouse_length)
        # # ── Auto-select entry point ──
        #entry_y = min(p[1] for p in result["outline"]) - 1.5  # place marker just outside bottom edge

        polygon = result.get("polygon")
        if polygon and hasattr(polygon, "centroid"):
            entry_x = polygon.centroid.x
            entry_y = polygon.bounds[1] - 1.5  # bottom edge offset
        else:
            entry_x = result["warehouse_width"] / 2
            entry_y = -2.0

        self.ax.plot(entry_x, entry_y, marker="v", color="green", markersize=10)
        self.ax.text(entry_x-4, entry_y - 0.5, "ENTRY", ha='center', fontsize=8, color='green')
        try:
            entry_y = min(p[1] for p in outline) - 1.5
        except Exception as e:
            print("❌ Failed to compute entry_y:", e)
            entry_y = -2.0

        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.grid(True, linestyle='--', alpha=0.3)

        # ── Auto-select entry point

        # Draw warehouse polygon
        warehouse_patch = MplPolygon(result["outline"], closed=True, edgecolor='black',
                                     facecolor='#f0f0f0', alpha=0.5)
        if len(outline) >= 3:
            warehouse_patch = MplPolygon(outline, closed=True, edgecolor='black',
                                         facecolor='#f0f0f0', alpha=0.5)
            self.ax.add_patch(warehouse_patch)

        else:
            print("🚫 Skipping warehouse polygon drawing: insufficient outline points.")

        bottom_y = min(p[1] for p in result["outline"])
        try:
            top_y = max(p[1] for p in result["outline"])
        except Exception:
            top_y = result["warehouse_length"]
        candidate_x = []
        step = 0.5

        exit_candidates = []
        step = 0.5

        for x in np.arange(0, result["warehouse_width"], step):
            exit_path = box(x - 0.25, 0, x + 0.25, top_y)
            blocked = False
            for _, _, w, l, tx, ty in result["placed"]:
                truck_box = box(tx, ty, tx + w, ty + l)
                if exit_path.intersects(truck_box):
                    blocked = True
                    break
            if not blocked:
                exit_candidates.append(x)

        if exit_candidates:
            exit_x = sorted(exit_candidates, key=lambda val: abs(val - result["warehouse_width"] / 2))[0]
            exit_y = top_y + 1.5
            self.ax.plot(exit_x, exit_y, marker="^", color="red", markersize=10)
            self.ax.text(exit_x, exit_y + 0.5, "EXIT", ha='center', fontsize=8, color='red')
        else:
            print("🚫 No clear exit path available.")
            exit_x, exit_y = None, None


        for x in np.arange(0, result["warehouse_width"], step):
            entry_path = box(x - 0.25, bottom_y, x + 0.25, result["warehouse_length"])
            blocked = False
            for tid, ttype, w, l, tx, ty in result["placed"]:
                truck_box = box(tx, ty, tx + w, ty + l)
                if entry_path.intersects(truck_box):
                    blocked = True
                    break
            if not blocked:
                candidate_x.append(x)

        # Choose center-most or widest point
        if candidate_x:
            entry_x = sorted(candidate_x, key=lambda val: abs(val - result["warehouse_width"] / 2))[0]
            entry_y = bottom_y - 1.5
            self.ax.plot(entry_x, entry_y, marker="v", color="green", markersize=10)
            self.ax.text(entry_x-4, entry_y - 0.5, "ENTRY", ha='center', fontsize=8, color='green')

        # 🔵 Truck Density Overlay + Colorbar
        if "density" in result and self.density_toggle.isChecked():
            density = np.array(result["density"])
            density_scaled = density / np.max(density) if np.max(density) > 0 else density
            flipped_density = np.flipud(density_scaled)
            im = self.ax.imshow(flipped_density, cmap='Blues', alpha=0.4, origin='lower',
                                extent=(0, float(result["warehouse_width"]), 0, float(result["warehouse_length"])))
            self.figure.colorbar(im, ax=self.ax, orientation='vertical', shrink=0.8)
            print("Max density:", np.max(density))
            print("Min density:", np.min(density))
        for tid, ttype, w, l, x, y in result["placed"]:
            buffer = Rectangle((x - clearance / 2, y - clearance / 2),
                               w + clearance, l + clearance,
                               facecolor='#d3d3d3', edgecolor='gray', alpha=0.4)
            truck = Rectangle((x, y), w, l, facecolor='skyblue', edgecolor='blue')
            maneuver = Rectangle((x, y + l), w, l * 0.6,
                                 facecolor='#ffcc99', edgecolor='orange', alpha=0.3)

            coords = list(polygon.exterior.coords)


            self.ax.add_patch(buffer)
            self.ax.add_patch(truck)
            self.ax.add_patch(maneuver)









            # ── Draw truck routing paths from entry to truck centers ──
            from matplotlib.lines import Line2D

            try:
                bottom_y = min(p[1] for p in result["outline"])
            except Exception:
                bottom_y = 0  # fallback if outline fails

            for tid, ttype, w, l, x, y in result["placed"]:
                dest_x = x + w / 2
                dest_y = y + l / 2

                # self.ax.annotate("",
                #                  xy=(dest_x, dest_y),
                #                  xytext=(entry_x, bottom_y),
                #                  arrowprops=dict(arrowstyle="->", color="purple", alpha=0.5, linewidth=1.5)
                #                  )

            self.ax.text(x + w / 2, y + l / 2, f"{tid}\n{ttype}",
                         ha='center', va='center', fontsize=7)
            self.ax.text(x + w / 2, y + l + (l * 0.6) / 2, "🟧 Maneuver",
                         ha='center', va='center', fontsize=6)

        # Strategy summary at bottom of plot
        strategy_note = f"📌 Optimization Mode: {result['goal']}\n📦 Strategy: {result['strategy']}"
        # self.ax.text(result["warehouse_width"] / 2, -2.5,
        #              strategy_note,
        #              ha='center', va='top', fontsize=9, color='black',
        #              bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

        print("🖼️ Ax limits:", self.ax.get_xlim(), self.ax.get_ylim())
        print("🔢 Truck count placed:", len(result["placed"]))
        print("📐 Outline points:", len(result.get("outline", [])))
        for _, _, w, l, tx, ty in result["placed"]:
            center_x = tx + w / 2
            center_y = ty + l / 2

            # Construct segmented path
            path_x = [entry_x, center_x, exit_x]
            path_y = [entry_y, center_y, exit_y]

            #self.ax.plot(path_x, path_y,
             #            color='red', linestyle='-', linewidth=1.5, alpha=0.6)

        # ── 1. Determine polygon bounds & centroid ──
        polygon = result.get("polygon")
        outline = result.get("outline", [])
        clearance = float(self.clearance_input.text())

        if not polygon or not outline:
            print("❌ Missing polygon or outline")
            return

        entry_x = polygon.centroid.x
        entry_y = polygon.bounds[1] - 1.5  # below bottom edge
        exit_x = polygon.centroid.x
        exit_y = polygon.bounds[3] + 1.5  # above top edge
        bottom_y = polygon.bounds[1]
        top_y = polygon.bounds[3]

        # ── 2. Draw warehouse polygon ──
        warehouse_patch = MplPolygon(outline, closed=True, edgecolor='black',
                                     facecolor='#f0f0f0', alpha=0.5)
        self.ax.add_patch(warehouse_patch)

        #self.save_polygon_to_pdf(result["polygon"], "optimized_layout_polygon.pdf")



        # ── 3. Draw entry and exit markers ──
        # self.ax.plot(entry_x, entry_y, marker="v", color="green", markersize=10)
        # self.ax.text(entry_x, entry_y - 0.5, "ENTRY", ha='center', fontsize=8, color='green')
        #
        # self.ax.plot(exit_x, exit_y, marker="^", color="red", markersize=10)
        # self.ax.text(exit_x, exit_y + 0.5, "EXIT", ha='center', fontsize=8, color='red')

        # ── 4. Draw central traversal lane ──

        truck_width = max(w for _, _, w, _, _, _ in result["placed"])

        lane_width = truck_width  # Adjust as needed
        drive_path = Rectangle(
            (entry_x - lane_width / 2, bottom_y),
            lane_width,
            top_y - bottom_y,
            facecolor='none',
            edgecolor='red',
            linestyle='dashdot',
            linewidth=2
        )
        self.ax.add_patch(drive_path)
        #self.ax.text(entry_x+5, bottom_y - 2, "TRUCK FLOW", fontsize=8,
                     #ha='center', color='red')

        # ── 5. Draw trucks and movement paths ──
        for tid, ttype, w, l, x, y in result["placed"]:
            buffer = Rectangle((x - clearance / 2, y - clearance / 2),
                               w + clearance, l + clearance,
                               facecolor='#d3d3d3', edgecolor='gray', alpha=0.4)
            truck = Rectangle((x, y), w, l, facecolor='skyblue', edgecolor='blue')
            maneuver = Rectangle((x, y + l), w, l * 0.6,
                                 facecolor='#ffcc99', edgecolor='orange', alpha=0.3)

            self.ax.add_patch(buffer)
            self.ax.add_patch(truck)
            self.ax.add_patch(maneuver)

            # Labels
            self.ax.text(x + w / 2, y + l / 2, f"{tid}\n{ttype}",
                         ha='center', va='center', fontsize=7)
            self.ax.text(x + w / 2, y + l + (l * 0.6) / 2, "🟧 Maneuver",
                         ha='center', va='center', fontsize=6)

            # ── 6. Draw full movement path: Entry → Truck → Exit ──
            center_x = x + w / 2
            center_y = y + l / 2

            path_x = [entry_x, center_x, exit_x]
            path_y = [entry_y, center_y, exit_y]

            #self.ax.plot(path_x, path_y,
                         #color='red', linestyle='-', linewidth=1.5, alpha=0.6)

        # ── 7. Final drawing setup ──
        self.ax.set_xlim(-1, result["warehouse_width"] + 1)
        self.ax.set_ylim(-3, result["warehouse_length"] + 3)
        self.ax.set_xlabel("X Position (m)")
        self.ax.set_ylabel("Y Position (m)")
        self.ax.grid(True, linestyle='--', alpha=0.3)

        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

            self.ax.text(mid_x, mid_y, f"{length:.2f} m",
                    fontsize=9,
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.7))

        self.canvas.draw()

        # Update summary panel
        summary_text = f"""
    📦 Strategy: {result['strategy']}
    📝 Mode: {result['goal']}

    📍 Warehouse Area: {result['area']} m²
    🚛 Trucks Requested: {result['truck_count']}
    ✅ Trucks Placed: {len(result['placed'])}

    🎨 Confidence Indicators:
    🟧 Maneuver zones applied to each truck (60% of length)
    🚫 Overlap-free placement validated
    🏗️ Irregular warehouse footprint based on convex hull
        """.strip()

        if self.config.get("optimize_goal") == "Maximize Truck Count":
            summary_text += (
                    "🧠 Cost Function: J = –Nₜ\n"
                    "  Objective: Place maximum number of trucks inside the polygon.\n"
                    "  Constraints:\n"
                    "   - Truck ⊆ Polygon\n"
                    "   - No overlap between trucks\n"
                    "   - Respect clearance ≥ %.2fm\n\n" % self.config.get("clearance", 2.0)
            )
        else:
            summary_text += (
                    "🧠 Cost Function: J = Area(P) + λ · max(0, N_req – Nₜ)\n"
                    "  Objective: Minimize warehouse area while fitting target trucks.\n"
                    "  Constraints:\n"
                    "   - Truck ⊆ Polygon\n"
                    "   - No overlap, clearance respected\n"
                    "   - Truck count ≥ %d\n\n" % self.config.get("truck_count", 10)
            )

        # ➕ Append agent strategy (already generated)
        summary_text += narration

        self.agent_output_display.setText(summary_text)





        self.agent_output_display.setPlainText(summary_text)

        self.spinner_movie.stop()
        self.spinner.setVisible(False)


    def _is_squareish(self, poly, tolerance=0.05):
        bounds = poly.bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        aspect_ratio = min(width, height) / max(width, height)
        return aspect_ratio, (aspect_ratio > (1 - tolerance))

    def generate_irregular_shape(self, num_points, max_x, max_y,
                                 min_ratio=0.8, max_ratio=1.0,
                                 dock_side='bottom', seed=None):
        if seed is not None:
            random.seed(seed)

        min_area = max_x * max_y * min_ratio
        max_area = max_x * max_y * max_ratio
        margin = 1.5

        def safe_return(poly):
            if poly is None or not poly.is_valid or poly.is_empty:
                print("⚠️ Invalid polygon.")
                return None
            coords = list(poly.exterior.coords)
            if len(coords) < 3:
                print("⚠️ Outline too short for polygon rendering.")
                return None
            bounds = poly.bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]
            aspect_ratio, is_square = self._is_squareish(poly)
            shape_type = "squareish" if is_square else "elongated"
            return {
                "outline": coords,
                "polygon": poly,
                "bounds": bounds,
                "centroid": tuple(poly.centroid.coords[0]),
                "shape_type": shape_type
            }

        for _ in range(50):
            points = [(random.uniform(margin, max_x - margin),
                       random.uniform(margin, max_y - margin)) for _ in range(num_points)]

            # 🚪 Bias shape edge toward dock side
            if dock_side == 'bottom':
                points += [(random.uniform(margin, max_x - margin), margin / 2) for _ in range(2)]
            elif dock_side == 'top':
                points += [(random.uniform(margin, max_x - margin), max_y - margin / 2) for _ in range(2)]

            try:
                hull = MultiPoint(points).convex_hull
            except Exception as e:
                print("❌ Convex hull error:", e)
                continue

            area = hull.area
            _, is_square = self._is_squareish(hull)

            if hull.is_valid and not hull.is_empty and len(hull.exterior.coords) >= 3:
                if min_area <= area <= max_area and not is_square:
                    return safe_return(hull)

        # 🔁 Fallback: Jittered quad
        try:
            fallback_quad = MultiPoint([
                (0, 0),
                (max_x * random.uniform(0.8, 1.0), random.uniform(margin, margin * 2)),
                (max_x - random.uniform(margin, margin * 2), max_y * random.uniform(0.8, 1.0)),
                (random.uniform(margin, margin * 2), max_y - random.uniform(margin, margin * 2))
            ]).convex_hull
            return safe_return(fallback_quad)
        except Exception as e:
            print("❌ Fallback shape failed:", e)
            return None

# ── Run App ──
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AgenticOptimizerGUI()
    window.resize(1150, 700)
    window.show()
    sys.exit(app.exec_())