from vispy.scene import Text, Line
import numpy as np
from PySide6.QtWidgets import QWidget, QSizePolicy, QMainWindow,QApplication
from vispy import scene, app
from vispy.scene import Line
from vispy.scene.visuals import Polygon, Text, Axis
import numpy as np
from vispy.color import Color
import sys
import random
from PySide6.QtCore import QTimer
from vispy import gloo
from vispy import app
import math
from typing import Union

class VispyPlotWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # Plot Widget properties
        self.grid = None
        self.canvas = None
        self.camera = None
        self.title_label = None
        self.title_widget = None
        self.yaxis = None
        self.xaxis = None
        self.xlabel = None
        self.ylabel = None
        self.configured = False
        self.rects = None
        self.lines = None

        self.right_padding = None
        self.left_padding = None
        self.bottom_padding = None
        self.view = None

        # Plot details
        self.plot_data = None
        self.plot_xdata = None
        self.line = None
        self.vertexes = None
        self.lines = None
        self.color = None
        self.connect_vertexes = None
        self.display_time = None
        self.sampling_frequency = None
        self.fread = None
        self.frame_len = None

        self.cursor_marker = None
        self.cursor_marker_pos = None
        self.marker_symbol = None
        self.marker_color = None
        self.marker_size = None

        # Feedback trajectory line
        self.feedback_line = None
        self.feedback_data = None
        self.trajectory_xdata = None
        self.feedback_color = [1, 0, 0, 0.5]

        self.scene = scene.SceneCanvas(parent=self, bgcolor="gray", resizable=True)
        self.scene.unfreeze()
        self.grid = self.scene.central_widget.add_grid(spacing=0, margin=10)
        self.central_widget = self.scene.native
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def configure_lines_plot(
            self,
            display_time=10,
            fs=256,
            lines=320,
            color_map=None,
    ) -> None:
        if self.configured:
            return
        self.view = self.grid.add_view(row=0, col=0, camera="panzoom")
        self.camera = self.view.camera
        self.camera.interactive = False

        self.display_time = display_time
        self.sampling_frequency = fs
        self.vertexes = self.display_time * self.sampling_frequency
        self.lines = lines

        self.reset_data()
        self.define_colors(color_map=color_map)
        self.define_connect()

        self.camera.set_range(
            x=[-0.01, self.plot_data[-1, 0]],
            y=[-0.1, 1.1],
        )

        self.line = scene.Line(
            pos=self.plot_data,
            color=self.color,
            parent=self.view.scene,
            connect=self.connect_vertexes,
            width=1,
            # method="agg",
        )
        self.configured = True

    def configure_feedback_plot(
            self,
            identifier: str = "",
            frame_len=8,
            display_time=10,
            fs=256,
            lines=2,
            marker_size=80,
    ) -> None:
        if self.configured:
            return

        # Viewbox (graph plot)
        self.view = self.grid.add_view(row=0, col=0, camera="panzoom")
        self.camera = self.view.camera
        self.camera.interactive = False
        # Image Feedback
        # image = scene.Image(np.zeros((1, 1, 3)), parent=self.scene.scene)
        # self.grid.add_widget(image, row=0, col=0, row_span=1, col_span=1)
        # Title placement
        # self.title_label = scene.Label(identifier, color="white", font_size=35)
        # self.title_label.height_max = 200
        # self.grid.add_widget(self.title_label, row=0, col=0)

        self.display_time = display_time
        self.sampling_frequency = fs
        self.frame_len = frame_len
        self.vertexes = self.display_time * self.sampling_frequency
        self.lines = lines
        self.marker_size = marker_size
        # self.camera.set_range(x=[-0.01, self.display_time], y=[0, self.lines + 1])

        self.reset_data()
        self.define_colors()
        self.define_connect()

        self.camera.set_range(
            x=[-0.01, self.plot_data[-1, 0]],
            y=[-0.1, 1],
        )
        self.reset_trajectory()
        self.feedback_line = scene.Line(
            pos=self.feedback_data,
            color=self.feedback_color,
            parent=self.view.scene,
            width=10,
        )
        self.line = scene.Line(
            pos=self.plot_data,
            color=self.color,
            parent=self.view.scene,
            connect=self.connect_vertexes,
            width=5,
            # method="agg",
        )
        self.cursor_marker = scene.Markers(size=100)
        self.view.add(self.cursor_marker)
        self.cursor_marker.set_data(
            pos=np.array([self.plot_data[-1, 0] / 2, 0]).reshape(-1, 2),
            symbol="o",
            size=self.marker_size,
            face_color="red",
        )

        self.configured = True

    def configure_display_result_plot(
            self, identifier, lines: int = 1, fs: int = 2048
    ) -> None:
        if self.configured:
            return

            # Create axis
            #     c0        c1      c2      c3
            #  r0 +---------+-------+-------+-------+
            #     |         |       | title |       |
            #  r1 |         +---------------+       |
            #     |         | ylabel|       |       |
            #               | yaxis | view  |       |
            #  r2 | padding +-------+-------+padding|
            #     |         |       | xaxis |       |
            #     |         |       | xlabel|       |
            #  r3 |         +---------------+       |
            #     |         |    padding    |       |
            #     +---------+---------------+-------+

        # Title placement
        self.title_label = scene.Label(identifier, color="white")
        self.title_label.height_max = 20
        self.grid.add_widget(self.title_label, row=0, col=2)

        # Right padding
        self.right_padding = self.grid.add_widget(row=0, col=3, row_span=3)
        self.right_padding.width_max = 10

        # Left padding
        self.left_padding = self.grid.add_widget(row=0, col=0, row_span=3)
        self.left_padding.width_max = 20

        # Bottom padding
        self.bottom_padding = self.grid.add_widget(row=3, col=1, col_span=1)
        self.bottom_padding.height_max = 20

        # Viewbox (graph plot)
        self.view = self.grid.add_view(row=1, col=2, camera="panzoom")
        self.camera = self.view.camera

        # x-Axis
        self.xaxis = scene.AxisWidget(
            orientation="bottom",
            # axis_label="Time",
            axis_font_size=12,
            # axis_label_margin=25,
            tick_color=(1, 1, 1, 1),
        )
        self.xaxis.stretch = (1, 0.1)
        self.grid.add_widget(self.xaxis, row=2, col=2)
        self.xaxis.link_view(self.view)

        self.sampling_frequency = fs
        self.vertexes = self.display_time * self.sampling_frequency
        self.lines = lines

        self.reset_data()
        self.define_colors()
        self.define_connect()
        self.configured = True

    def reset_data(self):
        self.plot_data = np.zeros((self.lines * self.vertexes, 2))
        self.plot_data = self.plot_data.reshape(self.lines, self.vertexes, 2)
        for i, line in enumerate(self.plot_data):
            line[:, 1] += i * 1 / (self.lines)
        self.plot_data = self.plot_data.reshape(self.lines * self.vertexes, 2)
        self.plot_xdata = (
                np.linspace(0, self.vertexes, self.vertexes) / self.sampling_frequency
        )
        self.plot_data[:, 0] = np.tile(self.plot_xdata, self.lines)

    def reset_trajectory(self):
        self.feedback_data = np.zeros((self.vertexes // 2, 2))
        self.trajectory_xdata = (
                np.linspace(0, self.vertexes // 2, self.vertexes // 2)
                / self.sampling_frequency
        )
        self.feedback_data[:, 0] = self.trajectory_xdata

    def define_connect(self):
        self.connect_vertexes = np.empty(
            (self.lines * self.vertexes - self.lines * 1, 2)
        )
        for line in range(self.lines):
            self.connect_vertexes[
            line * self.vertexes - line: self.vertexes * (line + 1) - (line + 1), 0
            ] = np.arange(self.vertexes * line, (line + 1) * self.vertexes - 1)
            self.connect_vertexes[
            line * self.vertexes - line: self.vertexes * (line + 1) - (line + 1), 1
            ] = np.arange(self.vertexes * line + 1, (line + 1) * self.vertexes)

    def define_colors(self, color_map=None):
        colors = []
        if color_map is None:
            colors = np.repeat(np.zeros((self.lines, 3)), self.vertexes, axis=0)
        else:
            for color in color_map:
                colors.append(color)
            colors = np.array(colors)
            colors = colors.reshape(-1, 4) / 255
            colors = np.repeat(colors, self.vertexes, axis=0)
        self.color = colors

    def resizeEvent(self, event) -> None:
        width = self.size().width()
        height = self.size().height()
        self.scene.size = (width, height)

    def set_plot_data(self, data: np.ndarray) -> None:
        frame_len = data.shape[1] if len(data.shape) == 2 else data.shape[0]
        data = data.reshape(self.lines, frame_len) / (self.lines)
        for i in range(data.shape[0]):
            data[i] = data[i] + i / (self.lines)
        plot_data = self.plot_data.copy()
        plot_data = plot_data.reshape(self.lines, self.vertexes, 2)
        plot_data[:, :-frame_len, 1] = plot_data[:, frame_len:, 1]
        plot_data[:, -frame_len:, 1] = data
        self.plot_data = plot_data.reshape(-1, 2)
        self.line.set_data(self.plot_data, self.color, connect=self.connect_vertexes)
        self.scene.update()

    def set_feedback_data(self, data: np.ndarray) -> None:
        frame_len = data.shape[1] if len(data.shape) == 2 else data.shape[0]
        feedback_data = self.feedback_data.copy()
        feedback_data = feedback_data.reshape(self.lines, self.vertexes // 2, 2)
        feedback_data[:, :-frame_len, 1] = feedback_data[:, frame_len:, 1]
        feedback_data[:, -frame_len:, 1] = data
        self.feedback_data = feedback_data.reshape(-1, 2)
        self.feedback_line.set_data(self.feedback_data, self.feedback_color)
        self.scene.update()

        mean_data = data.mean()
        self.cursor_marker.set_data(
            pos=np.array([self.display_time / 2, mean_data]).reshape(-1, 2),
            symbol="o",
            size=self.marker_size,
            face_color="red",
        )

    def refresh_plot(self):
        self.scene.central_widget.remove_widget(self.grid)
        self.grid = self.scene.central_widget.add_grid(spacing=0, margin=10)
        self.configured = False

    def measure_fps(self):
        self.scene.measure_fps()

    def adjust_camera_and_labels(self, n_bars, values, spacing):
        # Adjust camera to include space for labels
        self.view.camera.set_range(x=(-0.35, n_bars * spacing), y=(-0.2, 1.1))

    def add_fixed_y_axis(self):
        # Add Y-axis line
        y_axis_line_pos = np.array([[-0.3, 0], [-0.3, 1]])  # Adjust as needed
        Line(pos=y_axis_line_pos, color='black', parent=self.view.scene, width=2)

        # Add Y-axis ticks and labels
        y_ticks = np.linspace(0, 1, 20)  # Example: 5 ticks from 0 to 1
        for y in y_ticks:
            tick_pos = np.array([[-0.32, y], [-0.28, y]])  # Adjust as needed
            Line(pos=tick_pos, color='black', parent=self.view.scene, width=1)

            # Create tick label
            tick_label = scene.Text(text="{:.1f}".format(y), pos=(-0.23, y), color='black',
                                    font_size=5, parent=self.view.scene)
            tick_label.anchors = ('right', 'center')

    def plot_bar_graphs(self, values, boundaries, important_fingers = None):
        """
        Plots bar graphs with labels and horizontal boundary lines.

        :param values: A list of 5 values, one for each bar plot.
        :param boundaries: A list of 5 arrays/lists, each containing two values (lower and upper boundaries) for each bar plot.
        """
        self.scene.bgcolor = 'white'  # Set background to white

        labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        n_bars = len(values)
        bar_width = 0.1
        spacing = 0.2  # Increased spacing for clear label display

        # Convert hex colors to RGBA with 70% opacity
        pastel_blue = '#aec6cf'
        pastel_rose = '#fddde6'
        pastel_green = '#98fb98'

        # Adjust alpha to 0.7 for 70% opacity
        pastel_blue_transparent = (0.68,0.776,0.811,0.05)
        pastel_rose_transparent = (0.992,0.866,0.902,0.05)
        pastel_green_transparent = (0.596,0.984,0.596,0.05)




        first_trigger = 0
        if self.rects is None:
            first_trigger = 1
            self.rects = [None] * n_bars

        for i, value in enumerate(values):

            x_position = i * spacing
            height = value
            if important_fingers is not None:
                color = pastel_blue_transparent if i not in important_fingers else pastel_blue
            else:
                color = pastel_blue
            if self.rects[i] is None:
                self.rects[i] = scene.Rectangle(center=(x_position + bar_width / 2, height / 2), width=bar_width, height=height,
                                       color=color, parent=self.view.scene)
            else:
                self.rects[i].center=(x_position + bar_width / 2, height / 2)
                self.rects[i].width=bar_width
                self.rects[i].height = height
                self.rects[i].color=color
                self.rects[i].parent = self.view.scene

            # Plotting non-filled boundary rectangles instead of lines
            lower_boundary, upper_boundary = boundaries[i]
            if lower_boundary <= value <= upper_boundary:
                if important_fingers is not None:
                    color = pastel_green_transparent if i not in important_fingers else pastel_green
                else:
                    color = pastel_green
            else:
                if important_fingers is not None:
                    color = pastel_rose_transparent if i not in important_fingers else pastel_rose
                else:
                    color = pastel_rose

            boundary_height = upper_boundary - lower_boundary
            if self.lines is None:
                self.lines = [None] * 5

            self.plot_non_filled_rectangle(center=(x_position + bar_width / 2, lower_boundary + boundary_height / 2),
                                           width=bar_width, height=boundary_height,
                                           color=color, parent=self.view.scene,
                                           i = i)

            # Add X-axis labels
        for i, label in enumerate(labels):
            x_position = i * spacing + bar_width / 2
            text = scene.Text(text=label, pos=(x_position, -0.1), color='black',
                              font_size=7, parent=self.view.scene)
            text.rotation = -45

            # Adjust camera to ensure everything is visible
            # Ensure X-axis labels and Y-axis are correctly positioned and added
        if first_trigger == 1:
            self.adjust_camera_and_labels(n_bars=len(values), values=values, spacing=0.2)
            self.add_fixed_y_axis()

    def plot_non_filled_rectangle(self, center, width, height, color, parent, i):
        x, y = center
        half_width = width / 2.0
        half_height = height / 2.0
        corners = np.array([
            [x - half_width, y - half_height],  # Bottom-left
            [x + half_width, y - half_height],  # Bottom-right
            [x + half_width, y + half_height],  # Top-right
            [x - half_width, y + half_height],  # Top-left
            [x - half_width, y - half_height]  # Close the rectangle by returning to the start
        ])
        # Create a line visual for the rectangle's outline

        if self.lines[i] is None:
            self.lines[i] = Line(pos=corners, color=color, method='gl', parent=parent, width=5)
        else:
            self.lines[i].set_data(pos=corners, color=color, width=5)
    def add_axes(self):
        # This is a placeholder for adding axes
        # Adjust according to your specific needs
        # Axis visuals can be complex and might need manual adjustment or custom implementation
        y_axis = Axis(parent=self.view.scene, anchors=['left',"middle"], tick_color='black', font_size=8)
        y_axis.pos = [0, 0]  # Adjust position as needed
        y_axis.domain = (0, 1)  # Set domain for Y-axis to match the scale


COLORS = np.array(
    [
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 128, 0],
        [102, 0, 102],
        [255, 255, 255],
    ]
)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create the main window and set its size
    main_window = QMainWindow()
    main_window.resize(800, 600)

    # Create an instance of VispyPlotWidget
    plot_widget = VispyPlotWidget()
    # Ensuring the scene and view are correctly initialized
    plot_widget.scene = scene.SceneCanvas(keys='interactive', show=True)
    plot_widget.grid = plot_widget.scene.central_widget.add_grid(spacing=0)
    plot_widget.view = plot_widget.grid.add_view(row=0, col=0, camera='panzoom')
    plot_widget.camera = plot_widget.view.camera  # Assign the camera

    # Sample values and boundaries for the bar plots

    boundaries = [
        [0.6, 0.8],  # Thumb boundaries
        [0.4, 0.6],  # Index boundaries
        [0.7, 0.9],  # Middle boundaries
        [0.3, 0.5],  # Ring boundaries
        [0.5, 0.7]  # Pinky boundaries
    ]


    def update_bar_values():
        # Generate new random values for the bars within a range for demonstration
        new_values = [random.uniform(0, 1) for _ in range(5)]
        plot_widget.plot_bar_graphs(new_values, boundaries,[3] )
        # Create a QTimer to update the bar values periodically


    timer = QTimer()
    timer.timeout.connect(update_bar_values)  # Connect timeout signal to the update function
    timer.start(50)  # Update interval in milliseconds (e.g., 1000ms = 1 second)

    # Show the main window
    main_window.show()



    # Set the VispyPlotWidget instance as the central widget of the main window
    # main_window.setCentralWidget(plot_widget)

    # Show the main window
    main_window.show()

    # Start the application's event loop
    sys.exit(app.exec())

