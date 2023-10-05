import os
import pickle
import platform
import socket
import threading
import time
import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import ttk
from tkinter.font import Font

import numpy as np
import pandas as pd
import vlc
class PyPlayer(tk.Frame):
    def __init__(self, outer_instance, container, container_instance, title=None):
        tk.Frame.__init__(self, container_instance)
        self.container = container
        self.container_instance = container_instance
        self.initial_directory = Path(os.path.expanduser("~"))
        self.menu_font = Font(family="Verdana", size=20)
        self.default_font = Font(family="Times New Roman", size=16)
        self.timer = 0
        self.times = {}
        self.name = ""
        # create vlc instance
        self.vlc_instance, self.vlc_media_player_instance = self.create_vlc_instance()
        self.outer_Instance = outer_instance
        # main menubar
        self.menubar = tk.Menu(self.container_instance)
        self.menubar.config(font=self.menu_font)

        # vlc video frame
        self.video_panel = ttk.Frame(self.container_instance)
        self.canvas = tk.Canvas(self.video_panel, background="black")
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.video_panel.pack(fill=tk.BOTH, expand=1)
        self.media_list = [r"trainings_data/videos/" + x for x in os.listdir("trainings_data/videos")]
        self.media_list_copy = self.media_list.copy()
        # controls
        self.create_control_panel()

    def get_times(self):
        return self.times

    def create_control_panel(self):
        """Add control panel."""
        control_panel = ttk.Frame(self.container_instance)

        button = ttk.Button(control_panel, text="start capturing", command=self.start_capturing)
        repeat_this_movement = ttk.Button(control_panel, text="repeat movement", command=self.repeat)
        ttk.Button(control_panel, text="Play", command=self.play)

        # play.pack(side=tk.LEFT)
        button.pack(side=tk.LEFT)
        repeat_this_movement.pack(side=tk.LEFT)
        control_panel.pack(side=tk.BOTTOM)
        self.open()

    def repeat(self):
        if self.curent:
            last_film = self.media_list_copy.index(self.curent)-1
            this_film = self.media_list_copy.index(self.curent)
        #delete the data that was recorded for this movement
        del self.outer_Instance.movement_name[-1]
        del self.outer_Instance.movement_name[-1]

        last_key = list(self.outer_Instance.values_movie_start_emg.keys())[-1]
        del self.outer_Instance.values_movie_start_emg[last_key]
        last_key = list(self.times.keys())[-1]
        del self.times[last_key]


        self.media_list.insert(0, self.media_list_copy[this_film])
        self.media_list.insert(0, self.media_list_copy[last_film])
        self.vlc_media_player_instance.stop()
        print("reset the data for this movement and repeat the movement recording")
        self.open()

    def create_vlc_instance(self):
        vlc_instance = vlc.Instance()
        vlc_media_player_instance = vlc_instance.media_player_new()
        self.container_instance.update()
        return vlc_instance, vlc_media_player_instance

    def get_handle(self):
        return self.video_panel.winfo_id()

    def play(self):
        """Play a file."""
        self.outer_Instance.video_time = time.time()
        if not self.vlc_media_player_instance.get_media():
            self.open()
        else:
            if self.vlc_media_player_instance.play() == -1:
                pass

    def close(self):
        """Close the window."""
        self.container.delete_window()

    def pause(self):
        """Pause the player."""
        self.vlc_media_player_instance.pause()

    def start_capturing(self):
        #self.time is the time when the video started
        # itme_now is the time in the video when the recodring started

        time_now = time.time() - self.time

        self.outer_Instance.values_movie_start_emg.update({self.name: time_now})
        self.times.update({self.name: time_now})
        self.timer = time.time()

        print("start of capturing for movement: " + self.name +" video time: ", time_now)


        time.sleep(self.outer_Instance.recording_time)
        self.vlc_media_player_instance.stop()
        if len(self.media_list) == 0:
            self.close()
            self.outer_Instance.escape_loop = True
        self.outer_Instance.stop_emg_stream = True
        self.open()

    def open(self):
        try:
            self.curent = self.media_list[0]
            self.play_film(self.media_list[0])
        # self.close()
        except Exception as e:
            print(e)
            pass

    def play_film(self, file):
        """Invokes the `play` method on the vlc instance for the current file."""
        file_name = os.path.basename(file)
        self.name = file_name.split(".avi")[0]
        self.outer_Instance.movement_name.append(self.name)
        self.Media = self.vlc_instance.media_new(file)
        # self.Media.get_meta()
        self.vlc_media_player_instance.set_media(self.Media)
        if platform.system() == "Linux":  # for Linux using the X Server
            self.vlc_media_player_instance.set_xwindow(self.get_handle())
        elif platform.system() == "Windows":  # for Windows
            self.vlc_media_player_instance.set_hwnd(self.get_handle())
        elif platform.system() == "Darwin":  # for MacOS
            self.vlc_media_player_instance.set_nsobject(self.get_handle())
        # self.vlc_media_player_instance.play()
        self.media_list.pop(0)
        self.time = time.time()
        self.play()



class BaseTkContainer:
    def __init__(self):
        self.tk_instance = tk.Tk()
        self.tk_instance.title("py player")
        self.tk_instance.protocol("WM_DELETE_WINDOW", self.delete_window)
        self.tk_instance.geometry("1920x1080")  # default to 1080p
        self.tk_instance.configure(background="black")
        self.theme = ttk.Style()
        self.theme.theme_use("alt")

    def delete_window(self):
        tk_instance = self.tk_instance
        tk_instance.quit()
        tk_instance.destroy()
        # os._exit(1)

    def __repr__(self):
        return "Base tk Container"