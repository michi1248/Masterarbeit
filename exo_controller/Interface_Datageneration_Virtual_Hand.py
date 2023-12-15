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


class Interface(tk.Frame):
    def __init__(self, outer_instance, container, container_instance, title=None,movements=None, use_virtual_hand_interface_for_coord_generation = True):
        tk.Frame.__init__(self, container_instance)

        self.use_virtual_hand_interface_for_coord_generation = use_virtual_hand_interface_for_coord_generation

        self.container = container
        self.container_instance = container_instance

        self.menu_font = Font(family="Times New Roman", size=20)
        self.default_font = Font(family="Times New Roman", size=16)
        self.big_font = Font(family="Times New Roman", size=40)
        self.timer = 0
        self.times = {}
        self.name = ""

        self.movement_label = tk.Label(self.container_instance, text="", font=self.big_font, foreground="black", background="white")
        self.movement_label.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        self.outer_Instance = outer_instance
        # main menubar
        self.menubar = tk.Menu(self.container_instance)
        self.menubar.config(font=self.menu_font)


        self.media_list = movements
        self.media_list_copy = self.media_list.copy()
        # controls
        self.create_control_panel()

    def get_times(self):
        return self.times

    def create_control_panel(self):
        """Add control panel."""
        control_panel = ttk.Frame(self.container_instance)

        self.button = tk.Button(
            control_panel, text="start capturing", command=self.start_capturing
        )
        self.repeat_this_movement = tk.Button(
            control_panel, text="repeat movement", command=self.repeat,bg="white", fg="black"
        )


        # play.pack(side=tk.LEFT)
        self.button.pack(side=tk.LEFT)
        self.repeat_this_movement.pack(side=tk.LEFT)
        control_panel.pack(side=tk.BOTTOM)
        self.open()

    def repeat(self):

        self.button.configure(text="start capturing", bg="white", fg="black")
        self.update()

        if self.curent:
            last_film = self.media_list_copy.index(self.curent) - 1
            this_film = self.media_list_copy.index(self.curent)
        # delete the data that was recorded for this movement


        del self.outer_Instance.movement_name[-1]
        del self.outer_Instance.movement_name[-1]

        if self.use_virtual_hand_interface_for_coord_generation:
            del self.outer_Instance.movement_name_virtual_hand[-1]
            del self.outer_Instance.movement_name_virtual_hand[-1]
            self.outer_Instance.movement_count_virtual_hand -= 1

        self.outer_Instance.movement_count -= 1


        last_key = list(self.outer_Instance.values_movie_start_emg.keys())[-1]
        del self.outer_Instance.values_movie_start_emg[last_key]

        last_key = list(self.times.keys())[-1]
        del self.times[last_key]

        self.media_list.insert(0, self.media_list_copy[this_film])
        self.media_list.insert(0, self.media_list_copy[last_film])
        print("reset the data for this movement and repeat the movement recording")
        self.open()





    def close(self):
        """Close the window."""
        print("close the window")
        #self.container_instance.destroy()
        self.container.delete_window()




    def start_capturing(self):
        # self.time is the time when the video started
        # itme_now is the time in the video when the recodring started

        self.button.configure(text="start capturing", bg="green", fg="white")
        self.update()

        time_now = time.time() - self.time

        self.outer_Instance.values_movie_start_emg.update({self.name: time_now})

        self.times.update({self.name: time_now})
        self.timer = time.time()

        print(
            "start of capturing for movement: " + self.name + " video time: ", time_now
        )

        time.sleep(self.outer_Instance.recording_time)

        self.button.configure(text="start capturing", bg="white", fg="black")
        self.update()


        if len(self.media_list) == 0:
            self.outer_Instance.escape_loop = True
            self.outer_Instance.escape_loop_virtual_hand = True
            self.close()

        self.outer_Instance.stop_emg_stream = True
        self.outer_Instance.stop_coordinate_stream = True

        self.open()

    def open(self):
        try:
            self.curent = self.media_list[0]
            self.play_film(self.media_list[0])
        # self.close()
        except Exception as e:
            print(e)
            pass

    def play_film(self, movement):
        """Invokes the `play` method on the vlc instance for the current file."""

        self.name = movement
        self.outer_Instance.movement_name.append(self.name)
        self.outer_Instance.movement_name_virtual_hand.append(self.name)

        # Update the label with the current movement name
        self.movement_label.config(text=f"Current Movement: {self.name}")

        self.media_list.pop(0)
        self.time = time.time()

        self.outer_Instance.video_time = time.time()



class Container:
    def __init__(self):
        self.tk_instance = tk.Tk()
        self.tk_instance.title("py player")
        self.tk_instance.protocol("WM_DELETE_WINDOW", self.delete_window)
        self.tk_instance.geometry("1920x1080")  # default to 1080p
        self.tk_instance.configure(background="white")
        self.theme = ttk.Style()
        self.theme.theme_use("alt")

    def delete_window(self):
        tk_instance = self.tk_instance
        tk_instance.quit()
        tk_instance.destroy()
        # os._exit(1)

    def __repr__(self):
        return "Base tk Container"
