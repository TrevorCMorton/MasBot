import os.path
import p3.memory_watcher
import p3.menu_manager
import p3.pad
import p3.state
import p3.state_manager
import p3.screen_watcher
import numpy as np
import cv2
from threading import Thread
import time


class P4:
    def __init__(self, setup):
        self.setup = setup
        self.selected_fox = False
        self.selected_cpu = False
        self.setup_cpu = False
        self.sw = None
        self.dolphin_dir = None
        self.reward = 0
        self.players = []
        self.post_game = False
        self.restarting = False
        self.frame = None
        self.cpu_level = 0

    def find_dolphin_dir(self):
        """Attempts to find the dolphin user directory. None on failure."""
        candidates = ['~/.dolphin-emu', '~/.local/share/dolphin-emu']
        for candidate in candidates:
            path = os.path.expanduser(candidate)
            if os.path.isdir(path):
                return path
        return None

    def write_locations(self, dolphin_dir, locations):
        """Writes out the locations list to the appropriate place under dolphin_dir."""
        path = dolphin_dir + '/MemoryWatcher/Locations.txt'
        with open(path, 'w') as f:
            f.write('\n'.join(locations))

            dolphin_dir = self.find_dolphin_dir()
            if dolphin_dir is None:
                print('Could not detect dolphin directory.')
                return

    def run(self, state, sm, mw, pad):
        mm = p3.menu_manager.MenuManager()
        result = False
        while not result:
            last_frame = state.frame
            res = next(mw)
            if res is not None:
                sm.handle(*res)
            if state.frame > last_frame:
                result = self.make_action(state, pad, mm)

    def make_action(self, state, pad, mm):
        if state.menu == p3.state.Menu.Game:
            pad.reset()
            return True
        elif state.menu == p3.state.Menu.Characters:
            if self.setup:
                if mm.pick_cpu(state, pad):
                    if mm.pick_fox(state, pad):
                        if mm.set_rules(state, pad):
                            if mm.set_level(state, pad, self.cpu_level):
                                mm.press_start_lots(state, pad)
            else:
                if mm.pick_fox(state, pad):
                    return False
            return False
        elif state.menu == p3.state.Menu.Stages:
            if mm.pick_map(state, pad):
                mm.press_start_lots(state, pad)
            return False
        elif state.menu == p3.state.Menu.PostGame:
            mm.press_start_lots(state, pad)
            return False

    def start(self):
        self.dolphin_dir = self.find_dolphin_dir()
        if self.dolphin_dir is None:
            print('Could not find dolphin config dir.')
            return

        if self.cpu_level == 0:
            self.cpu_level = np.random.choice(range(1, 10))

        game_state = p3.state.State()
        sm = p3.state_manager.StateManager(game_state)
        self.write_locations(self.dolphin_dir, sm.locations())

        pad_path = self.dolphin_dir + '/Pipes/p3'
        mw_path = self.dolphin_dir + '/MemoryWatcher/MemoryWatcher'
        with p3.pad.Pad(pad_path) as pad, p3.memory_watcher.MemoryWatcher(mw_path) as mw:
            self.run(game_state, sm, mw, pad)
        self.sw = p3.screen_watcher.ScreenWatcher()
        thread = Thread(target=self.frame_reward)
        thread.start()

    def is_post_game(self):
        return self.post_game

    def get_frame_reward(self):
        temp = self.reward
        self.reward = 0
        return temp

    def frame_reward(self):
        game_state = p3.state.State()
        sm = p3.state_manager.StateManager(game_state)
        mw_path = self.dolphin_dir + '/MemoryWatcher/MemoryWatcher'
        mm = p3.menu_manager.MenuManager()
        with p3.memory_watcher.MemoryWatcher(mw_path) as mw:
            while True:
                last_frame = game_state.frame
                res = next(mw)
                if res is not None:
                    sm.handle(*res)
                if game_state.frame > last_frame:
                    self.frame = self.get_frame(84)

                    if game_state.menu == p3.state.Menu.PostGame:
                        self.post_game = True
                    elif game_state.menu == p3.state.Menu.Game:
                        self.post_game = False

                    if self.post_game:
                        pad_path = self.dolphin_dir + '/Pipes/p3'
                        with p3.pad.Pad(pad_path) as pad:
                            mm.press_start_lots(game_state, pad)

                    i = 0
                    players = game_state.players
                    players_tuples = []
                    while i < len(players):
                        tuple = (players[i].percent, players[i].stocks)
                        if len(self.players) != 0:
                            if i == 2:
                                if self.players[i][0] < tuple[0]:
                                    self.reward -= .1
                                if self.players[i][1] > tuple[1]:
                                    self.reward -= 1
                            else:
                                if self.players[i][0] < tuple[0]:
                                    self.reward += .1
                                if self.players[i][1] > tuple[1]:
                                    self.reward += 1
                        i += 1
                        players_tuples.append(tuple)
                    self.players = players_tuples

    def get_frame(self, size):
        arr = self.to_grayscale(cv2.resize(np.array(next(self.sw)), (size, size), interpolation=cv2.INTER_LINEAR)[:,:,:3])[:,:,:1].flatten()
        return (arr - arr.mean()) / np.abs(arr.max())

    def get_frame_fast(self):
        while self.frame is None:
            time.sleep(1)
        return self.frame

    def get_flat_frame(self):
        return self.get_frame_fast()

    def to_grayscale(self, im):
        im[:] = im.mean(axis=-1,keepdims=1)
        return im

    def execute(self, actions):
        pad_path = self.dolphin_dir + '/Pipes/p3'
        with p3.pad.Pad(pad_path) as pad:
            for action in actions:
                if action == 'MR':
                    pad.tilt_stick(p3.pad.Stick.MAIN, .5, .5)
                elif action == 'MN':
                    pad.tilt_stick(p3.pad.Stick.MAIN, .5, 1)
                elif action == 'MNW':
                    pad.tilt_stick(p3.pad.Stick.MAIN, 1, 1)
                elif action == 'MW':
                    pad.tilt_stick(p3.pad.Stick.MAIN, 1, .5)
                elif action == 'MSW':
                    pad.tilt_stick(p3.pad.Stick.MAIN, 1, 0)
                elif action == 'MS':
                    pad.tilt_stick(p3.pad.Stick.MAIN, .5, 0)
                elif action == 'MSE':
                    pad.tilt_stick(p3.pad.Stick.MAIN, 0, 0)
                elif action == 'ME':
                    pad.tilt_stick(p3.pad.Stick.MAIN, 0, .5)
                elif action == 'MNE':
                    pad.tilt_stick(p3.pad.Stick.MAIN, 0, 1)

                if action == 'CR':
                    pad.tilt_stick(p3.pad.Stick.C, .5, .5)
                elif action == 'CN':
                    pad.tilt_stick(p3.pad.Stick.C, .5, 1)
                elif action == 'CNW':
                    pad.tilt_stick(p3.pad.Stick.C, 1, 1)
                elif action == 'CW':
                    pad.tilt_stick(p3.pad.Stick.C, 1, .5)
                elif action == 'CSW':
                    pad.tilt_stick(p3.pad.Stick.C, 1, 0)
                elif action == 'CS':
                    pad.tilt_stick(p3.pad.Stick.C, .5, 0)
                elif action == 'CSE':
                    pad.tilt_stick(p3.pad.Stick.C, 0, 0)
                elif action == 'CE':
                    pad.tilt_stick(p3.pad.Stick.C, 0, .5)
                elif action == 'CNE':
                    pad.tilt_stick(p3.pad.Stick.C, 0, 1)

                if action == 'PA':
                    pad.press_button(p3.pad.Button.A)
                elif action == 'RA':
                    pad.release_button(p3.pad.Button.A)

                if action == 'PB':
                    pad.press_button(p3.pad.Button.B)
                elif action == 'RB':
                    pad.release_button(p3.pad.Button.B)

                if action == 'PY':
                    pad.press_button(p3.pad.Button.Y)
                elif action == 'RY':
                    pad.release_button(p3.pad.Button.Y)

                if action == 'PZ':
                    pad.press_button(p3.pad.Button.Z)
                elif action == 'RZ':
                    pad.release_button(p3.pad.Button.Z)
