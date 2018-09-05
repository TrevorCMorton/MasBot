import os.path


def find_dolphin_dir():
    """Attempts to find the dolphin user directory. None on failure."""
    candidates = ['~/.dolphin-emu', '~/.local/share/dolphin-emu']
    for candidate in candidates:
        path = os.path.expanduser(candidate)
        if os.path.isdir(path):
            return path
    return None

def write_locations(dolphin_dir, locations):
    """Writes out the locations list to the appropriate place under dolphin_dir."""
    path = dolphin_dir + '/MemoryWatcher/Locations.txt'
    with open(path, 'w') as f:
        f.write('\n'.join(locations))

        dolphin_dir = find_dolphin_dir()
        if dolphin_dir is None:
            print('Could not detect dolphin directory.')
            return

def run(state, sm, mw, pad):
    mm = p3.menu_manager.MenuManager()
    result = False
    while not result:
        last_frame = state.frame
        res = next(mw)
        if res is not None:
            sm.handle(*res)
        if state.frame > last_frame:
            result = make_action(state, pad, mm)

def make_action(state, pad, mm):
    if state.menu == p3.state.Menu.Game:
        pad.reset()
        return True
    elif state.menu == p3.state.Menu.Characters:
        if mm.pick_cpu(state, pad):
            mm.pick_fox(state, pad)
        return False
    elif state.menu == p3.state.Menu.Stages:
        # Handle this once we know where the cursor position is in memory.
        pad.tilt_stick(p3.pad.Stick.C, 0.5, 0.5)
        return False
    elif state.menu == p3.state.Menu.PostGame:
        mm.press_start_lots(state, pad)
        return False

def start():
    dolphin_dir = find_dolphin_dir()
    if dolphin_dir is None:
        print('Could not find dolphin config dir.')
        return

    state = p3.state.State()
    sm = p3.state_manager.StateManager(state)
    write_locations(dolphin_dir, sm.locations())

    pad_path = dolphin_dir + '/Pipes/p3'
    mw_path = dolphin_dir + '/MemoryWatcher/MemoryWatcher'
    with p3.pad.Pad(pad_path) as pad, p3.memory_watcher.MemoryWatcher(mw_path) as mw:
        run(state, sm, mw, pad)

if __name__ == '__main__':
    start()
