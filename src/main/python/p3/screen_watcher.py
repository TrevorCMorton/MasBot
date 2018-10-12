import Xlib.display
from mss import mss


class ScreenWatcher:
    """Reads and parses game memory changes.

    Pass the location of the socket to the constructor, then either manually
    call next() on this class to get a single change, or else use it like a
    normal iterator.
    """
    def __init__(self):
        self.sct = mss()

        display = Xlib.display.Display()
        root = display.screen().root
        geometry = root.get_geometry()
        #self.width = geometry.__getattr__('width')
        #self.height = geometry.__getattr__('height')
        #self.x = 0
        #self.y = 0

        #windowIDs = root.get_full_property(display.intern_atom('_NET_CLIENT_LIST'), Xlib.X.AnyPropertyType).value
        #for windowID in windowIDs:
            #window = display.create_resource_object('window', windowID)
            #name = window.get_wm_name()  # Title
            #if 'Dolphin 5.0 |' in str(name):
        window = self.get_active_window(display)
        geometry = window.get_geometry()
        self.width = geometry.__getattr__('width')
        self.height = geometry.__getattr__('height')

        self.x = 0
        self.y = 0

        tree = window.query_tree()
        parent = tree.__getattr__('parent')
        while parent is not 0:
            geometry = parent.get_geometry()
            self.x += geometry.__getattr__('x')
            self.y += geometry.__getattr__('y')
            parent_tree = parent.query_tree()
            parent = parent_tree.__getattr__('parent')
        print(self.x, self.y, self.width, self.height)

    def __iter__(self):
        """Iterate over this class in the usual way to get screen changes."""
        return self

    def __next__(self):
        left = self.x
        top = self.y
        right = left + self.width
        lower = top + self.height
        bbox = (left, top, right, lower)

        # Grab the picture
        im = self.sct.grab(bbox)
        return im

    def get_active_window(self, display):
        window = display.get_input_focus().focus
        wmname = window.get_wm_name()
        wmclass = window.get_wm_class()
        if wmclass is None:  # or wmname is None:
            window = window.query_tree().parent
            wmclass = window.get_wm_class()
            wmname = window.get_wm_name()
        print(wmname)
        winclass = wmclass[1]
        print(winclass)
        return window
