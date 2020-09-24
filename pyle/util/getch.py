# Provides a cross-platform way to check whether a key
# has been pressed without actually blocking.  On windows, this
# can be done with kbhit and getch from the msvcrt library.
# on Unix/MacOSX we have to play some games with the lower-level
# attributes of stdin.  The following code does the trick.
# In either case, we return the key code if one has been pressed,
# or None if no key has been pressed.
try:
    import msvcrt
    def getch():
        if msvcrt.kbhit():
            return msvcrt.getch()

except ImportError:    
    import sys, termios, select
    def getch():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        new = termios.tcgetattr(fd)
        new[3] = new[3] & ~termios.ICANON & ~termios.ECHO
        try:
            termios.tcsetattr(fd, termios.TCSANOW, new)
            dr = select.select([sys.stdin], [], [], 0)[0]
            if dr:
                return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, old)