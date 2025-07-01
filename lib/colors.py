class bcolors:
    """
    ANSI escape sequences for colored terminal text and styles.

    Attributes:
        HEADER (str): Purple color.
        OKBLUE (str): Blue color.
        OKGREEN (str): Green color.
        YELLOW (str): Yellow color.
        WARNING (str): Yellow color (alias).
        FAIL (str): Red color.
        ENDC (str): Reset to default terminal color.
        BOLD (str): Bold text style.
        UNDERLINE (str): Underline text style.
        GREYBG (str): Grey background.
        REDBG (str): Red background.
        GREENBG (str): Green background.
        YELLOWBG (str): Yellow background.
        BLUEBG (str): Blue background.
        PINKBG (str): Pink background.
        CYANBG (str): Cyan background.
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    YELLOW = '\033[93m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # Background colors:
    GREYBG = '\033[100m'
    REDBG = '\033[101m'
    GREENBG = '\033[102m'
    YELLOWBG = '\033[103m'
    BLUEBG = '\033[104m'
    PINKBG = '\033[105m'
    CYANBG = '\033[106m'
