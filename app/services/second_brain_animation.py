import os
import random
import time


# ============================================================
# SECOND BRAIN
# Terminal Startup Interface
# ============================================================

# ---------- Colors ----------

RESET = "\033[0m"
BOLD = "\033[1m"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
WHITE = "\033[97m"
ORANGE = "\033[38;5;208m"


# ---------- Terminal Helpers ----------

def clear():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def type_text(text, delay=0.03, color=WHITE):
    """Print text character by character."""
    for char in text:
        print(f"{color}{char}{RESET}", end="", flush=True)
        time.sleep(delay)
    print()


def loading_animation(text, duration=1.5, color=CYAN):
    """Display a simple loading animation."""
    symbols = "|/-\\"
    start_time = time.time()
    index = 0

    while time.time() - start_time < duration:
        symbol = symbols[index % len(symbols)]

        print(
            f"\r{color}{text} {symbol}{RESET}",
            end="",
            flush=True
        )

        time.sleep(0.12)
        index += 1

    print(f"\r{GREEN}{text} ✓{RESET}")


def matrix_effect(lines=8, width=65):
    """Display a short digital initialization effect."""
    for _ in range(lines):
        line = "".join(
            random.choice("01")
            for _ in range(width)
        )

        print(f"{GREEN}{line}{RESET}")
        time.sleep(0.05)


# ---------- Block Font ----------

FONT = {
    "A": [
        " ███ ",
        "██ ██",
        "██ ██",
        "█████",
        "██ ██",
        "██ ██",
        "██ ██",
    ],

    "B": [
        "████ ",
        "██ ██",
        "██ ██",
        "████ ",
        "██ ██",
        "██ ██",
        "████ ",
    ],

    "C": [
        " ████",
        "██   ",
        "██   ",
        "██   ",
        "██   ",
        "██   ",
        " ████",
    ],

    "D": [
        "████ ",
        "██ ██",
        "██ ██",
        "██ ██",
        "██ ██",
        "██ ██",
        "████ ",
    ],

    "E": [
        "█████",
        "██   ",
        "██   ",
        "████ ",
        "██   ",
        "██   ",
        "█████",
    ],

    "I": [
        "█████",
        "  ██ ",
        "  ██ ",
        "  ██ ",
        "  ██ ",
        "  ██ ",
        "█████",
    ],

    "N": [
        "██  ██",
        "██  ██",
        "███ ██",
        "██████",
        "██ ███",
        "██  ██",
        "██  ██",
    ],

    "R": [
        "████ ",
        "██ ██",
        "██ ██",
        "████ ",
        "██ ██",
        "██  ██",
        "██  ██",
    ],

    "S": [
        " ████",
        "██   ",
        "██   ",
        " ███ ",
        "   ██",
        "   ██",
        "████ ",
    ],

    "O": [
        " ███ ",
        "██ ██",
        "██ ██",
        "██ ██",
        "██ ██",
        "██ ██",
        " ███ ",
    ],

    " ": [
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
    ],
}


def create_ascii_text(text):
    """Convert text into large block lettering."""
    text = text.upper()

    lines = []

    for row in range(7):
        line = ""

        for character in text:
            if character in FONT:
                line += FONT[character][row] + "  "

        lines.append(line.rstrip())

    return lines


def display_logo(text):
    """Display the project name with a warm red/orange/yellow palette."""

    ascii_lines = create_ascii_text(text)

    colors = [
        RED,
        ORANGE,
        YELLOW,
        ORANGE,
        RED,
        ORANGE,
        YELLOW,
    ]

    for index, line in enumerate(ascii_lines):
        color = colors[index % len(colors)]

        print(
            f"{BOLD}{color}{line}{RESET}"
        )

        time.sleep(0.10)


# ---------- Main Startup Sequence ----------

def start_second_brain():

    clear()

    print()

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------

    type_text(
        ">>> INITIALIZING SECOND BRAIN...",
        delay=0.045,
        color=CYAN
    )

    time.sleep(0.4)

    loading_animation(
        "Loading knowledge systems",
        duration=1.4,
        color=CYAN
    )

    loading_animation(
        "Initializing memory architecture",
        duration=1.4,
        color=BLUE
    )

    loading_animation(
        "Connecting retrieval system",
        duration=1.4,
        color=MAGENTA
    )

    loading_animation(
        "Preparing reasoning engine",
        duration=1.4,
        color=YELLOW
    )

    print()

    type_text(
        ">>> Establishing cognitive workspace...",
        delay=0.035,
        color=WHITE
    )

    time.sleep(0.5)

    # --------------------------------------------------------
    # Digital initialization effect
    # --------------------------------------------------------

    matrix_effect(
        lines=8,
        width=65
    )

    time.sleep(0.6)

    clear()

    # --------------------------------------------------------
    # Project Logo
    # --------------------------------------------------------

    print()

    display_logo("SECOND BRAIN")

    print()

    # --------------------------------------------------------
    # System Status
    # --------------------------------------------------------

    time.sleep(0.5)

    type_text(
        ">>> SYSTEM READY.",
        delay=0.06,
        color=GREEN
    )

    type_text(
        ">>> SECOND BRAIN is online.",
        delay=0.045,
        color=CYAN
    )

    print()


# ---------- Application Entry Point ----------

