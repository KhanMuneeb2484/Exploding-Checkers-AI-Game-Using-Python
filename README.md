
# ♟️ Exploding Checkers

**Exploding Checkers** is a thrilling Python-based checkers game built with **Pygame**, featuring a unique explosive piece mechanic 💥. Players can detonate explosive pieces to clear a 3x3 grid, adding strategic depth to classic checkers.

Challenge a smart AI opponent 🤖 that uses **minimax with alpha-beta pruning** to execute tactical moves and high-impact explosions. The game includes smooth animations 🎞️, multi-capture sequences, king promotion 👑, and an intuitive UI for an engaging experience.

---

## 📽️ Demo Video

▶️ **Watch the demo (with voiceover by team member):**  
[![Watch Demo Video](https://img.shields.io/badge/Click%20to%20Watch-Demo%20Video-blue?logo=youtube)](https://drive.google.com/file/d/1WYAcRguT41h6WAzvD5Z91zI5adUUnori/view?usp=drive_link)


## 🚀 Features

- 💣 **Explosive Pieces**: Each player starts with 3 explosive pieces that can detonate a 3x3 grid with a **300ms green circle animation**.
- 🤖 **Smart AI Opponent**: Blue AI uses **minimax + alpha-beta pruning**, prioritizing captures and explosions to eliminate multiple pieces or kings.
- 🔁 **Multi-Capture**: Players must continue capturing if more captures are available, following traditional checkers rules.
- 👑 **King Promotion**: Reach the back row to promote a piece to a king with bidirectional movement.
- 🌀 **Smooth Animations**: Enjoy 500ms ease-in-out movement animations and 300ms explosion effects.
- 🖱️ **Intuitive UI**: Yellow dots indicate valid moves, glowing outlines show selected pieces, and a turn tracker keeps players informed.
- 🧠 **Turn Counting**: Tracks total turns for strategic analysis.
- 📄 **Single-File Code**: All logic is contained in `main.py` for easy exploration and edits.

---

## 🧰 Installation

### 1. 📥 Clone the Repository

```bash
git clone https://github.com/KhanMuneeb2484/Exploding-Checkers-AI-Game-Using-Python.git
cd Exploding-Checkers-AI-Game-Using-Python
```

### 2. 🐍 Install Python

Make sure you have Python 3.7+ installed.  
Download from [python.org](https://www.python.org/downloads/).

### 3. 🎮 Install Pygame

```bash
pip install pygame
```

### 4. 📁 Verify File

Ensure `main.py` is in your project directory.

---

## ▶️ Usage

### 🕹️ Run the Game

```bash
python main.py
```

### 🎲 Play

- 🔴 **Red Player (You)**:  
  - Left-click to select and move pieces.  
  - Right-click to detonate explosive pieces (green border with asterisk).

- 🔵 **Blue AI**:  
  - Automatically makes intelligent moves and triggers explosions.  
  - Prioritizes capturing multiple pieces or kings.

### 🎯 Objective

Eliminate all opponent pieces or block all of their valid moves to win.

---

## 📜 Gameplay Rules

- ♟️ **Standard Checkers**: Move diagonally on dark squares. Capture by jumping over opponent pieces. Kings move bidirectionally.
- 💣 **Explosive Pieces**: Each player gets 3 explosive pieces. Right-click (Red) or AI can trigger a 3x3 explosion. Chain reactions may occur.
- ⚔️ **Mandatory Captures**: If a capture is available, it must be taken. Multi-captures must be continued.
- 🧨 **Explosions**: Can be used any time during your turn if available.
- 🏁 **Winning**: Win by capturing all opponent pieces or blocking all their valid moves.

---

## 📁 Project Structure

```
Exploding-Checkers-AI-Game-Using-Python/
│
├── main.py  # All game logic, UI, AI, animations, and rules
└── README.md              # This file
```


---

## 🙏 Acknowledgments

- Built with ❤️ and [Pygame](https://www.pygame.org/).
- Inspired by classic checkers with an explosive twist 💥.
- Thanks to all playtesters for valuable feedback!

---

## 🐞 Issues & Feedback

Found a bug? Have a suggestion? Open an issue and include:

- 🔍 A clear description (e.g. “Red piece at (5,2) can’t move”)
- 🧩 Current board state (piece positions and explosives)
- 💻 Console error messages, if any
- 🧪 Python version and OS

Enjoy playing **Exploding Checkers**! ♟️💣🎮
