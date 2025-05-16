import pygame
import sys
import random
import math

# Initialize Pygame and set up the game window
pygame.init()
WIDTH, HEIGHT = 600, 630  # Window size: 600x600 board + 30px header
ROWS, COLS = 8, 8  # 8x8 checkers board
SQUARE_SIZE = WIDTH // COLS  # Size of each square (75px)
HEADER_HEIGHT = 30  # Header for turn indicator

# Define colors used in the game
RED = (220, 20, 60)  # Red pieces (player)
BLUE = (30, 144, 255)  # Blue pieces (AI)
BLACK = (20, 20, 20)  # Dark squares
WHITE = (240, 240, 240)  # Light squares
GREY = (100, 100, 100)  # Board border
YELLOW = (255, 215, 0)  # Valid move indicators
KING_BORDER = (255, 215, 0)  # Crown for king pieces
EXPLOSIVE_BORDER = (50, 205, 50)  # Green border for explosive pieces
HIGHLIGHT = (255, 255, 100)  # Glow for selected piece
BACKGROUND = (10, 10, 20)  # Dark background

# Set up the display window
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Exploding Checkers")

# Track the number of turns (increments when a turn ends)
turn_count = 0


# Piece class: Represents a single checkers piece (Red or Blue)
class Piece:
    PADDING = 15  # Space around piece for rendering
    OUTLINE = 2  # Outline thickness for piece

    # Initialize a piece with position, color, and explosive status
    def __init__(self, row, col, color, explosive=False):
        self.row = row
        self.col = col
        self.color = color
        self.king = False  # Becomes True if piece reaches opponent’s end
        self.explosive = explosive  # True for explosive pieces
        self.calc_pos()  # Set pixel coordinates

    # Calculate pixel coordinates for rendering (center of square)
    def calc_pos(self):
        self.x = self.col * SQUARE_SIZE + SQUARE_SIZE // 2
        self.y = self.row * SQUARE_SIZE + SQUARE_SIZE // 2 + HEADER_HEIGHT

    # Promote piece to king (when it reaches opponent’s end)
    def make_king(self):
        self.king = True

    # Draw the piece on the window at given or default position
    def draw(self, win, x=None, y=None, alpha=255, scale=1.0):
        x = self.x if x is None else x
        y = self.y if y is None else y
        radius = int((SQUARE_SIZE // 2 - self.PADDING) * scale)
        # Create a surface for the piece with transparency
        s = pygame.Surface((radius * 2 + 10, radius * 2 + 10), pygame.SRCALPHA)
        # Draw shadow and gradient-filled circle
        pygame.draw.circle(
            s, (40, 40, 40, alpha), (radius + 7, radius + 7), radius + self.OUTLINE
        )
        base_color = self.color if self.color == RED else (0, 100, 200)
        for r in range(radius, 0, -1):
            t = r / radius
            color = tuple(int(c * t + 255 * (1 - t)) for c in base_color) + (alpha,)
            pygame.draw.circle(s, color, (radius + 5, radius + 5), r)
        # Draw crown for king pieces
        if self.king:
            crown_img = pygame.Surface((radius * 1.6, radius * 1.2), pygame.SRCALPHA)
            points = [
                (0, crown_img.get_height()),
                (crown_img.get_width() // 4, crown_img.get_height() // 2),
                (crown_img.get_width() // 2, crown_img.get_height()),
                (3 * crown_img.get_width() // 4, crown_img.get_height() // 2),
                (crown_img.get_width(), crown_img.get_height()),
            ]
            pygame.draw.polygon(crown_img, KING_BORDER, points)
            s.blit(
                crown_img,
                (
                    radius + 5 - crown_img.get_width() // 2,
                    radius + 5 - crown_img.get_height() // 2,
                ),
            )
        # Draw pulsing border and asterisk for explosive pieces
        if self.explosive:
            t = pygame.time.get_ticks() % 600 / 600
            pulse_radius = radius + 5 + int(2 * t)
            pygame.draw.circle(
                s, EXPLOSIVE_BORDER, (radius + 5, radius + 5), pulse_radius, 3
            )
            font = pygame.font.SysFont("Arial", radius)
            text = font.render("*", True, (255, 100, 0))
            s.blit(
                text,
                (
                    radius + 5 - text.get_width() // 2,
                    radius + 5 - text.get_height() // 2,
                ),
            )
        # Render the piece on the window
        win.blit(s, (x - radius - 5, y - radius - 5))

    # Animate piece movement to a new position with smooth easing
    def animate_to(self, win, end_row, end_col, board, turn, duration=500):
        start_x, start_y = self.x, self.y
        end_x = end_col * SQUARE_SIZE + SQUARE_SIZE // 2
        end_y = end_row * SQUARE_SIZE + SQUARE_SIZE // 2 + HEADER_HEIGHT
        start_time = pygame.time.get_ticks()
        clock = pygame.time.Clock()
        while pygame.time.get_ticks() - start_time < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            # Use ease-in-out quadratic for smooth movement
            t = (pygame.time.get_ticks() - start_time) / duration
            t_ease = 3 * t * t - 2 * t * t * t  # Ease-in-out quadratic
            x = start_x + (end_x - start_x) * t_ease
            y = start_y + (end_y - start_y) * t_ease
            scale = 1.0 + 0.1 * math.sin(t * math.pi)  # Slight scale effect
            win.fill(BACKGROUND)
            board.draw(win, turn)
            self.draw(win, x, y, scale=scale)
            pygame.display.update()
            clock.tick(60)
            pygame.time.wait(5)
        # Final redraw at destination
        win.fill(BACKGROUND)
        board.draw(win, turn)
        self.draw(win)
        pygame.display.update()


# Board class: Manages the game board and piece interactions
class Board:
    # Initialize the board with pieces and game state
    def __init__(self):
        self.board = []  # 8x8 list of Piece objects or None
        self.red_left = self.blue_left = 12  # Initial piece counts
        self.red_kings = self.blue_kings = 0  # King counts
        self.red_explosions = self.blue_explosions = 3  # Explosion limits
        self.create_board()

    # Draw the checkers board with alternating squares
    def draw_squares(self, win):
        win.fill(BACKGROUND)
        for row in range(ROWS):
            for col in range(row % 2, COLS, 2):
                pygame.draw.rect(
                    win,
                    WHITE,
                    (
                        col * SQUARE_SIZE,
                        row * SQUARE_SIZE + HEADER_HEIGHT,
                        SQUARE_SIZE,
                        SQUARE_SIZE,
                    ),
                )
        pygame.draw.rect(
            win, (150, 150, 150), (0, HEADER_HEIGHT, WIDTH, HEIGHT - HEADER_HEIGHT), 2
        )

    # Set up initial piece positions and assign explosive pieces
    def create_board(self):
        for row in range(ROWS):
            self.board.append([])
            for col in range(COLS):
                if (row + col) % 2 == 1:  # Place pieces on dark squares
                    if row < 3:
                        self.board[row].append(Piece(row, col, BLUE))
                    elif row > 4:
                        self.board[row].append(Piece(row, col, RED))
                    else:
                        self.board[row].append(None)
                else:
                    self.board[row].append(None)
        # Assign one random explosive piece per starting row
        for color, rows in [(RED, range(5, 8)), (BLUE, range(0, 3))]:
            for row in rows:
                valid_cols = [
                    c
                    for c in range(COLS)
                    if self.board[row][c] and self.board[row][c].color == color
                ]
                if valid_cols:
                    col = random.choice(valid_cols)
                    self.board[row][col].explosive = True

    # Draw the board, pieces, and turn indicator
    def draw(self, win, turn):
        self.draw_squares(win)
        font = pygame.font.SysFont("Arial", 18, bold=True)
        turn_text = font.render(
            f"{'Red' if turn == RED else 'Blue'}'s Turn", True, WHITE
        )
        win.blit(
            turn_text,
            (
                WIDTH // 2 - turn_text.get_width() // 2,
                HEADER_HEIGHT // 2 - turn_text.get_height() // 2,
            ),
        )
        for row in self.board:
            for piece in row:
                if piece:
                    piece.draw(win)

    # Get piece at given row, col, or None if out of bounds
    def get_piece(self, row, col):
        if 0 <= row < ROWS and 0 <= col < COLS:
            return self.board[row][col]
        return None

    # Move a piece to a new position and handle king promotion
    def move(self, piece, row, col):
        if piece:
            self.board[piece.row][piece.col], self.board[row][col] = None, piece
            piece.row, piece.col = row, col
            piece.calc_pos()
            if not piece.king:
                if row == 0 and piece.color == RED:
                    piece.make_king()
                    self.red_kings += 1
                elif row == ROWS - 1 and piece.color == BLUE:
                    piece.make_king()
                    self.blue_kings += 1

    # Remove pieces from the board and update counts
    def remove(self, pieces, win, turn):
        for piece in pieces:
            if piece and self.board[piece.row][piece.col]:
                self.board[piece.row][piece.col] = None
                if piece.color == RED:
                    self.red_left -= 1
                else:
                    self.blue_left -= 1
        win.fill(BACKGROUND)
        self.draw(win, turn)
        pygame.display.update()

    # Get valid moves for a piece (moves, captures, or explosion)
    def get_valid_moves(self, piece, captures_only=False):
        moves = {}
        left = piece.col - 1
        right = piece.col + 1
        row = piece.row
        # Check moves/captures in allowed directions (based on color or king)
        if piece.color == RED or piece.king:
            moves.update(
                self._traverse_left(
                    row - 1,
                    max(row - 3, -1),
                    -1,
                    piece.color,
                    left,
                    captures_only=captures_only,
                )
            )
            moves.update(
                self._traverse_right(
                    row - 1,
                    max(row - 3, -1),
                    -1,
                    piece.color,
                    right,
                    captures_only=captures_only,
                )
            )
        if piece.color == BLUE or piece.king:
            moves.update(
                self._traverse_left(
                    row + 1,
                    min(row + 3, ROWS),
                    1,
                    piece.color,
                    left,
                    captures_only=captures_only,
                )
            )
            moves.update(
                self._traverse_right(
                    row + 1,
                    min(row + 3, ROWS),
                    1,
                    piece.color,
                    right,
                    captures_only=captures_only,
                )
            )
        # Add explosion option for explosive pieces if not capture-only
        if (
            not captures_only
            and piece.explosive
            and (
                (piece.color == RED and self.red_explosions > 0)
                or (piece.color == BLUE and self.blue_explosions > 0)
            )
        ):
            moves[("explode",)] = []
        return moves

    # Helper: Check left-diagonal moves/captures (updated for multi-capture)
    def _traverse_left(
        self, start, stop, step, color, left, skipped=None, captures_only=False
    ):
        if skipped is None:
            skipped = []
        moves = {}
        last = []
        for r in range(start, stop, step):
            if left < 0:
                break
            current = self.board[r][left]
            if current is None:
                if captures_only and not last:
                    break
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, left)] = last + skipped
                else:
                    moves[(r, left)] = last
                if last:  # If capture, check for multi-capture
                    new_skipped = last + skipped
                    row = max(r - 3, -1) if step == -1 else min(r + 3, ROWS)
                    moves.update(
                        self._traverse_left(
                            r + step,
                            row,
                            step,
                            color,
                            left - 1,
                            skipped=new_skipped,
                            captures_only=captures_only,
                        )
                    )
                    moves.update(
                        self._traverse_right(
                            r + step,
                            row,
                            step,
                            color,
                            left + 1,
                            skipped=new_skipped,
                            captures_only=captures_only,
                        )
                    )
                break
            elif current.color == color:
                break
            else:
                last = [current]
            left -= 1
        return moves

    # Helper: Check right-diagonal moves/captures (updated for multi-capture)
    def _traverse_right(
        self, start, stop, step, color, right, skipped=None, captures_only=False
    ):
        if skipped is None:
            skipped = []
        moves = {}
        last = []
        for r in range(start, stop, step):
            if right >= COLS:
                break
            current = self.board[r][right]
            if current is None:
                if captures_only and not last:
                    break
                if skipped and not last:
                    break
                elif skipped:
                    moves[(r, right)] = last + skipped
                else:
                    moves[(r, right)] = last
                if last:  # If capture, check for multi-capture
                    new_skipped = last + skipped
                    row = max(r - 3, -1) if step == -1 else min(r + 3, ROWS)
                    moves.update(
                        self._traverse_left(
                            r + step,
                            row,
                            step,
                            color,
                            right - 1,
                            skipped=new_skipped,
                            captures_only=captures_only,
                        )
                    )
                    moves.update(
                        self._traverse_right(
                            r + step,
                            row,
                            step,
                            color,
                            right + 1,
                            skipped=new_skipped,
                            captures_only=captures_only,
                        )
                    )
                break
            elif current.color == color:
                break
            else:
                last = [current]
            right += 1
        return moves

    # Get all pieces of a given color, ensuring valid board state
    def get_all_pieces(self, color):
        pieces = []
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece and piece.color == color and piece == self.get_piece(row, col):
                    pieces.append(piece)
        return pieces

    # Get all possible moves for a color (prioritizes captures)
    def get_all_moves(self, color):
        moves = []
        capture_moves = []
        explosion_limit = self.red_explosions if color == RED else self.blue_explosions
        explosion_threshold = 2  # Require 2+ opponent pieces for explosion
        # Check for capture moves first
        for piece in self.get_all_pieces(color):
            valid_moves = self.get_valid_moves(piece, captures_only=True)
            for move, skip in valid_moves.items():
                if move != ("explode",):
                    capture_moves.append((piece, move, skip, "move"))
        # If captures exist, use them; otherwise, include non-capture moves and explosions
        if capture_moves:
            moves.extend(capture_moves)
        else:
            for piece in self.get_all_pieces(color):
                valid_moves = self.get_valid_moves(piece)
                for move, skip in valid_moves.items():
                    if move != ("explode",):
                        moves.append((piece, move, skip, "move"))
                    elif move == ("explode",) and explosion_limit > 0:
                        # Validate piece exists and is explosive on current board
                        current_piece = self.get_piece(piece.row, piece.col)
                        if (
                            current_piece
                            and current_piece.explosive
                            and current_piece.color == color
                        ):
                            temp_board = self.copy()
                            temp_piece = temp_board.get_piece(piece.row, piece.col)
                            if (
                                temp_piece
                                and temp_piece.explosive
                                and temp_piece.color == color
                            ):
                                explosion_score, _, _ = temp_board.evaluate_explosion(
                                    temp_piece, color
                                )
                                if explosion_score >= explosion_threshold:
                                    moves.append((piece, None, [], "explode"))
        return moves

    # Evaluate board state for AI (higher score favors Red)
    def evaluate(self):
        score = 2.5 * (self.red_left - self.blue_left) + 1.0 * (
            self.red_kings - self.blue_kings
        )
        red_explosive = sum(1 for piece in self.get_all_pieces(RED) if piece.explosive)
        blue_explosive = sum(
            1 for piece in self.get_all_pieces(BLUE) if piece.explosive
        )
        score += 0.5 * (blue_explosive - red_explosive)
        if self.blue_left <= 2:
            score -= 10
        if self.red_left <= 2:
            score += 10
        return score

    # Evaluate the impact of an explosion for AI
    def evaluate_explosion(self, piece, color):
        red_removed = 0
        blue_removed = 0
        exploded_pieces = set()
        chain_explosions = [(piece.row, piece.col)]
        king_removed = 0
        all_to_remove = set()
        # Simulate chain reaction
        while chain_explosions:
            current_row, current_col = chain_explosions.pop(0)
            if (current_row, current_col) in exploded_pieces:
                continue
            current_piece = self.get_piece(current_row, current_col)
            if not current_piece:
                continue
            exploded_pieces.add((current_row, current_col))
            all_to_remove.add((current_row, current_col))
            if current_piece.color == RED:
                red_removed += 1
                if current_piece.king:
                    king_removed += 1
            else:
                blue_removed += 1
                if current_piece.king:
                    king_removed += 1
            # Check 3x3 grid for adjacent pieces
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = current_row + dr, current_col + dc
                    if 0 <= r < ROWS and 0 <= c < COLS:
                        adjacent = self.get_piece(r, c)
                        if adjacent and (r, c) not in all_to_remove:
                            all_to_remove.add((r, c))
                            if adjacent.color == RED:
                                red_removed += 1
                                if adjacent.king:
                                    king_removed += 1
                            else:
                                blue_removed += 1
                                if adjacent.king:
                                    king_removed += 1
                            if adjacent.explosive and (r, c) not in exploded_pieces:
                                chain_explosions.append((r, c))
        # Calculate score based on pieces removed
        score = (
            4 * red_removed - 3 * blue_removed + 3 * king_removed
            if color == BLUE
            else 4 * blue_removed - 3 * red_removed + 3 * king_removed
        )
        opponent_left = (
            self.blue_left - blue_removed
            if color == RED
            else self.red_left - red_removed
        )
        if opponent_left <= 2:
            score += 20
        return score, red_removed, blue_removed

    # Handle explosion: Remove 3x3 grid and trigger chain reactions
    def explode(self, piece, color, win, turn):
        if (color == RED and self.red_explosions <= 0) or (
            color == BLUE and self.blue_explosions <= 0
        ):
            return False
        # Validate initial piece exists and is explosive
        initial_piece = self.get_piece(piece.row, piece.col)
        if (
            not initial_piece
            or not initial_piece.explosive
            or initial_piece.color != color
        ):
            return False

        chain_explosions = [(piece.row, piece.col)]  # Queue for chain reactions
        exploded_pieces = set()  # Track exploded pieces
        all_to_remove = set()  # Track coordinates to remove
        clock = pygame.time.Clock()

        # Process each explosion in the chain
        while chain_explosions:
            current_row, current_col = chain_explosions.pop(0)
            if (current_row, current_col) in exploded_pieces:
                continue
            current_piece = self.get_piece(current_row, current_col)
            if (
                not current_piece
                or not current_piece.explosive
                or current_piece.color != color
            ):
                continue

            exploded_pieces.add((current_row, current_col))
            # Collect all coordinates in 3x3 grid
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = current_row + dr, current_col + dc
                    if 0 <= r < ROWS and 0 <= c < COLS:
                        all_to_remove.add((r, c))
                        # Check for adjacent explosive pieces
                        adjacent = self.get_piece(r, c)
                        if (
                            adjacent
                            and adjacent.explosive
                            and (r, c) not in exploded_pieces
                            and adjacent.color == color
                        ):
                            chain_explosions.append((r, c))

            # Animate explosion (green expanding circle, 300ms)
            start_time = pygame.time.get_ticks()
            center_x = current_col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = current_row * SQUARE_SIZE + SQUARE_SIZE // 2 + HEADER_HEIGHT
            while pygame.time.get_ticks() - start_time < 300:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                t = (pygame.time.get_ticks() - start_time) / 300
                win.fill(BACKGROUND)
                self.draw(win, turn)
                radius = int(SQUARE_SIZE * t)
                alpha = int(255 * (1 - t))
                pygame.draw.circle(
                    win, (*EXPLOSIVE_BORDER, alpha), (center_x, center_y), radius
                )
                pygame.display.update()
                clock.tick(60)
                pygame.time.wait(5)

            # Redraw board to show current state
            win.fill(BACKGROUND)
            self.draw(win, turn)
            pygame.display.update()
            pygame.time.wait(200)

        # Remove all pieces in collected coordinates after animations
        for row, col in all_to_remove:
            piece = self.get_piece(row, col)
            if piece:
                self.board[row][col] = None
                if piece.color == RED:
                    self.red_left -= 1
                else:
                    self.blue_left -= 1

        # Redraw board after removals
        win.fill(BACKGROUND)
        self.draw(win, turn)
        pygame.display.update()

        # Decrease explosion limit
        if color == RED:
            self.red_explosions -= 1
        else:
            self.blue_explosions -= 1
        return True

    # Create a deep copy of the board for AI evaluation
    def copy(self):
        new_board = Board()
        new_board.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        for row in range(ROWS):
            for col in range(COLS):
                piece = self.board[row][col]
                if piece:
                    new_piece = Piece(
                        piece.row, piece.col, piece.color, piece.explosive
                    )
                    new_piece.king = piece.king
                    new_board.board[row][col] = new_piece
        new_board.red_left = self.red_left
        new_board.blue_left = self.blue_left
        new_board.red_kings = self.red_kings
        new_board.blue_kings = self.blue_kings
        new_board.red_explosions = self.red_explosions
        new_board.blue_explosions = self.blue_explosions
        return new_board


# AI: Minimax algorithm with alpha-beta pruning (updated for multi-capture)
def minimax(board, depth, max_player, alpha, beta):
    if depth == 0 or board.red_left == 0 or board.blue_left == 0:
        return board.evaluate(), None
    color = RED if max_player else BLUE
    best_value = float("-inf") if max_player else float("inf")
    best_move = None
    # Evaluate all possible moves
    for piece, move, skipped, action_type in board.get_all_moves(color):
        temp_board = board.copy()
        temp_piece = temp_board.get_piece(piece.row, piece.col)
        if not temp_piece:
            continue
        if action_type == "move":
            temp_board.move(temp_piece, move[0], move[1])
            if skipped:
                temp_board.remove(skipped, WIN, color)
            value = temp_board.evaluate()
            # Check for multi-capture
            new_valid = temp_board.get_valid_moves(temp_piece, captures_only=True)
            if new_valid and skipped and depth > 1:
                sub_value, _ = minimax(temp_board, depth - 1, max_player, alpha, beta)
                value = max(value, sub_value) if max_player else min(value, sub_value)
        else:  # Explosion
            explosion_score, red_removed, blue_removed = temp_board.evaluate_explosion(
                temp_piece, color
            )
            temp_board.explode(temp_piece, color, WIN, color)
            value = temp_board.evaluate() + explosion_score * 0.5
            opponent_left = (
                temp_board.blue_left if color == RED else temp_board.red_left
            )
            if opponent_left <= 2:
                value += 50
        # Update best move and prune
        if max_player:
            if value > best_value:
                best_value = value
                best_move = (piece, move, skipped, action_type)
            alpha = max(alpha, best_value)
        else:
            if value < best_value:
                best_value = value
                best_move = (piece, move, skipped, action_type)
            beta = min(beta, best_value)
        if beta <= alpha:
            break
    return best_value, best_move


# Draw yellow circles to indicate valid moves
def draw_valid_moves(win, moves):
    for move in moves:
        if isinstance(move, tuple) and len(move) == 2:
            row, col = move
            center = (
                col * SQUARE_SIZE + SQUARE_SIZE // 2,
                row * SQUARE_SIZE + SQUARE_SIZE // 2 + HEADER_HEIGHT,
            )
            pygame.draw.circle(win, YELLOW, center, 12)


# Display winner message and wait for click
def show_winner(win, message, color):
    font = pygame.font.SysFont("Arial", 72, bold=True)
    text = font.render(message, True, color)
    win.blit(
        text, (WIDTH // 2 - text.get_width() // 2, HEIGHT // 2 - text.get_height() // 2)
    )
    pygame.display.update()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                waiting = False
        pygame.time.wait(100)


# Check if the game is over (no pieces left)
def check_game_over(board, win):
    if board.red_left == 0:
        show_winner(win, "BLUE WINS!", BLUE)
        return True
    if board.blue_left == 0:
        show_winner(win, "RED WINS!", RED)
        return True
    return False


# Convert mouse click to board coordinates
def get_row_col_from_mouse(pos):
    x, y = pos
    if y < HEADER_HEIGHT or x < 0 or x >= WIDTH or y >= HEIGHT:
        return None, None
    row = (y - HEADER_HEIGHT) // SQUARE_SIZE
    col = x // SQUARE_SIZE
    if 0 <= row < ROWS and 0 <= col < COLS:
        return row, col
    return None, None


# Main game loop: Handles player input and AI turns
def main():
    global turn_count
    run = True
    clock = pygame.time.Clock()
    board = Board()
    selected = None  # Currently selected piece
    valid_moves = {}  # Valid moves for selected piece
    turn = RED  # Start with Red (player)
    must_continue = False  # True if player must continue capturing
    animating = False  # True during animations
    while run:
        clock.tick(60)
        # AI turn (Blue): Process moves or explosions
        if turn == BLUE and not animating:
            while True:  # Loop for multi-capture
                _, best_move = minimax(board, 4, False, float("-inf"), float("inf"))
                if not best_move:
                    break
                piece, move, skipped, action_type = best_move
                current_piece = board.get_piece(piece.row, piece.col)
                if not current_piece or current_piece.color != BLUE:
                    break
                animating = True
                if action_type == "move":
                    piece.animate_to(WIN, move[0], move[1], board, turn)
                    board.move(piece, move[0], move[1])
                    if skipped:
                        board.remove(skipped, WIN, turn)
                    new_valid = board.get_valid_moves(piece, captures_only=True)
                    if new_valid and skipped:  # Continue if more captures
                        must_continue = True
                        selected = piece
                        valid_moves = new_valid
                    else:
                        must_continue = False
                        selected = None
                        valid_moves = {}
                        turn = RED
                        turn_count += 1
                        break
                else:  # Explosion
                    if current_piece.explosive:
                        board.explode(piece, BLUE, WIN, turn)
                        selected = None
                        valid_moves = {}
                        turn = RED
                        turn_count += 1
                    break
                animating = False
                pygame.event.clear()
                WIN.fill(BACKGROUND)
                board.draw(WIN, turn)
                pygame.display.update()
            animating = False
            if check_game_over(board, WIN):
                run = False
            WIN.fill(BACKGROUND)
            board.draw(WIN, turn)
            pygame.display.update()
            continue
        # Draw the board and UI
        if not animating:
            WIN.fill(BACKGROUND)
            board.draw(WIN, turn)
            if selected:
                t = pygame.time.get_ticks() % 500 / 500
                scale = 1 + 0.1 * t
                s = pygame.Surface(
                    (SQUARE_SIZE * scale, SQUARE_SIZE * scale), pygame.SRCALPHA
                )
                glow_radius = int(SQUARE_SIZE * 0.6)
                for r in range(glow_radius, 0, -1):
                    alpha = int(80 * (r / glow_radius))
                    pygame.draw.circle(
                        s,
                        (*HIGHLIGHT, alpha),
                        (SQUARE_SIZE * scale / 2, SQUARE_SIZE * scale / 2),
                        r,
                    )
                offset = (SQUARE_SIZE * scale - SQUARE_SIZE) / 2
                WIN.blit(
                    s,
                    (
                        selected.col * SQUARE_SIZE - offset,
                        selected.row * SQUARE_SIZE + HEADER_HEIGHT - offset,
                    ),
                )
                draw_valid_moves(WIN, valid_moves.keys())
            pygame.display.update()
        # Handle player input (Red)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN and not animating and turn == RED:
                pos = pygame.mouse.get_pos()
                row, col = get_row_col_from_mouse(pos)
                if row is not None and col is not None:
                    piece = board.get_piece(row, col)
                    has_captures = any(
                        len(skip) > 0
                        for _, _, skip, atype in board.get_all_moves(turn)
                        if atype == "move"
                    )
                    # Right-click: Trigger explosion (if no captures available)
                    if (
                        event.button == 3
                        and piece
                        and piece.color == turn
                        and piece.explosive
                        and not has_captures
                    ):
                        valid_moves = board.get_valid_moves(
                            piece, captures_only=has_captures
                        )
                        if ("explode",) in valid_moves:
                            animating = True
                            board.explode(piece, turn, WIN, turn)
                            animating = False
                            selected = None
                            valid_moves = {}
                            must_continue = False
                            turn = BLUE
                            turn_count += 1
                            pygame.event.clear()
                    # Left-click: Select piece or move
                    elif event.button == 1:
                        if not must_continue:
                            if piece and piece.color == turn:
                                valid_moves = board.get_valid_moves(
                                    piece, captures_only=has_captures
                                )
                                selected = piece
                            elif selected and (row, col) in valid_moves:
                                animating = True
                                selected.animate_to(WIN, row, col, board, turn)
                                board.move(selected, row, col)
                                if valid_moves.get((row, col), []):
                                    board.remove(valid_moves[(row, col)], WIN, turn)
                                new_valid = board.get_valid_moves(
                                    selected, captures_only=True
                                )
                                if new_valid and valid_moves.get(
                                    (row, col), []
                                ):  # Multi-capture
                                    valid_moves = new_valid
                                    must_continue = True
                                else:
                                    must_continue = False
                                    selected = None
                                    valid_moves = {}
                                    turn = BLUE
                                    turn_count += 1
                                animating = False
                                pygame.event.clear()
                            else:
                                selected = None
                                valid_moves = {}
                        elif selected and (row, col) in valid_moves:  # Continue capture
                            animating = True
                            selected.animate_to(WIN, row, col, board, turn)
                            board.move(selected, row, col)
                            if valid_moves.get((row, col), []):
                                board.remove(valid_moves[(row, col)], WIN, turn)
                            new_valid = board.get_valid_moves(
                                selected, captures_only=True
                            )
                            if new_valid and valid_moves.get(
                                (row, col), []
                            ):  # Multi-capture
                                valid_moves = new_valid
                                must_continue = True
                            else:
                                must_continue = False
                                selected = None
                                valid_moves = {}
                                turn = BLUE
                                turn_count += 1
                            animating = False
                            pygame.event.clear()
        if check_game_over(board, WIN):
            run = False
    pygame.quit()


if __name__ == "__main__":
    main()
