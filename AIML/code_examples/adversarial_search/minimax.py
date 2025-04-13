def minimax(board, depth, is_maximizing):
    winner = check_winner(board)
    if winner is not None:
        return winner - depth if winner == 1 else winner + depth if winner == -1 else winner # Adjust score based on depth

    if is_maximizing:
        best_score = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'X'
                    score = minimax(board, depth + 1, False)
                    board[i][j] = ''
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = 'O'
                    score = minimax(board, depth + 1, True)
                    board[i][j] = ''
                    best_score = min(score, best_score)
        return best_score

def check_winner(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != '':
            if row[0] == 'X':
                return 1
            else:
                return -1

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != '':
            if board[0][col] == 'X':
                return 1
            else:
                return -1

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '':
        if board[0][0] == 'X':
            return 1
        else:
            return -1

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '':
        if board[0][2] == 'X':
            return 1
        else:
            return -1

    # Check for draw
    for row in board:
        for cell in row:
            if cell == '':
                return None  # Game still in progress

    return 0  # It's a draw

def find_best_move(board):
    best_score = float('-inf')
    best_move = None
    for i in range(3):
        for j in range(3):
            if board[i][j] == '':
                board[i][j] = 'X'
                score = minimax(board, 0, False) # Start with depth 0 for the initial call
                board[i][j] = ''
                if score > best_score:
                    best_score = score
                    best_move = (i, j)
    return best_move

# Initial empty board
initial_board = [
    ['', '', ''],
    ['', '', ''],
    ['', '', '']
]

print("Initial Board:")
for row in initial_board:
    print(row)
print("\nThinking...")

# Find the best move for 'X'
best_move = find_best_move(initial_board)

if best_move:
    row, col = best_move
    initial_board[row][col] = 'X'
    print("\nBest move for X:", best_move)
    print("Board after X's move:")
    for row in initial_board:
        print(row)

    # Now, let's evaluate the final outcome from this initial move
    final_score = minimax(initial_board, 1, False) # Start depth at 1 as one move has been made

    print("\nPredicted outcome of the game (from X's perspective):", final_score)

    if final_score > 0:
        print("Conclusion: X will eventually win (assuming optimal play from both sides).")
    elif final_score < 0:
        print("Conclusion: O will eventually win (assuming optimal play from both sides).")
    else:
        print("Conclusion: The game will end in a draw (assuming optimal play from both sides).")
else:
    print("No moves possible on the board.")