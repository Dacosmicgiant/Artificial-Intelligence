def alpha_beta(board, depth, alpha, beta, is_maximizing, maximizing_player='X', minimizing_player='O'):
    winner = check_winner(board)
    if winner is not None:
        if winner == 1 and maximizing_player == 'X':
            return 1 - depth
        elif winner == 1 and maximizing_player == 'O':
            return 1 - depth
        elif winner == -1 and minimizing_player == 'O':
            return -1 + depth
        elif winner == -1 and minimizing_player == 'X':
            return -1 + depth
        else:
            return 0

    if is_maximizing:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = maximizing_player
                    eval = alpha_beta(board, depth + 1, alpha, beta, False, maximizing_player, minimizing_player)
                    board[i][j] = ''
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == '':
                    board[i][j] = minimizing_player
                    eval = alpha_beta(board, depth + 1, alpha, beta, True, maximizing_player, minimizing_player)
                    board[i][j] = ''
                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break
                if beta <= alpha:
                    break
        return min_eval

def check_winner(board):
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != '':
            return 1 if row[0] == 'X' else -1

    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] and board[0][col] != '':
            return 1 if board[0][col] == 'X' else -1

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] != '':
        return 1 if board[0][0] == 'X' else -1

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] != '':
        return 1 if board[0][2] == 'X' else -1

    # Check for draw
    for row in board:
        for cell in row:
            if cell == '':
                return None  # Game still in progress

    return 0  # It's a draw

def find_best_move_alpha_beta(board, player):
    best_score = float('-inf') if player == 'X' else float('inf')
    best_move = None
    alpha = float('-inf')
    beta = float('inf')

    for i in range(3):
        for j in range(3):
            if board[i][j] == '':
                board[i][j] = player
                if player == 'X':
                    score = alpha_beta(board, 0, alpha, beta, False, 'X', 'O')
                    if score > best_score:
                        best_score = score
                        best_move = (i, j)
                    alpha = max(alpha, score)
                else:
                    score = alpha_beta(board, 0, alpha, beta, True, 'O', 'X')
                    if score < best_score:
                        best_score = score
                        best_move = (i, j)
                    beta = min(beta, score)
                board[i][j] = ''
    return best_move

def print_board_formatted(board):
    print("---------")
    for row in board:
        print("| " + " | ".join(row) + " |")
        print("---------")

# Initialize the game
game_board = [
    ['', '', ''],
    ['', '', ''],
    ['', '', '']
]
current_player = 'X'
game_over = False

print("Starting the Tic-Tac-Toe game (AI vs AI using Alpha-Beta Pruning):")
print_board_formatted(game_board)

while not game_over:
    print(f"\n{current_player}'s turn:")
    best_move = find_best_move_alpha_beta(game_board, current_player)

    if best_move:
        row, col = best_move
        game_board[row][col] = current_player
        print_board_formatted(game_board)

        winner = check_winner(game_board)
        if winner == 1:
            print("\nGame Over! X wins!")
            game_over = True
        elif winner == -1:
            print("\nGame Over! O wins!")
            game_over = True
        elif all('' not in row for row in game_board):
            print("\nGame Over! It's a draw!")
            game_over = True
        else:
            # Switch to the other player
            current_player = 'O' if current_player == 'X' else 'X'
    else:
        print("\nNo more moves possible (this should not happen with optimal play).")
        game_over = True