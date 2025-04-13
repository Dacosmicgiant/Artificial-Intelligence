# Cryptarithmetic Puzzle: SEND + MORE = MONEY

import itertools

def crypto_arithmetic():
    letters = 'SENDMORY' # Unique letters
    # Try all unique assignments of 8 digits (0-9) to the 8 letters
    for perm in itertools.permutations(range(10), len(letters)):
        # Assign digits based on the current permutation
        s, e, n, d, m, o, r, y = perm

        # Constraint: Leading digits cannot be zero
        if s == 0 or m == 0:
            continue # Skip this permutation

        # Calculate numerical values
        send = s*1000 + e*100 + n*10 + d
        more = m*1000 + o*100 + r*10 + e
        money = m*10000 + o*1000 + n*100 + e*10 + y

        # Constraint: Check if the arithmetic holds true
        if send + more == money:
            # Solution found!
            return {'SEND': send, 'MORE': more, 'MONEY': money}

    return None # No solution found (shouldn't happen for this puzzle)

print("CryptoArithmetic solution:", crypto_arithmetic())
# Expected Output: CryptoArithmetic solution: {'SEND': 9567, 'MORE': 1085, 'MONEY': 10652}

# Map Coloring

def is_valid_color(state, region, color, neighbors):
    # Check all neighbors of the current region
    for neighbor in neighbors[region]:
        # If a neighbor exists in the current state and has the same color
        if state.get(neighbor) == color:
            return False # Invalid assignment
    return True # Color is valid for this region

def map_coloring(regions, colors, neighbors):
    # Inner recursive backtracking function
    def backtrack(state):
        # Base case: All regions are colored
        if len(state) == len(regions):
            return state # Solution found

        # Select the next unassigned region
        region = [r for r in regions if r not in state][0]

        # Try assigning each available color
        for color in colors:
            # Check if the color is valid for this region (doesn't conflict with neighbors)
            if is_valid_color(state, region, color, neighbors):
                # Assign the color
                state[region] = color
                # Recursively try to color the rest of the map
                result = backtrack(state)
                # If recursive call succeeded, return the solution
                if result:
                    return result
                # If recursive call failed, backtrack: remove the assignment
                del state[region]

        # If no color worked for this region, return None (triggering backtrack)
        return None

    # Start the backtracking process with an empty state
    return backtrack({})

# Australian states example
regions = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
colors = ['Red', 'Green', 'Blue']
neighbors = {
    'WA': ['NT', 'SA'], 'NT': ['WA', 'SA', 'Q'], 'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q': ['NT', 'SA', 'NSW'], 'NSW': ['Q', 'SA', 'V'], 'V': ['SA', 'NSW'], 'T': []
}

print("Map Coloring solution:", map_coloring(regions, colors, neighbors))
# Example Possible Output: Map Coloring solution: {'WA': 'Red', 'NT': 'Green', 'SA': 'Blue', 'Q': 'Red', 'NSW': 'Green', 'V': 'Red', 'T': 'Red'}
# (Note: The exact colors assigned might differ depending on iteration order, but the constraints will be met)

# N-Queens Problem

def n_queens(n):
    # Helper function to check if placing queen at (row, col) is safe
    def is_safe(board, row, col):
        # Check previous rows (0 to row-1)
        for i in range(row):
            # Check column conflict OR diagonal conflict
            if board[i] == col or abs(board[i] - col) == abs(i - row):
                return False # Unsafe
        return True # Safe

    # Recursive backtracking solver
    def solve(board, row):
        # Base case: All N queens placed successfully
        if row == n:
            # Return a list containing a copy of the solution board
            return [board.copy()]

        solutions = [] # List to store solutions found from this point
        # Try placing queen in each column of the current row
        for col in range(n):
            # Check if placing queen at (row, col) is safe
            if is_safe(board, row, col):
                # Place the queen
                board[row] = col
                # Recursively call solve for the next row
                # Add any solutions found from the recursive call
                solutions += solve(board, row + 1)
                # No explicit backtrack needed here (board[row] will be overwritten
                # or function returns after loop if no solutions found)

        # Return all solutions found starting from this 'row' state
        return solutions

    # Start solving from row 0 with an empty board representation
    # board[i] = -1 means no queen in row i yet
    return solve([-1] * n, 0)

# Solve for 8-Queens and print the first solution found
print("One solution for 8-Queens:", n_queens(8)[0])
# Example Possible Output: One solution for 8-Queens: [0, 4, 7, 5, 2, 6, 1, 3]
# (This means queen in row 0 is in col 0, row 1 in col 4, etc.)