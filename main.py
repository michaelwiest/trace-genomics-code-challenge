import numpy as np
import argparse

# Helper function for handling boolean argparse arguments from
# stackoverflow here: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_center(garden_array):
    '''
    Function for finding the center of the an input array that can be
    either odd or even dimensioned.
    '''

    rows, cols = garden_array.shape

    # If we are even set the appropriate bounds for our sub array
    # This will only have size 1.
    if rows % 2 == 1:
        row_min = int(rows / 2)
        row_max = int(rows / 2) + 1
    # On the other hand if it is even the bounds for the sub array make it of
    # size two.
    else:
        row_min = int(rows / 2) - 1
        row_max = int(rows / 2) + 1

    # Do the same for the columns.
    if cols % 2 == 1:
        col_min = int(cols / 2)
        col_max = int(cols / 2) + 1

    else:
        col_min = int(cols / 2) - 1
        col_max = int(cols / 2) + 1

    # Find the maximum of our center array.
    center_array = garden_array[row_min: row_max,
                                col_min: col_max]

    # Unravel the index corresponding to the center.
    indices = np.argmax(center_array)
    row_sub_max_index, col_sub_max_index = np.unravel_index(indices,
                                                            center_array.shape)

    # Convert sub-array indices.
    row_start = row_min + row_sub_max_index
    col_start = col_min + col_sub_max_index

    return row_start, col_start

def get_next_move_indices(current_row, current_col, garden_array,
                          row_movement=1,
                          col_movement=1):
    '''
    Function given the garden matrix and the current position it will return
    where the index of the next move is. I began to implement moving more than
    one space at a time (row_movement, col_movement), but focused on other
    stuff instead. The portion where I pad the sub-array could certainly
    be cleaner but I did what I had in the time provided.
    '''
    
    # Get the bounds for the sub-matrix
    row_lower = max(current_row - row_movement, 0)
    row_upper = min(current_row + row_movement + 1, garden_array.shape[0])
    col_lower = max(current_col - col_movement, 0)
    col_upper = min(current_col + col_movement + 1, garden_array.shape[1])

    sub_array = garden_array[row_lower:row_upper,
                             col_lower:col_upper].copy()

    # If our sub-matrix is not square, then we need to pad it with zeros
    # indicating invalid move directions (outside the garden)
    if current_row - row_movement < 0:
        sub_array = np.concatenate((np.zeros((1, sub_array.shape[1])),
                                   sub_array), axis=0)
    if current_row + row_movement + 1 > garden_array.shape[0]:
        sub_array = np.concatenate((sub_array,
                                    np.zeros((1, sub_array.shape[1]))), axis=0)

    if current_col - col_movement < 0:
        sub_array = np.concatenate((np.zeros((sub_array.shape[0], 1)),
                                   sub_array), axis=1)
    if current_col + col_movement + 1 > garden_array.shape[1]:
        sub_array = np.concatenate((sub_array,
                                    np.zeros((sub_array.shape[0], 1))), axis=1)


    # Set the corners of the movement matrix as invalid. Because can't move
    # diagonally. This doesn't support variable step sizes.
    sub_array[-1, -1] = 0
    sub_array[0, -1] = 0
    sub_array[0, 0] = 0
    sub_array[-1, 0] = 0

    # Check if there are no remaining moves.
    if sub_array.sum() == 0:
        return None
    # Get the best move to take.
    else:
        argmax_index = np.argmax(sub_array)
        row_max_index, col_max_index = np.unravel_index(argmax_index,
                                                        (sub_array.shape))

    # Need to shift our indices by one.
    next_row = current_row + row_max_index - 1
    next_col = current_col + col_max_index - 1

    return (next_row, next_col)


def eat(garden_array_orig, verbose=False):
    '''
    Main function of the assignment. The bunny will traverse the supplied
    garden until it has no valid moves left (all neighboring zeros)
    '''
    # Copy the garden so that if you used it elsewhere in the code.
    garden_array = garden_array_orig.copy()
    # Counter to return
    total_eaten = 0

    start_row, start_col = get_center(garden_array)
    if verbose:
        print('\nCenter is at: {}, {}\n'.format(start_row, start_col))

    # Increment how many eaten and mark this square as checked.
    total_eaten += garden_array[start_row, start_col]
    garden_array[start_row, start_col] = 0
    if verbose:
        print(garden_array)

    # Get the initial next move.
    next_move_indices = get_next_move_indices(start_row, start_col, garden_array)

    # Iteratively check all of the spaces until you are done. This could maybe
    # have been done recursively but I did it iteratively. Works well enough.
    while next_move_indices is not None:
        nr, nc = next_move_indices
        if verbose:
            print('Next move is: {}, {}'.format(nr, nc))

        # Increment how many eaten and mark this square as checked.
        total_eaten += garden_array[nr, nc]
        garden_array[nr, nc] = 0
        if verbose:
            print(garden_array)
        next_move_indices = get_next_move_indices(nr, nc, garden_array)

    return total_eaten


def make_random_garden(row, col, max_int=5):
    '''
    Helper function to instantiate a garden matrix.
    '''
    if row is None:
        row = np.random.randint(low=1, high=10)
    if col is None:
        col = np.random.randint(low=1, high=10)
    garden = np.random.randint(max_int, size=(row, col))
    return garden


def main():

    # Some arguments for generating a random matrix.
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", type=str2bool, nargs='?',
                        const=True, default='False',
                        help="Print the update step each time you move.")
    parser.add_argument("-r", "--row", type=int,
                        help="Number of rows in input.")
    parser.add_argument("-c", "--col", type=int,
                        help="Number of cols in input.")
    parser.add_argument("-m", "--max", type=int,
                        help="Max integer value in the garden.",
                        default=5)
    args = parser.parse_args()

    # Generate a random garden matrix
    row = args.row
    col = args.col
    garden = make_random_garden(row, col, args.max)

    # Display the garden we will traverse.
    print('Original garden is:')
    print(garden)

    eaten = eat(garden, verbose=args.verbose)
    print('Ate: {} carrots'.format(eaten))


if __name__ == '__main__':
    main()
