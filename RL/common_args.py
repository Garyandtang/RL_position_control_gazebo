import argparse


def get_args():
    parser = argparse.ArgumentParser(description='RL Position Control')
    # environment
    parser.add_argument('--box_size', default=6, type=int, help='size of the box')
    parser.add_argument('--box_no', default=100, type=int, help='no of the box')
    parser.add_argument('--item_type', default=2, type=int, help='size of the item type')
    parser.add_argument('--buffer_size', default=2, type=int, help='size of the buffer')
    parser.add_argument('--random_init', default=True, type=bool, help='init the state randomly or not')
    parser.add_argument('--state_no', default=100, type=int, help='no of state')
    parser.add_argument('--state_init_file', default=file_name, type=str, help='state init file name')

    # DQN

    args = parser.parse_args()
    return args