import logging
file_name = 'testing'
logging.basicConfig(filename=file_name, level=logging.INFO)

if __name__ == '__main__':
    logging.info("how are you")