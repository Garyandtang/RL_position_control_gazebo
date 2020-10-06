import logging
file_name = 'testing'
logging.basicConfig(filename=file_name, level=logging.INFO)
def main():
   for i in range(3):
       logging.info(i)

if __name__ == '__main__':
    main()