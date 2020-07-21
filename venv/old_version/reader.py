class Reader:

    def __init__(self, filename='example.txt'):
        self.filename = filename

    def read(self):
        try:
            file = open(self.filename)
            return file.read()
        except IOError:
            return "File not found"


def main():
    x = Reader('example.txt')
    print(x.read())


if __name__ == "__main__":
    main()
