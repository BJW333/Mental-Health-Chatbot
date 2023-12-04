import os

yesorno = input("are you sure you want to trim the lines to 150 yes or no:")
if yesorno == "no":
    print('quitting')
    exit()
if yesorno == "yes":
    print('triming')
    def trim_file(filename, line_count=150):
        """
        Trims the file to the desired number of lines.
        """
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Get only the first line_count lines
        trimmed_lines = lines[:line_count]

        with open(filename, 'w') as file:
            file.writelines(trimmed_lines)

    # Trim both files to 200 lines
    trim_file("/Users/blakeweiss/Desktop/hello/conversations.txt")
    trim_file("/Users/blakeweiss/Desktop/hello/answers.txt")

    print("Files have been trimmed to 150 lines.")
