# Define the path of the original file
original_file_path = '/Users/blakeweiss/Desktop/hello/data/conversation_data.txt' # Replace with your file path

# Define paths for the new files
user_file_path = '/Users/blakeweiss/Desktop/hello/seperatedfilestxt/seperateduser.txt' # Replace with your desired path
bot_file_path = '/Users/blakeweiss/Desktop/hello/seperatedfilestxt/seperatedbot.txt'  # Replace with your desired path

# Read the original file and process each line
with open(original_file_path, 'r') as file:
    lines = file.readlines()

# Open two new files for writing
with open(user_file_path, 'w') as user_file, open(bot_file_path, 'w') as bot_file:
    for line in lines:
        # Check if the line starts with 'User:' and write to user_file
        if line.startswith('User:'):
            user_file.write(line)
        # Check if the line starts with 'Bot:' and write to bot_file
        elif line.startswith('Bot:'):
            bot_file.write(line)
