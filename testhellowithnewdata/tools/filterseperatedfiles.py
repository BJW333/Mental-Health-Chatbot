import json

user_file_path = '/Users/blakeweiss/Desktop/hello/seperatedfilestxt/seperateduser.txt'
bot_file_path = '/Users/blakeweiss/Desktop/hello/seperatedfilestxt/seperatedbot.txt'

# Function to process lines by removing a specific prefix
def process_lines(lines, prefix):
    return [line.replace(prefix, '') for line in lines if line.startswith(prefix)]

# Read, process, and write for user file
with open(user_file_path, 'r') as file:
    user_lines = file.readlines()

user_lines = process_lines(user_lines, 'User: ')

with open(user_file_path, 'w') as file:
    for line in user_lines:
        file.write(line)

# Read, process, and write for bot file
with open(bot_file_path, 'r') as file:
    bot_lines = file.readlines()

bot_lines = process_lines(bot_lines, 'Bot: ')

with open(bot_file_path, 'w') as file:
    for line in bot_lines:
        file.write(line)

   
   
   
   