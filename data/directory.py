import os

print("hi")

def print_directory_contents(path):
    for root, dirs, files in os.walk(path):
        print(f"\nFolder: {root}")
        
        if files:
            print("Files:")
            for file in files[:3]:
                print(f"  - {file}")
            
            if len(files) > 3:
                print(f"  ...and {len(files)} total files in this folder.")
        else:
            print("No files in this folder.")

# Use current working directory
current_path = os.getcwd()
print_directory_contents(current_path)
