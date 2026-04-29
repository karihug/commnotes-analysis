# Navigate to the directory containing your code
cd path/to/final_code

# Create a new folder for publication code
mkdir publication_code

# Copy the relevant code files (replace 'script1.py', 'script2.py', etc. with actual file names)
cp script1.py publication_code/
cp script2.py publication_code/
# Repeat for all relevant files

# If you have data files, copy them as well
cp data_file.csv publication_code/
# Repeat for all relevant data files

# Optionally, create a requirements.txt file
echo "numpy" >> publication_code/requirements.txt
echo "matplotlib" >> publication_code/requirements.txt
# Add any other dependencies

# Create a README file
echo "# Publication Code" > publication_code/README.md
echo "This folder contains the code and files used to generate figures for publication." >> publication_code/README.md
# Add more details as necessary