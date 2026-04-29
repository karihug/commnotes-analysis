# Step 1: Navigate to the directory containing your code
cd path/to/your/project

# Step 2: Create a new folder for publication code
mkdir publication_code

# Step 3: Copy the relevant files
# Replace 'source_file1', 'source_file2', etc. with actual file names
cp final_code/your_script.py publication_code/
cp final_code/another_script.R publication_code/
cp final_code/data/your_data.csv publication_code/data/
# Add more cp commands as needed for all relevant files

# Step 4: Create a requirements file if necessary
# For Python, you can generate it using pip freeze
pip freeze > publication_code/requirements.txt

# Step 5: Create a README file
echo "This folder contains the code and data used to generate the figures in final_code/figures." > publication_code/README.md
echo "Instructions on how to run the code:" >> publication_code/README.md
echo "1. Install the required packages listed in requirements.txt." >> publication_code/README.md
echo "2. Run the scripts in the order specified." >> publication_code/README.md