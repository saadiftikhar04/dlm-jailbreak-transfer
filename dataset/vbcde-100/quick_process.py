import pandas as pd

input_file = "./vbcde-100_reference.csv"
df = pd.read_csv(input_file)
### In "ciphered_prompt" column, replace all "an image in 1536x1024." with "a photo-realistic image in 1536x1024."
df['ciphered_prompt'] = df['ciphered_prompt'].str.replace("an image in 1536x1024.", "a photo-realistic image in 1536x1024.", regex=False).replace("You must You must", "You must", regex=False)
### Swap the "ciphered_prompt" and "keywords" columns
df['ciphered_prompt'], df['keywords'] = df['keywords'], df['ciphered_prompt']

### Write to the original file
df.to_csv(input_file, index=False)