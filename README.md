# DSCMR-with-continual-learning-on-CUB

Download the data with embeddings from https://iitkgpacin-my.sharepoint.com/:u:/g/personal/vg_sravan_iitkgp_ac_in/ETNLWfd1ZTlPiNKGwcTfVowB6yYpk5Uq7fqB71Xu8nD_5g?e=O6NOdM

Extract and place it with main directory.

Data is arranged in the following way.

Data  ------------> changed_data -------> preprocessed_data

      ------------> images (images of CUB)
      
      ------------> texts (preprocessed char-CNN-RNN text embeddings for birds from https://github.com/hanzhanggit/StackGAN/tree/master/Data)
      
If there is some problem in arrangement, change the path variables accordingly

Running Code 
(Step 1 and 2 are to be run while running the code for the first time)
Step 1: Inside the main directory type
        >> python data_creation.py
Step 2: Inside the main directory after step 1, type
        >> python preprocess.py
Step 3: Inside the main directory after step 1 and 2, type
        >> python main.py
        
Make sure to store the output for easier access.
Wait for the code to complete and results can be seen in Results folder(created dynamically).

Any doubts ?? 
ping me at vg.sravan@gmail.com
        
