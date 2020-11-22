# DSCMR-with-continual-learning-on-CUB

Download the data with embeddings from https://iitkgpacin-my.sharepoint.com/:u:/g/personal/vg_sravan_iitkgp_ac_in/ETNLWfd1ZTlPiNKGwcTfVowB6yYpk5Uq7fqB71Xu8nD_5g?e=O6NOdM<br/>

Extract and place it with main directory.<br/>

Data is arranged in the following way.<br/>

Data  ------------> changed_data -------> preprocessed_data<br/>
      ------------> images (images of CUB)<br/>
      ------------> texts (preprocessed char-CNN-RNN text embeddings for birds from https://github.com/hanzhanggit/StackGAN/tree/master/Data)<br/>
      
If there is some problem in arrangement, change the path variables accordingly

Running Code <br/>
(Step 1 and 2 are to be run while running the code for the first time)<br/>
Step 1: Inside the main directory type<br/>
        >> python data_creation.py<br/>
Step 2: Inside the main directory after step 1, type<br/>
        >> python preprocess.py<br/>
Step 3: Inside the main directory after step 1 and 2, type<br/>
        >> python main.py<br/>
        
Make sure to store the output for easier access.<br/>
Wait for the code to complete and results can be seen in Results folder(created dynamically).<br/>

Any doubts ??
ping me at vg.sravan@gmail.com
        
