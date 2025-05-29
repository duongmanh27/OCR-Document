AI-Based Document Information Extraction
  - Developed a system to extract data from document images such as ID cards, land certificates, and driver’s licenses.<br>
  - Used the ViNTERN-1B model combined with OCR and image processing techniques for accurate recognition and extraction of key information.<br>
  - Implemented automatic classification and storage of extracted data by document type and name, enabling easy retrieval and management.<br>
  - Built a Flask API to accept image inputs and return results in JSON format for seamless integration.<br>
  - Deployed on PyTorch with GPU acceleration on Google Colab, optimizing processing performance.<br>

To run on Google Colab :<br>
First : download the requirements file. <br>
```
!pip install requirements.txt 
```
Then :<br>
  - Register an account at https://dashboard.ngrok.com<br>
  - Go to "Auth" in the dashboard to get the command line in the form:<br>
    ```
    22P1jNxxxxxxxxxxxxxxx
    ```
  - Create a new code cell on google colab and run<br>
    ```
    !ngrok config add-authtoken 22P1jNxxxxxxxxxxxxxxx
    ```
  - And run run_on_gg_colab.py in this<br>
After you run the app.py and run_local.py file on your computer and change the image path<br>
