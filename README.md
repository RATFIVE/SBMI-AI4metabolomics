# SBMI - AI4metabolomics

Analyzing an organism's metabolism is crucial for understanding and monitoring diseases and treatments. Magnetic resonance imaging (MRI) is the only technique to measure the metabolism noninvasively, in vivo, and in vitro.

With this applications you can detect and measure the abundance of prespecified substances of NMR spectra. 




## Run the App
### Notes
This app is not a fully debugged application. Some cases may lead to bugs. These bugs should not apply to the fitting process itself. If you run into an error, please restart the app. It is important to quit the app via the terminal by pressing CTRL + C.

It is recommended to use python version >= 3.10 for this application. 

### Install Git
To use git, please install it with following this [link](https://git-scm.com/downloads).


### Download the GitHub repository

1. Download this directory and put it into a place on your system where you like. You can download the directory using either git:

```bash
git clone --depth 1 https://github.com/RATFIVE/SBMI-AI4metabolomics.git
```
or download the directory from the github repository by  

    1. open this link: https://github.com/RATFIVE/SBMI-AI4metabolomics,
    2. press on the green code button and 
    3. 'Download ZIP'. Depending on your internet speed it will take a certain time because of past development code.

### Install Virtual Envirnoment
You can install all the libaries in a seperate python enviroment.

1. Go into the app directory:

```bash

cd path_of_your_download_dir\SBMI-AI4metabolomics-main\app
```


2. Create a Virtual Environment in the app dir:
```bash
python -m venv .SBMI
```

3. Activate the Virtual Environment
```bash
.SBMI\Scripts\activate
```

4. Use this command to install the required python packages

```bash
pip install -r requirements.txt
```


5. Start the app
```bash
streamlit run app.py --server.port=8501
```
The app should open now in the browser

You can always access the app by the url. Paste this in your browser(for example Firefox) 
```bash
http://localhost:8501
```

### Quit the app
To quit the app go back to the running terminal and press `CTRL + C`. This is important. If this is not done more and more instances would be created and will eventually fill up the memory(RAM). 
